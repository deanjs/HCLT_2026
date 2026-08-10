"""모델 로딩 래퍼 — GQA group 확인 포함(CLAUDE.md §3 GQA 주의).

torch/transformers는 지연 임포트한다. 조건 스키마·결과 저장은 이 무거운 의존성
없이도 동작해야 하므로(스키마 검증, 노트북 셀 구성 등), 실제 모델을 만질 때만
임포트가 일어난다. 실제 사용은 step A부터.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .conditions import ModelSpec
from .attention_probe import find_char_spans, locate_token_spans, span_metrics


@dataclass
class GQAInfo:
    """Grouped Query Attention 구성.

    Key/Value 치환은 반드시 KV group 단위로 수행해야 한다(§3). group_size는
    하나의 KV를 공유하는 query head 수다.
    """
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    @property
    def group_size(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def is_gqa(self) -> bool:
        return self.num_key_value_heads != self.num_attention_heads


@dataclass
class ModelHandle:
    """로드된 모델·토크나이저·설정을 함께 들고 다니는 핸들."""
    spec: ModelSpec
    model: Any
    tokenizer: Any
    config: Any

    @property
    def num_layers(self) -> int:
        return int(self.config.num_hidden_layers)

    def gqa_info(self) -> GQAInfo:
        cfg = self.config
        n_attn = int(cfg.num_attention_heads)
        n_kv = int(getattr(cfg, "num_key_value_heads", n_attn))
        head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // n_attn))
        return GQAInfo(n_attn, n_kv, head_dim)

    def relative_layer(self, layer: int) -> float:
        """층의 상대 위치(0~1). L25가 전체 대비 어디인지 확인용(계획서 §2.5)."""
        return layer / max(1, self.num_layers - 1)

    def chat_generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        seed: int = 0,
        temperature: float = 0.0,
    ) -> str:
        """chat 템플릿으로 메시지를 받아 새로 생성된 텍스트만 반환한다.

        step A의 순차 생성에 쓴다. temperature=0이면 그리디(재현성). torch 지연 임포트.
        """
        import torch

        tok = self.tokenizer
        # return_dict=True로 받아야 버전에 무관하게 dict(BatchEncoding)이 온다.
        # (return_tensors만 주면 버전에 따라 텐서/딕셔너리가 갈려 .shape에서 깨진다)
        enc = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        input_len = enc["input_ids"].shape[1]
        torch.manual_seed(seed)
        do_sample = bool(temperature and temperature > 0)
        with torch.no_grad():
            out = self.model.generate(
                **enc,  # input_ids + attention_mask 함께 전달 (pad 경고 방지)
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0, input_len:], skip_special_tokens=True)

    def observe_generation_query(
        self,
        messages: list[dict[str, str]],
        *,
        groups: dict[str, list[str]],
        instruction_text: Optional[str] = None,
        forced_prefix: str = "def ",
    ) -> dict[str, Any]:
        """이름을 생성하는 디코딩 시점의 query 한 행을 구간별로 관측한다(stepB).

        절차:
          1) chat 템플릿으로 프롬프트 문자열을 만들고 `forced_prefix`("def ")를 이어 붙여
             **다음 예측 토큰이 함수 이름**이 되도록 teacher-forcing한다.
          2) 그 문자열을 offset_mapping과 함께 토큰화해 **모델 입력과 offset을 일치**시킨다.
          3) output_attentions=True·use_cache=True로 1회 forward.
          4) 층별로 마지막 query 행의 어텐션 [Hq, seq]과 KV value의 ‖v‖ [Hkv, seq]만 꺼내
             span_metrics(방법 A)로 구간별 집계.

        groups: 구간 이름 → 이 구간으로 묶을 함수 이름 목록(예: code_camel / code_snake).
                이름은 `def <name>(` 형태로 찾은 뒤 **이름 부분 토큰만** 구간에 넣는다.
        instruction_text: 있으면 'instruction' 구간으로 지침 문장 전체를 잡는다.

        반환: {'per_layer': {층: {구간: {attention_weight, av_norm, v_norm, n_tokens}}},
               'spans': {구간: 토큰수}, 'seq_len', 'gqa': {...}}.

        메모리: 전체 어텐션 행렬을 뜨므로(output_attentions) 짧은 컨텍스트에만 쓴다.
        stepB 합성 프롬프트(수백 토큰)는 T4에서 감당된다. 긴 컨텍스트는 축약 경로 필요(§2.7).
        output_attentions는 eager 어텐션에서만 가중치를 돌려준다 → load_model(attn_implementation='eager').
        """
        import torch

        tok = self.tokenizer
        gqa = self.gqa_info()

        # 1) 프롬프트 문자열 + teacher-forcing 접두
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_text = prompt_text + forced_prefix

        # 2) 문자열을 그대로 토큰화(offset 포함) → 모델 입력과 offset이 정확히 일치
        enc = tok(
            prompt_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        seq_len = enc["input_ids"].shape[1]

        # 구간 문자 위치 → 토큰 인덱스. 이름은 `def <name>(`로 찾고 이름 부분만 잡는다.
        char_spans: dict[str, list[tuple[int, int]]] = {}
        for gname, names in groups.items():
            ranges: list[tuple[int, int]] = []
            for nm in names:
                for s, e in find_char_spans(prompt_text, [f"def {nm}("]):
                    ranges.append((s + 4, e - 1))   # "def " 4자 뒤 ~ "(" 앞 = 이름
            char_spans[gname] = ranges
        if instruction_text:
            char_spans["instruction"] = find_char_spans(prompt_text, [instruction_text])
        spans = locate_token_spans(offsets, char_spans)

        # 3) forward — 어텐션·KV value 확보
        with torch.no_grad():
            out = self.model(**enc, output_attentions=True, use_cache=True)
        attentions = out.attentions            # tuple[L] each [1, Hq, seq, seq]
        values = _value_cache(out.past_key_values)  # list[L] each [1, Hkv, seq, d]

        # 4) 층별 집계 — 마지막 query 행과 ‖v‖만 리스트로 옮겨 순수 함수 호출
        per_layer: dict[int, dict] = {}
        for layer in range(len(attentions)):
            attn_last = attentions[layer][0, :, -1, :].float().cpu().tolist()   # [Hq, seq]
            vnorm_kv = values[layer][0].float().norm(dim=-1).cpu().tolist()     # [Hkv, seq]
            per_layer[layer] = span_metrics(
                attn_last, vnorm_kv, group_size=gqa.group_size, spans=spans
            )

        return {
            "per_layer": per_layer,
            "spans": {k: len(v) for k, v in spans.items()},
            "seq_len": seq_len,
            "gqa": {
                "num_attention_heads": gqa.num_attention_heads,
                "num_key_value_heads": gqa.num_key_value_heads,
                "group_size": gqa.group_size,
            },
        }


def _value_cache(past: Any) -> list:
    """past_key_values에서 층별 value 텐서 [1, Hkv, seq, d] 목록을 꺼낸다.

    transformers 버전에 따라 legacy tuple 또는 Cache 객체다. 둘 다 처리한다.
    """
    if past is None:
        raise ValueError("past_key_values가 None이다 (use_cache=True 필요)")
    if hasattr(past, "value_cache"):          # DynamicCache 등
        return list(past.value_cache)
    return [layer_kv[1] for layer_kv in past]  # legacy tuple: (key, value)


def load_model(
    spec: ModelSpec,
    device_map: Optional[str] = "auto",
    attn_implementation: Optional[str] = None,
) -> ModelHandle:
    """HF 허브에서 instruct 모델을 로드한다. torch/transformers 지연 임포트.

    attn_implementation: stepB 관측(observe_generation_query)은 output_attentions로
        어텐션 가중치를 읽으므로 'eager'로 로드해야 한다(sdpa/flash는 가중치를 None으로 준다).
        생성 실험(step A)은 지정하지 않아도 된다.
    양자화를 쓰는 경우 표현 치환 정밀도 검증을 먼저 수행할 것(§5, 계획서 §5 step 2).
    """
    import torch  # noqa: F401  (dtype 매핑에 필요)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float16": "float16",
        "bfloat16": "bfloat16",
        "float32": "float32",
    }
    if spec.dtype not in dtype_map:
        raise ValueError(f"알 수 없는 dtype: {spec.dtype}")

    load_kwargs: dict[str, Any] = {
        "torch_dtype": getattr(__import__("torch"), spec.dtype),
        "device_map": device_map,
    }
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation
    if spec.quantization == "8bit":
        load_kwargs["load_in_8bit"] = True
    elif spec.quantization == "4bit":
        load_kwargs["load_in_4bit"] = True
    elif spec.quantization is not None:
        raise ValueError(f"지원하지 않는 quantization: {spec.quantization}")

    config = AutoConfig.from_pretrained(spec.name)
    tokenizer = AutoTokenizer.from_pretrained(spec.name)
    model = AutoModelForCausalLM.from_pretrained(spec.name, **load_kwargs)
    model.eval()
    return ModelHandle(spec=spec, model=model, tokenizer=tokenizer, config=config)
