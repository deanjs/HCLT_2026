"""모델 로딩 래퍼 — GQA group 확인 포함(CLAUDE.md §3 GQA 주의).

torch/transformers는 지연 임포트한다. 조건 스키마·결과 저장은 이 무거운 의존성
없이도 동작해야 하므로(스키마 검증, 노트북 셀 구성 등), 실제 모델을 만질 때만
임포트가 일어난다. 실제 사용은 step A부터.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .conditions import ModelSpec


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
        input_ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        torch.manual_seed(seed)
        do_sample = temperature and temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=bool(do_sample),
                temperature=temperature if do_sample else None,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)


def load_model(spec: ModelSpec, device_map: Optional[str] = "auto") -> ModelHandle:
    """HF 허브에서 instruct 모델을 로드한다. torch/transformers 지연 임포트.

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
