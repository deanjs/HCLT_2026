"""모델 로딩 래퍼 — GQA group 확인 포함(CLAUDE.md §3 GQA 주의).

torch/transformers는 지연 임포트한다. 조건 스키마·결과 저장은 이 무거운 의존성
없이도 동작해야 하므로(스키마 검증, 노트북 셀 구성 등), 실제 모델을 만질 때만
임포트가 일어난다. 실제 사용은 step A부터.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .conditions import ModelSpec
from .attention_probe import cosine, find_char_spans, locate_token_spans, span_metrics
from .intervention import align_name_groups, align_name_tokens, preference_score, recovery_rate


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
        notation_spans: Optional[dict[str, Any]] = None,
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
        # 지침 안의 '표기 지시어 토큰'(camelCase/snake_case)을 통째 지침과 별도로 잡는다.
        # 값 두 종류를 허용한다:
        #   · str            : 프롬프트 전체에서 그 단어의 모든 등장을 찾는다(초판, 규칙문+후보열거 합침).
        #   · [(a,b), ...]   : **지침 문장 내부 offset**. 지침이 프롬프트에 박힌 시작점을 더해 재기준
        #                      (규칙문/후보열거 분리 — instruction_notation_spans). 지침을 못 찾으면 빈 구간.
        if notation_spans:
            inst_char = char_spans.get("instruction") or (
                find_char_spans(prompt_text, [instruction_text]) if instruction_text else []
            )
            inst_start = inst_char[0][0] if inst_char else None
            for span_name, spec in notation_spans.items():
                if isinstance(spec, str):
                    char_spans[span_name] = find_char_spans(prompt_text, [spec])
                elif inst_start is None:
                    char_spans[span_name] = []
                else:
                    char_spans[span_name] = [(inst_start + a, inst_start + b) for a, b in spec]
        spans = locate_token_spans(offsets, char_spans)

        # 조용한 실패 금지: 구간을 못 찾으면 지표가 None으로 저장되고, 집계에서 `.get(키, 0)`을
        # 쓰면 "0을 봤다"로 오독된다. chat 템플릿이 지침 문장을 변형하는 모델에서 실제로 생긴다.
        empty = [name for name, idxs in spans.items() if not idxs]
        if empty:
            raise ValueError(
                f"프롬프트에서 찾지 못한 관측 구간: {empty}. "
                "chat 템플릿이 문장을 변형했거나 이름/지시어 문자열이 어긋났다. "
                "빈 구간을 그대로 저장하면 '주목하지 않았다'로 오독되므로 실행을 중단한다."
            )

        # 3) forward — 어텐션·KV value 확보
        with torch.no_grad():
            out = self.model(**enc, output_attentions=True, use_cache=True)
        attentions = out.attentions            # tuple[L] each [1, Hq, seq, seq]
        values = _value_cache(out.past_key_values)  # list[L] each [1, Hkv, seq, d]

        # 4) 층별 집계 — 마지막 query 행과 ‖v‖만 리스트로 옮겨 순수 함수 호출.
        #    code_camel/code_snake는 per-token 상세도 뽑아 따로 저장(밑줄/형태 마커 분석용).
        detail_spans = ("code_camel", "code_snake")
        per_layer: dict[int, dict] = {}
        token_detail: dict[str, dict] = {sp: {"tokens": [], "per_layer": {}} for sp in detail_spans}
        for layer in range(len(attentions)):
            attn_last = attentions[layer][0, :, -1, :].float().cpu().tolist()   # [Hq, seq]
            vnorm_kv = values[layer][0].float().norm(dim=-1).cpu().tolist()     # [Hkv, seq]
            m = span_metrics(
                attn_last, vnorm_kv, group_size=gqa.group_size,
                spans=spans, detail=detail_spans,
            )
            for sp in detail_spans:
                toks = m[sp].pop("tokens", [])   # 그룹 값만 per_layer에 남기고 상세는 분리
                if layer == 0:                   # 토큰 텍스트는 층 무관 → 한 번만
                    token_detail[sp]["tokens"] = [
                        {"t": d["t"], "text": prompt_text[offsets[d["t"]][0]:offsets[d["t"]][1]]}
                        for d in toks
                    ]
                token_detail[sp]["per_layer"][str(layer)] = {
                    "a": [d["a"] for d in toks],
                    "av": [d["av"] for d in toks],
                    "v": [d["v"] for d in toks],
                }
            per_layer[layer] = m

        return {
            "per_layer": per_layer,
            "token_detail": token_detail,
            "spans": {k: len(v) for k, v in spans.items()},
            "seq_len": seq_len,
            "gqa": {
                "num_attention_heads": gqa.num_attention_heads,
                "num_key_value_heads": gqa.num_key_value_heads,
                "group_size": gqa.group_size,
            },
        }

    def _preference_context(
        self,
        viol_messages: list[dict[str, str]],
        comp_messages: list[dict[str, str]],
        *,
        viol_names: list[str],
        donor_names: list[str],
        candidate_compliant: str,
        candidate_violation: str,
        donor_messages: Optional[list[dict[str, str]]] = None,
        forced_prefix: str = "def ",
        token_unit: str = "all",
        span_kind: str = "def_name",
    ) -> dict[str, Any]:
        """개입 측정의 공통 준비 — 캐시·후보·정렬쌍·채점 함수·기준 S(계획서 §구현).

        단일 층(step C)과 전 층 스윕(step 1)이 **같은 설정을 공유**한다(CLAUDE.md §3,
        로직 중복 금지). 여기서 위반/준수/공여 프롬프트를 각 한 번씩만 forward하므로,
        스윕에서 층을 바꿀 때 선행 프롬프트를 재계산하지 않는다(무료 티어 비용 절감).

        span_kind: 치환 대상 토큰을 찾는 방식.
          "def_name" = 코드 함수 이름(`def <nm>(`의 이름 부분, stepC/step1).
          "literal"  = 문자열 nm 그대로의 첫 등장(step5 지침 지시어 "camelCase").

        반환 dict:
          viol_cache/comp_cache/donor_cache, viol_last, pairs, skipped,
          S(cache,last)→준수 선호 점수, s_base(위반), s_clean(깨끗).
        """
        import torch

        tok = self.tokenizer

        def prefix(messages):
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ) + forced_prefix
            enc = tok(text, return_tensors="pt", return_offsets_mapping=True,
                      add_special_tokens=False)
            offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
            ids = enc["input_ids"].to(self.model.device)
            return ids, text, offsets

        def name_positions(text, offsets, names):
            return _locate_target_tokens(text, offsets, names, span_kind)

        def cand_ids(s):
            return tok(s, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(
                self.model.device)

        viol_ids, viol_text, viol_off = prefix(viol_messages)
        comp_ids, comp_text, comp_off = prefix(comp_messages)
        cc, cv = cand_ids(candidate_compliant), cand_ids(candidate_violation)

        # base 캐시 — prefix[:-1]을 forward (마지막 토큰은 채점 forward에서 넣는다)
        with torch.no_grad():
            viol_cache = self.model(viol_ids[:, :-1], use_cache=True).past_key_values
            comp_cache = self.model(comp_ids[:, :-1], use_cache=True).past_key_values
        viol_last, comp_last = viol_ids[:, -1:], comp_ids[:, -1:]

        # 공여(donor) 소스: 기본은 준수(clean) 캐시. 무관 코드 통제면 별도 forward.
        if donor_messages is None:
            donor_cache, donor_text, donor_off = comp_cache, comp_text, comp_off
        else:
            d_ids, donor_text, donor_off = prefix(donor_messages)
            with torch.no_grad():
                donor_cache = self.model(d_ids[:, :-1], use_cache=True).past_key_values

        vpos = name_positions(viol_text, viol_off, viol_names)
        dpos = name_positions(donor_text, donor_off, donor_names)
        if token_unit == "mean":
            # mean-pool: 공여 토큰 평균을 위반 이름 전 자리에 브로드캐스트(개수 불일치 허용).
            groups, skipped = align_name_groups(vpos, dpos)
            pairs = None
            n_sub = sum(len(vp) for vp, _ in groups)   # 덮어쓴 위반 토큰 자리 수
        else:
            pairs, skipped = align_name_tokens(vpos, dpos, mode=token_unit)
            groups = None
            n_sub = len(pairs)

        if n_sub == 0:
            # 조용한 실패 금지: 한 자리도 안 덮으면 S_int == S_base가 되어
            # "개입했는데 효과가 없다"와 수치가 완전히 같아진다(회복률 0.0). 구분되어야 한다.
            raise ValueError(
                "치환할 토큰 자리를 하나도 찾지 못했다 — 개입이 실제로 일어나지 않는다. "
                f"(정렬 단위={token_unit!r}, 구간 종류={span_kind!r}, "
                f"이름 {len(viol_names)}개 중 {len(skipped)}개 정렬 실패). "
                "이름·지시어를 프롬프트에서 못 찾았거나, token_unit='all'에서 토큰 수가 어긋난 경우다."
            )

        def logp_candidate(cache, last_tok, cand):
            # [마지막 프롬프트 토큰 + cand[:-1]]을 캐시로 forward → 후보 토큰 logP 합.
            # 캐시를 복제해 forward가 원본을 늘리지 않게 한다(층 스윕에서 재사용).
            c = _clone_cache(cache)
            inp = torch.cat([last_tok, cand[:-1].unsqueeze(0)], dim=1)
            with torch.no_grad():
                logits = self.model(inp, past_key_values=c, use_cache=True).logits[0]
            lp = torch.log_softmax(logits.float(), dim=-1)
            return float(sum(lp[k, cand[k]].item() for k in range(cand.shape[0])))

        def S(cache, last_tok):
            return preference_score(logp_candidate(cache, last_tok, cc),
                                    logp_candidate(cache, last_tok, cv))

        return {
            "viol_cache": viol_cache,
            "comp_cache": comp_cache,
            "donor_cache": donor_cache,
            "viol_last": viol_last,
            "pairs": pairs,
            "groups": groups,
            "n_substituted": n_sub,
            "skipped": skipped,
            "S": S,
            "s_base": S(viol_cache, viol_last),
            "s_clean": S(comp_cache, comp_last),
        }

    def _score_layer_kind(self, ctx: dict[str, Any], work_cache: Any, layer: int, kind: str) -> float:
        """work_cache의 `layer`만 공여값으로 치환해 S_int를 잰 뒤 **원상 복구**한다.

        치환은 KV group 단위(캐시 [B, n_kv, seq, d]). 스윕은 하나의 work_cache를
        층·kind마다 편집→측정→복구로 재사용하므로 층당 깊은 복사가 필요 없다.
        S 내부에서 다시 복제해 forward하므로 work_cache는 측정 중에도 오염되지 않고,
        측정 후 백업으로 되돌려 다음 층이 깨끗한 위반 상태에서 시작한다.

        token_unit="mean"이면 pairs 대신 groups로 mean-pool 치환한다: 공여 위치들의 KV를
        평균 내(한 벡터) 위반 이름의 모든 위치에 브로드캐스트(개수 불일치 허용, 스킵 없음).
        """
        if kind not in ("key", "value", "key_value"):
            # 조용한 실패 금지: 미지원 kind(예: attn_amplify)는 아무것도 안 바꾸고
            # recovery=0.0을 돌려주어 "개입해도 안 돌아온다"처럼 보인다. 반드시 터뜨린다.
            raise NotImplementedError(
                f"치환으로 구현되지 않은 개입 종류: {kind!r}. "
                "KV 캐시 편집으로 표현할 수 있는 것은 key/value/key_value뿐이다. "
                "어텐션 증폭(attn_amplify)은 forward 훅이 필요하며 아직 구현되지 않았다."
            )
        ek, ev = _cache_kv(work_cache, layer)
        dk, dv = _cache_kv(ctx["donor_cache"], layer)
        edit_k = kind in ("key", "key_value")
        edit_v = kind in ("value", "key_value")

        if ctx.get("groups") is not None:                     # ── mean-pool 경로 ──
            bk: dict[int, Any] = {}
            bv: dict[int, Any] = {}
            for vps, dps in ctx["groups"]:
                if edit_k:
                    km = dk[:, :, dps, :].mean(dim=2)          # [B, n_kv, d] 공여 평균
                    for vp in vps:
                        bk[vp] = ek[:, :, vp, :].clone()
                        ek[:, :, vp, :] = km
                if edit_v:
                    vm = dv[:, :, dps, :].mean(dim=2)
                    for vp in vps:
                        bv[vp] = ev[:, :, vp, :].clone()
                        ev[:, :, vp, :] = vm
            s_int = ctx["S"](work_cache, ctx["viol_last"])
            for vp, b in bk.items():
                ek[:, :, vp, :] = b
            for vp, b in bv.items():
                ev[:, :, vp, :] = b
            return s_int

        pairs = ctx["pairs"]                                  # ── 1:1 pairs 경로 ──
        bk = {vp: ek[:, :, vp, :].clone() for vp, _ in pairs} if edit_k else {}
        bv = {vp: ev[:, :, vp, :].clone() for vp, _ in pairs} if edit_v else {}
        for vp, dp in pairs:
            if edit_k:
                ek[:, :, vp, :] = dk[:, :, dp, :]
            if edit_v:
                ev[:, :, vp, :] = dv[:, :, dp, :]
        s_int = ctx["S"](work_cache, ctx["viol_last"])
        for vp, b in bk.items():
            ek[:, :, vp, :] = b
        for vp, b in bv.items():
            ev[:, :, vp, :] = b
        return s_int

    def intervene_preference(
        self,
        viol_messages: list[dict[str, str]],
        comp_messages: list[dict[str, str]],
        *,
        viol_names: list[str],
        donor_names: list[str],
        candidate_compliant: str,
        candidate_violation: str,
        layer: int,
        kind: str = "key_value",
        donor_messages: Optional[list[dict[str, str]]] = None,
        forced_prefix: str = "def ",
        token_unit: str = "all",
        span_kind: str = "def_name",
    ) -> dict[str, Any]:
        """단일 층 KV 캐시 편집으로 준수 선호도 회복을 측정한다(step C, 방법 B).

        세 상태의 준수 선호 점수 S(clean/baseline/intervened)와 회복률을 얻는다.
        회복률 = (S_int − S_base)/(S_clean − S_base).
        kind: 'key' / 'value' / 'key_value'. 텍스트는 그대로 두고 **내부 표현만** 바꾼다.
        token_unit: 'all'(전체 토큰) / 'last'(마지막 토큰만) — 이름 토큰 정렬 단위.
        span_kind: 'def_name'(코드 이름) / 'literal'(지침 지시어, step5).
        """
        ctx = self._preference_context(
            viol_messages, comp_messages,
            viol_names=viol_names, donor_names=donor_names,
            candidate_compliant=candidate_compliant, candidate_violation=candidate_violation,
            donor_messages=donor_messages, forced_prefix=forced_prefix, token_unit=token_unit,
            span_kind=span_kind,
        )
        work = _clone_cache(ctx["viol_cache"])
        s_int = self._score_layer_kind(ctx, work, layer, kind)
        s_clean, s_base = ctx["s_clean"], ctx["s_base"]
        return {
            "S_clean": s_clean,
            "S_base": s_base,
            "S_int": s_int,
            "recovery": recovery_rate(s_clean, s_base, s_int),
            "layer": layer,
            "kind": kind,
            "n_substituted_tokens": ctx["n_substituted"],
            "skipped_names": ctx["skipped"],
        }

    def intervene_preference_sweep(
        self,
        viol_messages: list[dict[str, str]],
        comp_messages: list[dict[str, str]],
        *,
        viol_names: list[str],
        donor_names: list[str],
        candidate_compliant: str,
        candidate_violation: str,
        layers: Optional[Sequence[int]] = None,
        kinds: Sequence[str] = ("key", "value", "key_value"),
        donor_messages: Optional[list[dict[str, str]]] = None,
        forced_prefix: str = "def ",
        token_unit: str = "all",
        span_kind: str = "def_name",
    ) -> dict[str, Any]:
        """전 층 × K/V 분해 스윕(step 1) — 각 층에서 세 경로의 S_int·회복률을 모두 잰다.

        선행 프롬프트를 한 번만 forward하고(비용 절감), 하나의 work 캐시를 층·kind마다
        편집→측정→복구로 재사용한다. S_clean·S_base는 층과 무관하므로 한 번만.
        layers=None이면 전 층(0..num_layers-1). 산출은 층별 회복률 곡선 3개(kind별)와
        피크 층·K/V 경로별 기여도의 원자료가 된다(계획서 §5 Step 1).
        token_unit: 'all'(전체 토큰) / 'last'(마지막 토큰만) — 이름 토큰 정렬 단위.
        span_kind: 'def_name'(코드 이름, step1) / 'literal'(지침 지시어, step5).
        """
        if layers is None:
            layers = list(range(self.num_layers))
        ctx = self._preference_context(
            viol_messages, comp_messages,
            viol_names=viol_names, donor_names=donor_names,
            candidate_compliant=candidate_compliant, candidate_violation=candidate_violation,
            donor_messages=donor_messages, forced_prefix=forced_prefix, token_unit=token_unit,
            span_kind=span_kind,
        )
        s_clean, s_base = ctx["s_clean"], ctx["s_base"]
        work = _clone_cache(ctx["viol_cache"])
        per_layer: dict[int, dict[str, dict[str, Any]]] = {}
        for layer in layers:
            by_kind: dict[str, dict[str, Any]] = {}
            for kind in kinds:
                s_int = self._score_layer_kind(ctx, work, layer, kind)
                by_kind[kind] = {
                    "S_int": s_int,
                    "recovery": recovery_rate(s_clean, s_base, s_int),
                }
            per_layer[int(layer)] = by_kind
        return {
            "S_clean": s_clean,
            "S_base": s_base,
            "per_layer": per_layer,
            "layers": list(layers),
            "kinds": list(kinds),
            "n_substituted_tokens": ctx["n_substituted"],
            "skipped_names": ctx["skipped"],
        }

    def kv_substitution_diagnostics(
        self,
        viol_messages: list[dict[str, str]],
        comp_messages: list[dict[str, str]],
        *,
        viol_names: list[str],
        donor_names: list[str],
        candidate_compliant: str,
        candidate_violation: str,
        donor_messages: Optional[list[dict[str, str]]] = None,
        forced_prefix: str = "def ",
        token_unit: str = "mean",
        span_kind: str = "def_name",
    ) -> dict[str, Any]:
        """평균 덮어쓰기가 **Key를 Value보다 불리하게 망가뜨리는가**를 진단한다.

        왜 필요한가
        -----------
        캐시에 저장된 Key에는 **그 토큰이 있던 위치의 회전(RoPE)이 이미 적용**돼 있다.
        Value에는 위치 정보가 없다. 그런데 우리는 서로 다른 위치의 조각을 **평균**내어 덮는다.
        회전 방향이 엇갈린 벡터를 평균하면 서로 지워져 **원래보다 짧은 벡터**가 된다.
        그러면 "Key를 바꿔도 안 돌아온다"가 어텐션 경로의 성질인지, 우리가 값을 망가뜨린 탓인지
        구분할 수 없다 — 편향의 방향이 우리 결론과 같으므로 반드시 확인해야 한다.

        무엇을 재나 (층마다)
        --------------------
        ``줄어듦 = ‖조각들의 평균‖ / 조각 노름들의 평균``
          · 1에 가까움 → 조각들이 같은 방향을 봤다. 평균 내도 안 줄었다 = 문제 없음
          · 많이 작음   → 서로 지워졌다 = 쪼그라든 값을 넣고 있었다

        **Key만 재면 판단할 수 없다.** 서로 다른 토큰을 평균 내면 회전이 없어도 조금은 줄어들기
        때문이다. 그래서 **Value의 같은 값을 기준선으로 함께 잰다.** 두 값이 비슷하면 회전 탓의
        추가 손해가 없다는 뜻이고, Key 쪽만 크게 작으면 RoPE 탓이다.

        함께 남기는 것: 조각들 사이의 평균 코사인(방향이 얼마나 흩어졌나)과
        **덮을 자리와 공여 자리의 위치 차이**(위치가 어긋날수록 회전 위상도 어긋난다).

        점수를 매기지 않으므로 층 스윕보다 훨씬 가볍다 — 프롬프트 forward 몇 번이면 끝난다.
        """
        import torch

        if token_unit != "mean":
            raise ValueError(
                "이 진단은 평균 덮어쓰기(mean) 규격을 대상으로 한다 "
                f"(받음: {token_unit!r})"
            )

        ctx = self._preference_context(
            viol_messages, comp_messages,
            viol_names=viol_names, donor_names=donor_names,
            candidate_compliant=candidate_compliant, candidate_violation=candidate_violation,
            donor_messages=donor_messages, forced_prefix=forced_prefix, token_unit=token_unit,
            span_kind=span_kind,
        )
        groups = ctx["groups"]

        def shrink_and_spread(t: Any) -> tuple[float, float]:
            """t: [B, n_kv, n, d] → (줄어듦 비율, 조각들 사이 평균 코사인)."""
            mean_norm = t.mean(dim=2).norm(dim=-1)              # ‖평균‖        [B, n_kv]
            norm_mean = t.norm(dim=-1).mean(dim=2)              # 노름의 평균   [B, n_kv]
            ratio = float((mean_norm / norm_mean.clamp_min(1e-9)).mean())
            n = t.shape[2]
            if n < 2:
                return ratio, 1.0                                # 조각이 하나면 흩어질 것이 없다
            u = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            gram = torch.einsum("bknd,bkmd->bknm", u, u)         # 조각쌍 코사인
            off = (gram.sum(dim=(-1, -2)) - n) / (n * (n - 1))   # 대각선 제외 평균
            return ratio, float(off.mean())

        per_layer: dict[int, dict[str, float]] = {}
        with torch.no_grad():
            for layer in range(self.num_layers):
                dk, dv = _cache_kv(ctx["donor_cache"], layer)
                ks, kc, vs, vc = [], [], [], []
                for _vps, dps in groups:
                    r, c = shrink_and_spread(dk[:, :, dps, :]); ks.append(r); kc.append(c)
                    r, c = shrink_and_spread(dv[:, :, dps, :]); vs.append(r); vc.append(c)
                k_shrink = sum(ks) / len(ks)
                v_shrink = sum(vs) / len(vs)
                per_layer[layer] = {
                    "key_shrink": k_shrink,             # Key 줄어듦 (1이면 손해 없음)
                    "value_shrink": v_shrink,           # Value 줄어듦 (회전 없는 기준선)
                    "key_minus_value": k_shrink - v_shrink,   # 음수가 크면 Key만 손해
                    "key_piece_cosine": sum(kc) / len(kc),
                    "value_piece_cosine": sum(vc) / len(vc),
                }

        # 덮을 자리와 공여 자리의 위치 차이 — 위치가 어긋나면 회전 위상도 어긋난다.
        offsets = [ (sum(vps) / len(vps)) - (sum(dps) / len(dps)) for vps, dps in groups ]
        return {
            "per_layer": per_layer,
            "S_clean": ctx["s_clean"],
            "S_base": ctx["s_base"],
            "n_substituted_tokens": ctx["n_substituted"],
            "skipped_names": ctx["skipped"],
            "position_offsets": offsets,
            "position_offset_abs_mean": sum(abs(o) for o in offsets) / len(offsets),
            "donor_piece_counts": [len(dps) for _vps, dps in groups],
            "target_piece_counts": [len(vps) for vps, _dps in groups],
            "num_layers": self.num_layers,
        }

    def name_v_cosine_sweep(
        self,
        camel_messages: list[dict[str, str]],
        snake_messages: list[dict[str, str]],
        *,
        camel_names: list[str],
        snake_names: list[str],
        forced_prefix: str = "def ",
        token_unit: str = "all",
    ) -> dict[str, Any]:
        """전 층 v 코사인 궤적(step 1 관측 측) — 같은 이름의 camel판 v와 snake판 v 방향차.

        같은 선행을 camel/snake로만 달리 렌더한 두 프롬프트를 각각 forward해, **역할별로
        정렬한 이름 토큰 위치**에서 층별 value 벡터의 코사인 유사도를 KV head·토큰쌍
        평균으로 낸다. ‖v‖(크기)가 안 변해도(배율 1.002) 방향이 갈리는지를 본다
        (계획서 §2.4). 낮으면 "크기 같아도 다른 정보" → Value 치환 결과와 맞물린다.
        """
        import torch

        tok = self.tokenizer

        def prefix(messages):
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ) + forced_prefix
            enc = tok(text, return_tensors="pt", return_offsets_mapping=True,
                      add_special_tokens=False)
            offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
            ids = enc["input_ids"].to(self.model.device)
            return ids, text, offsets

        def name_positions(text, offsets, names):
            out = []
            for nm in names:
                spans = find_char_spans(text, [f"def {nm}("])
                if not spans:
                    out.append([]); continue
                s, e = spans[0]
                s, e = s + 4, e - 1
                out.append([ti for ti, (ts, te) in enumerate(offsets)
                            if te > ts and ts < e and te > s])
            return out

        camel_ids, camel_text, camel_off = prefix(camel_messages)
        snake_ids, snake_text, snake_off = prefix(snake_messages)
        with torch.no_grad():
            camel_vals = _value_cache(self.model(camel_ids, use_cache=True).past_key_values)
            snake_vals = _value_cache(self.model(snake_ids, use_cache=True).past_key_values)

        if token_unit == "mean":
            # 코사인은 **짝지어진 두 벡터**가 있어야 정의된다. mean-pool은 짝을 없애므로
            # 개념상 맞지 않는다. 조용히 last로 바꿔치기하면 다른 정렬 단위로 잰 값이
            # 통일 규격인 척 저장되므로, 여기서 명시적으로 거절한다(§3).
            raise ValueError(
                "v 코사인 관측은 token_unit='mean'을 지원하지 않는다 "
                "(코사인은 1:1 짝이 필요). 이 관측만 'last'로 조건을 따로 만들고, "
                "정렬 단위가 다르다는 사실을 결과 해석에 명시할 것."
            )

        # 역할별 정렬: 같은 이름의 camel 토큰 위치 ↔ snake 토큰 위치.
        pairs, skipped = align_name_tokens(
            name_positions(camel_text, camel_off, camel_names),
            name_positions(snake_text, snake_off, snake_names),
            mode=token_unit,
        )

        per_layer: dict[int, dict[str, Any]] = {}
        for layer in range(len(camel_vals)):
            cvals = camel_vals[layer][0]   # [Hkv, seq, d]
            svals = snake_vals[layer][0]
            n_kv = cvals.shape[0]
            cosines: list[float] = []
            for cp, sp in pairs:
                for k in range(n_kv):
                    u = cvals[k, cp, :].float().cpu().tolist()
                    w = svals[k, sp, :].float().cpu().tolist()
                    c = cosine(u, w)
                    if c is not None:
                        cosines.append(c)
            per_layer[int(layer)] = {
                "v_cosine": (sum(cosines) / len(cosines)) if cosines else None,
                "n": len(cosines),
            }
        return {
            "per_layer": per_layer,
            "n_pairs": len(pairs),
            "skipped_names": skipped,
        }

    def intervene_generate(
        self,
        viol_messages: list[dict[str, str]],
        *,
        viol_names: list[str],
        donor_names: list[str],
        donor_messages: list[dict[str, str]],
        layer: int,
        kind: str = "key_value",
        forced_prefix: str = "def ",
        max_new_tokens: int = 24,
        token_unit: str = "all",
        span_kind: str = "def_name",
    ) -> dict[str, Any]:
        """치환 전(baseline)·후(intervened)로 **실제 이름을 생성**한다(step C 생성 기반).

        선호 점수(intervene_preference)의 직관적 재확인 — 텍스트는 그대로 두고 L25 KV만
        준수 값으로 바꾼 뒤, `def ` 다음을 그리디로 생성해 어떤 표기의 이름이 나오는지 본다.
        donor_messages는 공여 표현의 출처 프롬프트(같은 이름 준수판 또는 무관 준수/위반 코드).
        분류(camel/snake)는 호출부에서 한다 — 여기서는 생성 텍스트만 돌려준다.
        """
        import torch

        tok = self.tokenizer

        def prefix(messages):
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ) + forced_prefix
            enc = tok(text, return_tensors="pt", return_offsets_mapping=True,
                      add_special_tokens=False)
            offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
            return enc["input_ids"].to(self.model.device), text, offsets

        def name_positions(text, offsets, names):
            return _locate_target_tokens(text, offsets, names, span_kind)

        viol_ids, viol_text, viol_off = prefix(viol_messages)
        d_ids, donor_text, donor_off = prefix(donor_messages)
        viol_last = viol_ids[:, -1:]

        with torch.no_grad():
            viol_cache = self.model(viol_ids[:, :-1], use_cache=True).past_key_values
            donor_cache = self.model(d_ids[:, :-1], use_cache=True).past_key_values

        vpos = name_positions(viol_text, viol_off, viol_names)
        dpos = name_positions(donor_text, donor_off, donor_names)
        if token_unit == "mean":
            # 선호 점수 경로(_score_layer_kind)와 **같은** mean-pool 규격을 쓴다(§3 통일).
            groups, skipped = align_name_groups(vpos, dpos)
            pairs = None
            n_sub = sum(len(vp) for vp, _ in groups)
        else:
            pairs, skipped = align_name_tokens(vpos, dpos, mode=token_unit)
            groups = None
            n_sub = len(pairs)
        if n_sub == 0:
            raise ValueError(
                "치환할 토큰 자리를 하나도 찾지 못했다 — 개입 없이 생성하는 것과 같아진다. "
                f"(정렬 단위={token_unit!r}, 구간 종류={span_kind!r})"
            )

        def greedy(cache):
            c = _clone_cache(cache)
            cur = viol_last
            gen: list[int] = []
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    out = self.model(cur, past_key_values=c, use_cache=True)
                    c = out.past_key_values
                    nxt = int(out.logits[0, -1].argmax())
                    if nxt == tok.eos_token_id:
                        break
                    gen.append(nxt)
                    cur = torch.tensor([[nxt]], device=self.model.device)
            return tok.decode(gen)

        text_baseline = greedy(viol_cache)                 # 개입 없음

        if kind not in ("key", "value", "key_value"):
            raise NotImplementedError(f"치환으로 구현되지 않은 개입 종류: {kind!r}")

        edited = _clone_cache(viol_cache)                  # 개입: layer만 공여값으로
        ek, ev = _cache_kv(edited, layer)
        dk, dv = _cache_kv(donor_cache, layer)
        edit_k = kind in ("key", "key_value")
        edit_v = kind in ("value", "key_value")
        if groups is not None:                              # mean-pool 경로
            for vps, dps in groups:
                km = dk[:, :, dps, :].mean(dim=2) if edit_k else None
                vm = dv[:, :, dps, :].mean(dim=2) if edit_v else None
                for vp in vps:
                    if edit_k:
                        ek[:, :, vp, :] = km
                    if edit_v:
                        ev[:, :, vp, :] = vm
        else:                                               # 1:1 pairs 경로
            for vp, dp in pairs:
                if edit_k:
                    ek[:, :, vp, :] = dk[:, :, dp, :]
                if edit_v:
                    ev[:, :, vp, :] = dv[:, :, dp, :]
        text_intervened = greedy(edited)

        return {
            "text_baseline": forced_prefix + text_baseline,
            "text_intervened": forced_prefix + text_intervened,
            "layer": layer,
            "kind": kind,
            "n_substituted_tokens": n_sub,
            "skipped_names": skipped,
        }


def _locate_target_tokens(text, offsets, names, span_kind: str = "def_name"):
    """치환 대상 문자열들의 토큰 인덱스 목록(이름 순)을 돌려준다.

    span_kind:
      "def_name" = `def <nm>(`에서 이름 부분만(stepC/step1 코드 이름).
      "literal"  = 문자열 nm 그대로의 **첫 등장**(step5 지침 지시어 "camelCase").
                   지침 문장의 지시어는 rule 문장이 closed 문장보다 앞서므로 첫 등장이
                   실제 지시어(rule) 위치가 된다.
    한쪽이라도 못 찾으면 그 이름은 빈 리스트 → align_name_tokens가 스킵한다.
    """
    out = []
    for nm in names:
        if span_kind == "literal":
            spans = find_char_spans(text, [nm])
            if not spans:
                out.append([]); continue
            s, e = spans[0]
        else:
            spans = find_char_spans(text, [f"def {nm}("])
            if not spans:
                out.append([]); continue
            s, e = spans[0]
            s, e = s + 4, e - 1                      # "def " 뒤 ~ "(" 앞 = 이름
        out.append([ti for ti, (ts, te) in enumerate(offsets)
                    if te > ts and ts < e and te > s])
    return out


def _cache_kv(cache: Any, layer: int):
    """layer의 (key, value) 텐서 [B, n_kv, seq, d]를 돌려준다(편집 가능한 원본 참조).

    transformers 버전마다 KV 캐시 레이아웃이 다르다:
      - 신형(4.54+): cache.layers[i].keys / .values
      - 구형: cache.key_cache[i] / cache.value_cache[i]
      - legacy tuple: cache[i][0] / cache[i][1]
    """
    if hasattr(cache, "layers"):
        lyr = cache.layers[layer]
        return lyr.keys, lyr.values
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache[layer], cache.value_cache[layer]
    return cache[layer][0], cache[layer][1]


def _clone_cache(cache: Any):
    """past_key_values 깊은 복사 — 후보마다 캐시를 재사용하되 서로 오염되지 않게.

    torch 텐서는 deepcopy가 같은 디바이스에 clone을 만든다. 내부 구조를 몰라도
    버전 무관하게 안전하다(Cache 객체의 리스트·텐서·카운터를 통째로 복제).
    """
    import copy
    return copy.deepcopy(cache)


def _value_cache(past: Any) -> list:
    """past_key_values에서 층별 value 텐서 [1, Hkv, seq, d] 목록을 꺼낸다(버전 무관)."""
    if past is None:
        raise ValueError("past_key_values가 None이다 (use_cache=True 필요)")
    if hasattr(past, "layers"):                       # 신형 DynamicCache
        return [lyr.values for lyr in past.layers]
    if hasattr(past, "value_cache"):                  # 구형 DynamicCache
        return list(past.value_cache)
    return [layer_kv[1] for layer_kv in past]         # legacy tuple


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
