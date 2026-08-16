"""step6 처방 엔진 — 값 조향(CAA식)과 Spotlight 어텐션 조향.

두 개입 모두 **교사강제 점수**(camel 선호도)를 개입 하에서 재서 회복률을 얻는다.
step3·5의 KV 치환과 달리, 여기서는 forward 훅으로 개입한다:
  - VALUE_ADD    : 그 층 잔차에 `strength × (camel−snake 방향)` 더하기 (활성값 더하기 조향[CAA/ITI]).
  - SPOTLIGHT    : 전 층·전 헤드 어텐션에서 지시어 스팬 비중을 목표치로 끌어올림
                   (Spotlight, arXiv:2505.12025, 식(2)~(5) 재구현).

torch/transformers는 지연 임포트(스키마·테스트가 무거운 의존성 없이 돌도록).
Spotlight 수식은 순수 함수 spotlight_reweight_np로 분리해 단위테스트한다(Appendix C).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence


# ─────────────────────────────────────────────────────────────────────────────
# 순수 수식 (검증용) — Spotlight 재가중, Appendix C 폐형과 일치
# ─────────────────────────────────────────────────────────────────────────────
def spotlight_reweight_np(probs, span_mask, psi_target: float):
    """Spotlight 재가중(확률 공간, 로짓 바이어스와 수학적으로 동치).

    probs      : [..., k] 소프트맥스 어텐션 가중치 (합=1)
    span_mask  : [k] bool, 지시어 스팬 열
    psi_target : 목표 비중 ψ_target ∈ (0,1)

    반환: (new_probs, psi_current, psi_new)
      게이팅: ψ_current ≥ ψ_target이면 개입 없음.
      개입 시 스팬 열에 (ψ_target/ψ_current) 배 후 재정규화 → 언더슛(식 5):
        ψ_new = ψ_target / (1 + ψ_target − ψ_current).
    """
    import numpy as np

    probs = np.asarray(probs, dtype=np.float64)
    span_mask = np.asarray(span_mask, dtype=bool)
    psi_cur = probs[..., span_mask].sum(axis=-1, keepdims=True)          # [...,1]
    apply = psi_cur < psi_target
    factor = np.where(apply, psi_target / np.clip(psi_cur, 1e-12, None), 1.0)
    new = probs.copy()
    new[..., span_mask] = probs[..., span_mask] * factor
    new = new / new.sum(axis=-1, keepdims=True)
    psi_new = new[..., span_mask].sum(axis=-1, keepdims=True)
    return new, psi_cur, psi_new


# ─────────────────────────────────────────────────────────────────────────────
# 모델 구조 접근
# ─────────────────────────────────────────────────────────────────────────────
def decoder_layers(model) -> Sequence[Any]:
    """디코더 블록 리스트(module list). Llama/Qwen/DeepSeek/StableCode 계열 공통."""
    base = model
    for attr in ("model", "transformer", "gpt_neox"):
        if hasattr(base, attr):
            base = getattr(base, attr)
            break
    for attr in ("layers", "h"):
        if hasattr(base, attr):
            return getattr(base, attr)
    raise RuntimeError("디코더 블록 리스트를 찾지 못함 (model.model.layers 예상)")


class value_add_hook:
    """forward 훅: 지정 층 블록 출력(잔차)에 `strength × vec`를 더한다.

    with value_add_hook(layer_module, vec, strength): ... (torch forward)
    vec: [hidden] 1D 텐서. 모든 위치에 브로드캐스트.
    """

    def __init__(self, layer_module, vec, strength: float):
        self.layer_module = layer_module
        self.vec = vec
        self.strength = float(strength)
        self._h = None

    def __enter__(self):
        add = self.strength * self.vec

        def fn(_mod, _inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                return (hs + add.to(hs.dtype).to(hs.device),) + tuple(out[1:])
            return out + add.to(out.dtype).to(out.device)

        self._h = self.layer_module.register_forward_hook(fn)
        return self

    def __exit__(self, *exc):
        if self._h is not None:
            self._h.remove()
            self._h = None
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Spotlight 어텐션 조향 — transformers 어텐션 함수 레지스트리로 주입
# ─────────────────────────────────────────────────────────────────────────────
class spotlight_attention:
    """전 층·전 헤드 어텐션에서 지시어 스팬 비중을 ψ_target으로 끌어올린다.

    Spotlight 식(2)~(5) 재구현. 확률 공간 재가중(Appendix C, 로짓 바이어스와 동치)으로 구현.
    - 게이팅: ψ_current ≥ ψ_target이면 개입 없음.
    - 헤드별 독립 ψ / 전 쿼리 위치 적용 (자유도는 논문 미명시 → 우리 결정).

    구현: transformers의 어텐션 함수 레지스트리에 'spotlight'를 등록하고,
    컨텍스트 동안 model.config._attn_implementation을 'spotlight'로 바꾼다. (최신 transformers)
    개입 후 스팬 비중을 self.observed_psi에 누적(측정 ①: 실제로 올랐는지 검증).
    """

    def __init__(self, model, span_positions: Sequence[int], psi_target: float, key_len: int):
        self.model = model
        self.span_positions = list(span_positions)
        self.psi_target = float(psi_target)
        self.key_len = int(key_len)
        self.observed_psi: list = []
        self._prev_impl = None
        self._fn_name = "spotlight_step6"

    def __enter__(self):
        import torch
        try:
            from transformers.masking_utils import ALL_ATTENTION_FUNCTIONS  # 최신 위치
        except Exception:
            try:
                from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "이 transformers 버전엔 ALL_ATTENTION_FUNCTIONS가 없어 Spotlight 주입 불가. "
                    "attn_implementation='eager'로 로드했는지, transformers>=4.48인지 확인."
                ) from e

        span = self.span_positions
        psi_t = self.psi_target
        sink = self.observed_psi

        def eager_repeat(x, n):
            # repeat_kv: [B, n_kv, s, d] -> [B, n_kv*n, s, d]
            if n == 1:
                return x
            b, h, s, d = x.shape
            return x[:, :, None, :, :].expand(b, h, n, s, d).reshape(b, h * n, s, d)

        def spotlight_fn(module, query, key, value, attention_mask,
                         scaling=None, dropout=0.0, **kwargs):
            groups = getattr(module, "num_key_value_groups", 1)
            k = eager_repeat(key, groups)
            v = eager_repeat(value, groups)
            if scaling is None:
                scaling = query.shape[-1] ** -0.5
            attn = torch.matmul(query, k.transpose(2, 3)) * scaling      # [B,Hq,q,k]
            if attention_mask is not None:
                attn = attn + attention_mask[..., : k.shape[-2]]
            # 정품 eager처럼 소프트맥스는 float32로 올려 계산(fp16 NaN 방지)
            probs = torch.softmax(attn, dim=-1, dtype=torch.float32)
            klen = probs.shape[-1]
            cols = [p for p in span if 0 <= p < klen]
            if cols:
                idx = torch.tensor(cols, device=probs.device)
                psi_cur = probs.index_select(-1, idx).sum(-1, keepdim=True)   # [B,Hq,q,1]
                factor = torch.where(psi_cur < psi_t, psi_t / psi_cur.clamp_min(1e-12),
                                     torch.ones_like(psi_cur))
                span_vals = probs.index_select(-1, idx) * factor
                probs = probs.index_copy(-1, idx, span_vals)
                probs = probs / probs.sum(-1, keepdim=True)
                # 측정 ①: 마지막 쿼리행(이름 생성 위치)의 스팬 비중 평균
                sink.append(float(probs.index_select(-1, idx).sum(-1)[..., -1].mean().item()))
            probs = probs.to(query.dtype)
            out = torch.matmul(probs, v)                 # [B,Hq,q,d]
            out = out.transpose(1, 2).contiguous()       # [B,q,Hq,d]
            return out, probs

        ALL_ATTENTION_FUNCTIONS[self._fn_name] = spotlight_fn
        self._prev_impl = self.model.config._attn_implementation
        self.model.config._attn_implementation = self._fn_name
        # 하위 모듈 config도 갱신(일부 버전은 모듈별 참조)
        for m in self.model.modules():
            if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
                pass  # 같은 config 객체 참조라 위 한 줄로 충분한 경우가 대부분
        return self

    def __exit__(self, *exc):
        self.model.config._attn_implementation = self._prev_impl
        return False

    def mean_observed_psi(self) -> Optional[float]:
        if not self.observed_psi:
            return None
        return sum(self.observed_psi) / len(self.observed_psi)


# ─────────────────────────────────────────────────────────────────────────────
# 조향 방향 추출 + 교사강제 점수(개입 하)
# ─────────────────────────────────────────────────────────────────────────────
def build_steer_vector(handle, samples, layer: int, forced_prefix: str = "def "):
    """잔차 공간 camel−snake 방향 (그 층). CAA식 조향 벡터.

    samples: [{'camel_messages','snake_messages','camel_names','snake_names'}, ...]
    각 표본을 forward해 층 출력 잔차를 이름 토큰 위치에서 평균, camel−snake.
    hidden_states[layer+1] = layer(0-indexed) 출력.
    """
    import torch
    from .model import _locate_target_tokens

    tok = handle.tokenizer
    dev = handle.model.device
    cam, sna = [], []
    for smp in samples:
        for mk, nk, acc in (("camel_messages", "camel_names", cam),
                            ("snake_messages", "snake_names", sna)):
            text = tok.apply_chat_template(smp[mk], tokenize=False, add_generation_prompt=True) + forced_prefix
            enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
            offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
            ids = enc["input_ids"].to(dev)
            pos = [p for lst in _locate_target_tokens(text, offsets, smp[nk], "def_name") for p in lst]
            if not pos:
                continue
            with torch.no_grad():
                hs = handle.model(ids, output_hidden_states=True).hidden_states[layer + 1][0]
            acc.append(hs[pos].float().mean(0))
    if not cam or not sna:
        raise RuntimeError("조향 벡터 추출 실패: 이름 토큰 위치를 못 찾음")
    return torch.stack(cam).mean(0) - torch.stack(sna).mean(0)     # [hidden]


def steer_preference(handle, viol_messages, comp_messages, cand_comp, cand_viol, *,
                     kind: str, layer: Optional[int] = None, strength: Optional[float] = None,
                     steer_vec=None, psi_target: Optional[float] = None,
                     span_names: Optional[Sequence[str]] = None, forced_prefix: str = "def "):
    """개입 하 camel 선호점수 회복. kind: 'none'|'value_add'|'spotlight'.

    S_clean = 천장(오염 없는 선행) · S_base = 절벽(개입 없음) · S_int = 개입 후.
    회복률 = (S_int − S_base)/(S_clean − S_base). spotlight면 개입 후 지시어 어텐션 비중도.
    """
    import torch
    from contextlib import nullcontext
    from .intervention import preference_score, recovery_rate
    from .model import _locate_target_tokens

    tok = handle.tokenizer
    dev = handle.model.device

    def prefix(messages):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + forced_prefix
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
        offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
        return enc["input_ids"].to(dev), text, offsets

    def cand_ids(s):
        return tok(s, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(dev)

    cc, cv = cand_ids(cand_comp), cand_ids(cand_viol)

    def logp(pids, cand, ctx):
        inp = torch.cat([pids, cand[:-1].unsqueeze(0)], dim=1)
        with ctx, torch.no_grad():
            logits = handle.model(inp).logits[0]
        lp = torch.log_softmax(logits.float(), dim=-1)
        plen = pids.shape[1]
        return float(sum(lp[plen - 1 + k, cand[k]].item() for k in range(cand.shape[0])))

    def S(pids, ctx_factory):
        return preference_score(logp(pids, cc, ctx_factory()), logp(pids, cv, ctx_factory()))

    comp_ids, _, _ = prefix(comp_messages)
    viol_ids, viol_text, viol_off = prefix(viol_messages)
    s_clean = S(comp_ids, nullcontext)
    s_base = S(viol_ids, nullcontext)

    observed = None
    if kind == "none":
        s_int = s_base
    elif kind == "value_add":
        layer_mod = decoder_layers(handle.model)[layer]
        s_int = S(viol_ids, lambda: value_add_hook(layer_mod, steer_vec, strength))
    elif kind == "spotlight":
        span_pos = [p for lst in _locate_target_tokens(viol_text, viol_off, list(span_names), "literal") for p in lst]
        observed = spotlight_attention(handle.model, span_pos, psi_target, key_len=viol_ids.shape[1])
        s_int = S(viol_ids, lambda: observed)
    else:
        raise ValueError(f"알 수 없는 kind: {kind}")

    out = {"S_clean": s_clean, "S_base": s_base, "S_int": s_int,
           "recovery": recovery_rate(s_clean, s_base, s_int), "layer": layer, "kind": kind}
    if observed is not None:
        out["attn_span_after"] = observed.mean_observed_psi()
        out["n_span_tokens"] = len(span_pos)
    return out
