"""step6 처방 엔진 — 값 조향(CAA식)과 Spotlight 어텐션 조향.

두 개입 모두 **교사강제 점수**(준수 선호도)를 개입 하에서 재서 회복률을 얻는다.
step3·5의 KV 치환과 달리 여기서는 forward 훅으로 개입한다.

  · VALUE_ADD : 그 층 잔차에 `세기 × (camel−snake 방향)` 더하기 [CAA(ACL 2024)·ITI(NeurIPS 2023)]
  · SPOTLIGHT : 전 층·전 헤드 어텐션에서 지시어 스팬 비중을 목표치로 끌어올림
                (Spotlight, arXiv:2505.12025, 식(2)~(5) 재구현)

torch/transformers는 지연 임포트한다(스키마·테스트가 무거운 의존성 없이 돌도록).
Spotlight 수식은 순수 함수 `spotlight_reweight`로 분리해 모델 없이 검증한다.

주의 — 여기서 조용히 틀리기 쉬운 곳 (전부 막아 두었다)
  ① **인과 마스크.** 최신 transformers는 어텐션 함수에 `attention_mask=None`을 넘긴다
     (인과성을 내부에서 처리). 그대로 두면 우리가 다시 계산할 때 **미래 토큰까지 보게 되어**
     모델이 망가진 채로 측정된다. mask가 없으면 **직접 인과 마스크를 씌운다.**
  ② **주입 확인.** 어텐션 함수가 실제로 불렸는지 세어 둔다. 0이면 예외 —
     "개입했는데 효과 없음"과 "개입이 안 걸림"이 같은 숫자로 남으면 안 된다.
  ③ **개입 전후 비중.** ψ를 개입 전·후 모두 기록한다. 실제로 올랐는지 보여야
     "구현을 못해서 진 것"이라는 반박을 막는다.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

# 주입된 어텐션 함수가 몇 번 불렸는지 — 개입이 실제로 걸렸는지 확인용(위 ②).
_ATTENTION_FN_NAME = "spotlight_step6"


# ─────────────────────────────────────────────────────────────────────────────
# 순수 수식 — 모델 없이 검증 가능
# ─────────────────────────────────────────────────────────────────────────────
def spotlight_reweight(probs: Sequence[float], span_mask: Sequence[bool], psi_target: float):
    """Spotlight 재가중(확률 공간). 로짓에 log(ψ_target/ψ_current)를 더하고 재정규화한 것과 동치.

    probs      : 소프트맥스 어텐션 가중치 한 행(합=1)
    span_mask  : 지시어 스팬이면 True
    psi_target : 목표 비중 ψ_target ∈ (0,1)

    반환 (new_probs, ψ_before, ψ_after)
      · 게이팅: ψ_before ≥ ψ_target이면 손대지 않는다(원문 식 3).
      · 개입 시 스팬에 (ψ_target/ψ_before)를 곱하고 재정규화 → 언더슛(원문 식 5):
            ψ_after = ψ_target / (1 + ψ_target − ψ_before)
        목표에 정확히 도달하지 않는 것이 **정상**이다.
    """
    p = [float(x) for x in probs]
    mask = [bool(x) for x in span_mask]
    if len(p) != len(mask):
        raise ValueError(f"길이 불일치: probs {len(p)} vs mask {len(mask)}")
    before = sum(x for x, m in zip(p, mask) if m)
    if before >= psi_target:                       # 게이팅 — 이미 충분히 본다
        return p, before, before
    factor = psi_target / max(before, 1e-12)
    new = [x * factor if m else x for x, m in zip(p, mask)]
    total = sum(new)
    new = [x / total for x in new]
    after = sum(x for x, m in zip(new, mask) if m)
    return new, before, after


# ─────────────────────────────────────────────────────────────────────────────
# 모델 구조 접근
# ─────────────────────────────────────────────────────────────────────────────
def decoder_layers(model) -> Sequence[Any]:
    """디코더 블록 리스트. Llama/Qwen/DeepSeek/StableCode 계열 공통 경로를 훑는다."""
    base = model
    for attr in ("model", "transformer", "gpt_neox"):
        if hasattr(base, attr):
            base = getattr(base, attr)
            break
    for attr in ("layers", "h"):
        if hasattr(base, attr):
            return getattr(base, attr)
    raise RuntimeError("디코더 블록 리스트를 찾지 못했다 (model.model.layers 예상)")


class value_add_hook:
    """지정 층 블록 출력(잔차)에 `세기 × 방향`을 더하는 forward 훅.

    호출 횟수를 세어 둔다 — 훅이 안 걸렸는데 "효과 없음"으로 읽히는 일을 막는다.
    """

    def __init__(self, layer_module, vec, strength: float):
        self.layer_module = layer_module
        self.vec = vec
        self.strength = float(strength)
        self.n_calls = 0
        self._h = None

    def __enter__(self):
        add = self.strength * self.vec

        def fn(_mod, _inp, out):
            self.n_calls += 1
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
# Spotlight 어텐션 조향
# ─────────────────────────────────────────────────────────────────────────────
class spotlight_attention:
    """전 층·전 헤드에서 지시어 스팬의 어텐션 비중을 ψ_target으로 끌어올린다.

    자유도(원문 미명시)는 우리가 정하고 명시한다:
      · 전 쿼리 위치에 적용   · 헤드별 독립 ψ   · 스팬은 조건에서 지정
    """

    def __init__(self, model, span_positions: Sequence[int], psi_target: float):
        self.model = model
        self.span_positions = sorted(set(int(p) for p in span_positions))
        self.psi_target = float(psi_target)
        self.psi_before: list[float] = []
        self.psi_after: list[float] = []
        self.n_calls = 0
        self._prev_impl = None
        if not self.span_positions:
            raise ValueError("Spotlight 스팬이 비어 있다 — 어디를 밀지 정해지지 않았다")

    def __enter__(self):
        import torch
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        span, psi_t, outer = self.span_positions, self.psi_target, self

        def repeat_kv(x, n):
            if n == 1:
                return x
            b, h, s, d = x.shape
            return x[:, :, None, :, :].expand(b, h, n, s, d).reshape(b, h * n, s, d)

        def spotlight_fn(module, query, key, value, attention_mask,
                         scaling=None, dropout=0.0, **kwargs):
            outer.n_calls += 1
            groups = getattr(module, "num_key_value_groups", 1)
            k = repeat_kv(key, groups)
            v = repeat_kv(value, groups)
            if scaling is None:
                scaling = query.shape[-1] ** -0.5
            attn = torch.matmul(query, k.transpose(2, 3)) * scaling
            if attention_mask is not None:
                attn = attn + attention_mask[..., : k.shape[-2]]
            else:
                # 최신 transformers는 mask=None으로 넘기고 인과성을 내부에서 처리한다.
                # 여기서 직접 씌우지 않으면 **미래 토큰까지 보게 되어** 모델이 망가진다.
                q_len, k_len = attn.shape[-2], attn.shape[-1]
                causal = torch.ones(q_len, k_len, dtype=torch.bool,
                                    device=attn.device).tril(diagonal=k_len - q_len)
                attn = attn.masked_fill(~causal, torch.finfo(attn.dtype).min)
            probs = torch.softmax(attn, dim=-1, dtype=torch.float32)   # fp16 NaN 방지

            cols = [p for p in span if 0 <= p < probs.shape[-1]]
            if cols:
                idx = torch.tensor(cols, device=probs.device)
                before = probs.index_select(-1, idx).sum(-1, keepdim=True)
                factor = torch.where(before < psi_t,
                                     psi_t / before.clamp_min(1e-12),
                                     torch.ones_like(before))
                probs = probs.index_copy(-1, idx, probs.index_select(-1, idx) * factor)
                probs = probs / probs.sum(-1, keepdim=True)
                after = probs.index_select(-1, idx).sum(-1, keepdim=True)
                # 이름을 쓰는 마지막 쿼리 행 기준으로 개입 전·후를 남긴다
                outer.psi_before.append(float(before[..., -1, :].mean()))
                outer.psi_after.append(float(after[..., -1, :].mean()))
            probs = probs.to(query.dtype)
            out = torch.matmul(probs, v).transpose(1, 2).contiguous()
            return out, probs

        ALL_ATTENTION_FUNCTIONS[_ATTENTION_FN_NAME] = spotlight_fn
        self._prev_impl = self.model.config._attn_implementation
        self.model.config._attn_implementation = _ATTENTION_FN_NAME
        return self

    def __exit__(self, *exc):
        self.model.config._attn_implementation = self._prev_impl
        return False

    def summary(self) -> dict[str, Any]:
        """개입이 실제로 걸렸는지 + 비중이 실제로 올랐는지."""
        if self.n_calls == 0:
            raise RuntimeError(
                "Spotlight 어텐션 함수가 한 번도 불리지 않았다 — 개입이 안 걸렸다. "
                "transformers 버전의 어텐션 디스패치 경로를 확인할 것."
            )
        b = sum(self.psi_before) / len(self.psi_before) if self.psi_before else None
        a = sum(self.psi_after) / len(self.psi_after) if self.psi_after else None
        return {"attn_span_before": b, "attn_span_after": a,
                "attn_calls": self.n_calls,
                "psi_target": self.psi_target,
                "n_span_tokens": len(self.span_positions)}


# ─────────────────────────────────────────────────────────────────────────────
# 조향 방향 추출
# ─────────────────────────────────────────────────────────────────────────────
def build_steer_vector(handle, samples, layer: int, forced_prefix: str = "def "):
    """그 층 잔차 공간의 **camel − snake 방향**. 모델마다 자체 유도한다.

    반환 (vec, meta). meta에 표본 수·건너뛴 수·노름을 담아 결과에 남긴다 —
    무엇으로 만든 방향인지 파일만 보고 알 수 있어야 한다.
    건너뛴 표본이 절반을 넘으면 예외(조용한 실패 금지).
    """
    import torch
    from .model import _locate_target_tokens

    tok, dev = handle.tokenizer, handle.model.device
    cam, sna, skipped = [], [], 0
    for smp in samples:
        for msg_key, name_key, acc in (("camel_messages", "camel_names", cam),
                                       ("snake_messages", "snake_names", sna)):
            text = tok.apply_chat_template(smp[msg_key], tokenize=False,
                                           add_generation_prompt=True) + forced_prefix
            enc = tok(text, return_tensors="pt", return_offsets_mapping=True,
                      add_special_tokens=False)
            offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
            pos = [p for lst in _locate_target_tokens(text, offsets, smp[name_key], "def_name")
                   for p in lst]
            if not pos:
                skipped += 1
                continue
            with torch.no_grad():
                hs = handle.model(enc["input_ids"].to(dev),
                                  output_hidden_states=True).hidden_states[layer + 1][0]
            acc.append(hs[pos].float().mean(0))
    total = 2 * len(samples)
    if not cam or not sna:
        raise RuntimeError(f"조향 방향 추출 실패: 이름 토큰을 못 찾았다 (건너뜀 {skipped}/{total})")
    if skipped > total / 2:
        raise RuntimeError(f"조향 방향 추출 표본의 절반 이상을 건너뛰었다 ({skipped}/{total})")
    vec = torch.stack(cam).mean(0) - torch.stack(sna).mean(0)
    meta = {"n_samples": len(samples), "n_skipped": skipped,
            "vec_norm": float(vec.norm()), "layer": layer}
    return vec, meta


# ─────────────────────────────────────────────────────────────────────────────
# 개입 하 준수 선호 점수
# ─────────────────────────────────────────────────────────────────────────────
def steer_preference(handle, viol_messages, comp_messages, cand_comp, cand_viol, *,
                     kind: str, layer: Optional[int] = None, strength: Optional[float] = None,
                     steer_vec=None, psi_target: Optional[float] = None,
                     span_positions: Optional[Sequence[int]] = None,
                     forced_prefix: str = "def "):
    """개입 하 준수 선호도 회복. kind: 'none' | 'value_add' | 'spotlight'.

    S_깨끗 = 오염 없는 선행(천장) · S_기준 = 절벽(개입 없음) · S_개입 = 개입 후.
    회복률 = (S_개입 − S_기준)/(S_깨끗 − S_기준).
    """
    import torch
    from contextlib import nullcontext
    from .intervention import (UNDECIDABLE_GAP, effect_size, is_undecidable,
                               preference_score, recovery_rate)

    tok, dev = handle.tokenizer, handle.model.device

    def prefix(messages):
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True) + forced_prefix
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        return enc["input_ids"].to(dev)

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

    def S(pids, make_ctx):
        return preference_score(logp(pids, cc, make_ctx()), logp(pids, cv, make_ctx()))

    comp_ids, viol_ids = prefix(comp_messages), prefix(viol_messages)
    s_clean = S(comp_ids, nullcontext)
    s_base = S(viol_ids, nullcontext)

    detail: dict[str, Any] = {}
    if kind == "none":
        s_int = s_base
    elif kind == "value_add":
        if steer_vec is None or strength is None or layer is None:
            raise ValueError("value_add에는 steer_vec·strength·layer가 모두 필요하다")
        mod = decoder_layers(handle.model)[layer]
        hooks: list = []

        def make():
            h = value_add_hook(mod, steer_vec, strength)
            hooks.append(h)
            return h

        s_int = S(viol_ids, make)
        calls = sum(h.n_calls for h in hooks)
        if calls == 0:
            raise RuntimeError("값 조향 훅이 한 번도 불리지 않았다 — 개입이 안 걸렸다")
        detail["hook_calls"] = calls
    elif kind == "spotlight":
        if psi_target is None or not span_positions:
            raise ValueError("spotlight에는 psi_target과 span_positions가 필요하다")
        probe = spotlight_attention(handle.model, span_positions, psi_target)
        s_int = S(viol_ids, lambda: probe)
        detail.update(probe.summary())
    else:
        raise ValueError(f"알 수 없는 개입 종류: {kind!r} (none|value_add|spotlight)")

    out = {"S_clean": s_clean, "S_base": s_base, "S_int": s_int,
           "recovery": recovery_rate(s_clean, s_base, s_int),
           "gap": effect_size(s_clean, s_base),
           "undecidable": is_undecidable(s_clean, s_base),
           "undecidable_gap_min": UNDECIDABLE_GAP,
           "layer": layer, "kind": kind}
    out.update(detail)
    return out

def name_health(name: Optional[str]) -> dict[str, Any]:
    """생성된 이름이 **멀쩡한가**. 조향을 세게 걸면 점수만 오르고 이름이 깨질 수 있다.

    선호 점수만 보면 "천장의 4배 회복" 같은 값이 나오는데, 실제로는
    `removeDuplicatesDuplicates` 같은 걸 뱉고 있을 수 있다. 그걸 잡는다.

    반환: ok(전부 통과) + 어디서 걸렸는지.
    """
    import re
    if not name:
        return {"ok": False, "reason": "이름 없음", "name": None}
    checks = {
        "식별자 형식": bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)),
        "길이 정상": 2 <= len(name) <= 40,
        "같은 조각 반복 없음": not _has_repeat(name),
    }
    bad = [k for k, v in checks.items() if not v]
    return {"ok": not bad, "reason": ("" if not bad else ", ".join(bad)),
            "name": name, "length": len(name)}


def _has_repeat(name: str) -> bool:
    """`removeDuplicatesDuplicates`처럼 같은 조각이 연달아 반복되는가."""
    import re
    parts = [w.lower() for w in re.findall(r"[A-Z]?[a-z0-9]+", name)]
    return any(a == b for a, b in zip(parts, parts[1:]))


def steer_generate(handle, messages, *, kind: str, layer: Optional[int] = None,
                   strength: Optional[float] = None, steer_vec=None,
                   psi_target: Optional[float] = None,
                   span_positions: Optional[Sequence[int]] = None,
                   forced_prefix: str = "def ", max_new_tokens: int = 24) -> dict[str, Any]:
    """조향을 건 채로 **실제 이름을 생성**한다. 점수 회복이 진짜인지 눈으로 확인하는 용도."""
    import torch
    from contextlib import nullcontext

    tok, dev = handle.tokenizer, handle.model.device
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True) + forced_prefix
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)

    if kind == "none":
        make_ctx = nullcontext
    elif kind == "value_add":
        mod = decoder_layers(handle.model)[layer]
        make_ctx = lambda: value_add_hook(mod, steer_vec, strength)      # noqa: E731
    elif kind == "spotlight":
        make_ctx = lambda: spotlight_attention(handle.model, span_positions, psi_target)  # noqa: E731
    else:
        raise ValueError(f"알 수 없는 개입 종류: {kind!r}")

    gen: list[int] = []
    with make_ctx(), torch.no_grad():
        cur = ids
        past = None
        for _ in range(max_new_tokens):
            out = handle.model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())
            if nxt == tok.eos_token_id:
                break
            gen.append(nxt)
            cur = torch.tensor([[nxt]], device=dev)
    return {"text": forced_prefix + tok.decode(gen)}
