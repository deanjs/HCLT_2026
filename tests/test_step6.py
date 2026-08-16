"""step6 처방 엔진 검사 — 값 조향(CAA식)과 Spotlight 어텐션 조향.

여기서 조용히 틀리면 논문의 헤드라인 비교표가 통째로 무의미해진다. 특히:
  · Spotlight가 **실제로 걸렸는지** (안 걸리고 "효과 없음"이 나오면 우리 주장이 거짓이 된다)
  · 인과 마스크가 보존되는지 (최신 transformers는 mask=None을 넘긴다 — 안 씌우면 모델이 망가진다)
  · 훅이 측정 후 **완전히 제거**되는지

수식 검사는 모델 없이 돌고, 나머지는 **무작위 가중치 소형 Llama**로 돈다(다운로드 없음).
"""

from __future__ import annotations

import pytest

from harness.steer import spotlight_reweight


# ── 수식 (모델 불필요) ────────────────────────────────────────────────────
def test_재가중이_원문_폐형과_일치한다():
    """개입 후 비중 = ψ_target / (1 + ψ_target − ψ_before)  (원문 식 5, 언더슛)."""
    probs = [0.1, 0.2, 0.3, 0.4]
    mask = [True, False, True, False]           # ψ_before = 0.4
    new, before, after = spotlight_reweight(probs, mask, 0.6)
    assert before == pytest.approx(0.4)
    assert after == pytest.approx(0.6 / (1 + 0.6 - 0.4))
    assert sum(new) == pytest.approx(1.0)


def test_이미_충분히_보면_손대지_않는다():
    """게이팅(원문 식 3) — ψ_before ≥ ψ_target이면 개입 없음."""
    probs = [0.1, 0.2, 0.3, 0.4]
    mask = [True, False, True, False]           # ψ_before = 0.4
    new, before, after = spotlight_reweight(probs, mask, 0.3)
    assert new == probs and before == after == pytest.approx(0.4)


def test_길이가_어긋나면_예외():
    with pytest.raises(ValueError):
        spotlight_reweight([0.5, 0.5], [True], 0.5)


# ── 모델이 필요한 검사 ────────────────────────────────────────────────────
torch = pytest.importorskip("torch", reason="조향 검사는 torch가 필요하다")
transformers = pytest.importorskip("transformers", reason="조향 검사는 transformers가 필요하다")

from harness.steer import decoder_layers, spotlight_attention, value_add_hook  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    """무작위 가중치 소형 Llama — 다운로드 없이 구조만 쓴다."""
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def ids():
    torch.manual_seed(1)
    return torch.randint(0, 64, (1, 8))


def _logits(model, ids):
    with torch.no_grad():
        return model(ids).logits.clone()


def test_스포트라이트가_인과성을_깨지_않는다(tiny, ids):
    """가장 위험한 지점.

    최신 transformers는 어텐션 함수에 attention_mask=None을 넘기고 인과성을 내부에서 처리한다.
    우리가 어텐션을 다시 계산하면서 마스크를 안 씌우면 **미래 토큰까지 보게 되어** 모델이
    망가진 채로 측정된다. ψ_target을 0에 가깝게 두면 개입이 없는 것과 같아야 한다.
    """
    base = _logits(tiny, ids)
    with spotlight_attention(tiny, [1, 2], psi_target=1e-9):
        out = _logits(tiny, ids)
    assert torch.allclose(base, out, atol=1e-4), "인과 마스크가 보존되지 않았다"


def test_스포트라이트가_실제로_비중을_올린다(tiny, ids):
    probe = spotlight_attention(tiny, [1, 2], psi_target=0.5)
    with probe:
        out = _logits(tiny, ids)
    s = probe.summary()
    assert s["attn_calls"] == 2, "층마다 한 번씩 불려야 한다"
    assert s["attn_span_after"] > s["attn_span_before"], "비중이 오르지 않았다"
    assert s["attn_span_after"] <= s["psi_target"] + 1e-6, "언더슛이어야 정상(원문 식 5)"
    assert not torch.allclose(_logits(tiny, ids), out, atol=1e-4), "출력이 그대로다"


def test_스포트라이트가_안_걸리면_예외(tiny):
    """개입이 안 걸렸는데 '효과 없음'으로 조용히 남으면 안 된다."""
    probe = spotlight_attention(tiny, [1], psi_target=0.5)
    with pytest.raises(RuntimeError):
        probe.summary()


def test_스포트라이트가_끝나면_원래_구현으로_돌아온다(tiny, ids):
    before = tiny.config._attn_implementation
    with spotlight_attention(tiny, [1], psi_target=0.5):
        pass
    assert tiny.config._attn_implementation == before
    assert torch.allclose(_logits(tiny, ids), _logits(tiny, ids), atol=1e-6)


def test_빈_스팬은_만들_때_막힌다(tiny):
    with pytest.raises(ValueError):
        spotlight_attention(tiny, [], psi_target=0.5)


def test_값조향_훅이_걸리고_해제된다(tiny, ids):
    base = _logits(tiny, ids)
    vec = torch.randn(32)
    hook = value_add_hook(decoder_layers(tiny)[1], vec, strength=2.0)
    with hook:
        out = _logits(tiny, ids)
    assert hook.n_calls == 1, "지정 층에서 한 번 불려야 한다"
    assert not torch.allclose(base, out, atol=1e-4), "조향이 출력을 안 바꿨다"
    assert torch.allclose(base, _logits(tiny, ids), atol=1e-6), "훅이 남아 있다"


def test_세기가_0이면_아무_일도_없다(tiny, ids):
    base = _logits(tiny, ids)
    with value_add_hook(decoder_layers(tiny)[1], torch.randn(32), strength=0.0):
        out = _logits(tiny, ids)
    assert torch.allclose(base, out, atol=1e-6)


def test_조향_세기가_커지면_출력_변화도_커진다(tiny, ids):
    base = _logits(tiny, ids)
    vec = torch.randn(32)
    diffs = []
    for strength in (0.5, 1.0, 2.0):
        with value_add_hook(decoder_layers(tiny)[1], vec, strength):
            diffs.append(float((_logits(tiny, ids) - base).abs().max()))
    assert diffs[0] < diffs[1] < diffs[2], f"세기에 단조롭게 반응하지 않는다: {diffs}"
