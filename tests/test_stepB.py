"""step B(RQ2 관측) 단위 테스트 — 모델 없이 이름 짝 풀·균형 배치 검증.

관측 경로(measure_attention)의 모델 내부 훅은 GPU가 필요하므로 여기서 다루지 않고,
표본 설계(docs/stepB/plan.md)의 전제 — 80짝 풀의 건전성과 6/6 균형 배치 — 만 검증한다.
"""

import re

from harness import (
    NAME_PAIR_POOL,
    build_preceding_code,
    classify_name,
    render_camel,
    render_snake,
)
from harness.conditions import (
    Composition,
    Condition,
    Instruction,
    InstructionForm,
    ModelSpec,
    Notation,
    PrecedingCode,
)

MODEL = ModelSpec(name="Qwen/Qwen2.5-Coder-3B-Instruct", family="qwen")


def _cond(target=Notation.CAMEL, seed=0):
    return Condition(
        model=MODEL,
        preceding=PrecedingCode(n_compliant=6, n_functions=12, composition=Composition.POOL),
        instruction=Instruction(form=InstructionForm.POSITIVE, target_notation=target),
        seed=seed,
    )


# ── 이름 짝 풀 건전성 ──────────────────────────────────────────────────────

def test_pool_size_and_uniqueness():
    assert len(NAME_PAIR_POOL) >= 80                       # 예비 80쌍 규모
    bases = [t.words for t in NAME_PAIR_POOL]
    assert len(set(bases)) == len(bases)                   # 전부 고유 짝


def test_pool_names_are_two_words_and_notation_distinct():
    for t in NAME_PAIR_POOL:
        assert len(t.words) >= 2                           # 두 단어 → camel≠snake
        cam, sna = render_camel(t.words), render_snake(t.words)
        assert cam != sna
        assert classify_name(cam) == "camel"               # 판정기가 정확히 분류
        assert classify_name(sna) == "snake"


def test_pool_bodies_are_uniform():
    # 본문·매개변수가 전 항목 공통이어야 이름 표기만 변수로 남는다(stepB 통제).
    params = {t.params for t in NAME_PAIR_POOL}
    bodies = {t.body for t in NAME_PAIR_POOL}
    assert len(params) == 1
    assert len(bodies) == 1


# ── 6/6 균형 배치 ─────────────────────────────────────────────────────────

def _counts(code):
    defs = re.findall(r"def (\w+)\(", code)
    cls = [classify_name(d) for d in defs]
    return defs, cls


def test_balanced_placement_is_6_6_regardless_of_instruction():
    # 지침이 camel이든 snake든 선행은 항상 camel 6 / snake 6 (개수 편향 제거).
    for target in (Notation.CAMEL, Notation.SNAKE):
        for seed in range(4):
            defs, cls = _counts(build_preceding_code(_cond(target, seed)))
            assert len(defs) == 12
            assert cls.count("camel") == 6
            assert cls.count("snake") == 6
            assert len(set(defs)) == 12                     # 12개 모두 서로 다른 이름


def test_seeds_draw_different_names():
    # seed마다 다른 이름 집합이어야 효과가 특정 단어에 묶이지 않는다.
    n0, _ = _counts(build_preceding_code(_cond(Notation.CAMEL, 0)))
    n5, _ = _counts(build_preceding_code(_cond(Notation.CAMEL, 5)))
    assert set(n0) != set(n5)


def test_placement_is_deterministic_per_seed():
    a = build_preceding_code(_cond(Notation.CAMEL, 3))
    b = build_preceding_code(_cond(Notation.CAMEL, 3))
    assert a == b


# ── 어텐션 집계(GQA 방법 A) ───────────────────────────────────────────────

from harness.attention_probe import find_char_spans, locate_token_spans, span_metrics

# Hq=4, Hkv=2, group_size=2 → head 0·1→KV0, head 2·3→KV1
_ATTN = [
    [0.1, 0.2, 0.7],   # head0 (KV0)
    [0.0, 0.5, 0.5],   # head1 (KV0)
    [0.3, 0.3, 0.4],   # head2 (KV1)
    [0.2, 0.2, 0.6],   # head3 (KV1)
]
_VNORM = [
    [1.0, 2.0, 3.0],     # KV0
    [10.0, 20.0, 30.0],  # KV1
]


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_span_metrics_gqa_mapping_hand_computed():
    m = span_metrics(_ATTN, _VNORM, group_size=2, spans={"A": [0, 1]})["A"]
    # 어텐션 합(구간 [0,1]) head 평균: (0.3+0.5+0.6+0.4)/4
    assert _close(m["attention_weight"], 0.45)
    # ‖av‖ 합, head는 h//2로 KV 매핑: head0,1→KV0 / head2,3→KV1
    #   h0:0.1*1+0.2*2=0.5  h1:0.5*2=1.0  h2:0.3*10+0.3*20=9.0  h3:0.2*10+0.2*20=6.0
    #   평균 (0.5+1.0+9.0+6.0)/4 = 4.125
    assert _close(m["av_norm"], 4.125)
    # ‖v‖ 은 고유 KV 2개로 접어 평균: (1+2+10+20)/4 = 8.25
    assert _close(m["v_norm"], 8.25)
    assert m["n_tokens"] == 2


def test_span_metrics_wrong_kv_mapping_would_differ():
    # head를 KV에 잘못 매핑(그냥 head 인덱스로 v를 집으면) 값이 달라야 한다 —
    # 즉 우리 구현이 실제로 h//group_size를 쓰고 있음을 확인.
    m = span_metrics(_ATTN, _VNORM, group_size=2, spans={"A": [2]})["A"]
    # 올바른 매핑: head0,1→KV0(v=3.0), head2,3→KV1(v=30.0)
    #   av = (0.7*3 + 0.5*3 + 0.4*30 + 0.6*30)/4 = (2.1+1.5+12+18)/4 = 8.4
    assert _close(m["av_norm"], 8.4)


def test_span_metrics_empty_span_is_none():
    m = span_metrics(_ATTN, _VNORM, group_size=2, spans={"E": []})["E"]
    assert m["attention_weight"] is None and m["n_tokens"] == 0


def test_span_metrics_rejects_bad_shapes():
    import pytest
    with pytest.raises(ValueError):
        span_metrics(_ATTN, _VNORM, group_size=3, spans={"A": [0]})  # 4 != 2*3


def test_locate_token_spans_overlap():
    offsets = [(0, 0), (0, 3), (3, 7), (7, 10)]        # 첫 토큰은 특수(빈 offset)
    got = locate_token_spans(offsets, {"x": [(3, 7)], "y": [(4, 6)]})
    assert got["x"] == [2]                              # 정확히 겹치는 토큰
    assert got["y"] == [2]                              # 부분 겹침도 포착


def test_find_char_spans_multiple_occurrences():
    text = "def parseHeader(v): return v\n\ndef parseHeader(v): return v"
    spans = find_char_spans(text, ["def parseHeader("])
    assert len(spans) == 2
    for s, e in spans:
        assert text[s:e] == "def parseHeader("


# ── per-token 상세 ────────────────────────────────────────────────────────

def test_span_metrics_detail_returns_per_token():
    m = span_metrics(_ATTN, _VNORM, group_size=2, spans={"A": [0, 1, 2]}, detail=("A",))["A"]
    toks = m["tokens"]
    assert [t["t"] for t in toks] == [0, 1, 2]              # 토큰 인덱스 보존
    assert all(set(t) == {"t", "a", "av", "v"} for t in toks)


def test_detail_sums_back_to_group():
    # per-token 합/평균이 그룹 값과 정확히 일치해야 한다(내림-집계 가능성 보장).
    m = span_metrics(_ATTN, _VNORM, group_size=2, spans={"A": [0, 1, 2]}, detail=("A",))["A"]
    toks = m["tokens"]
    assert _close(sum(t["a"] for t in toks), m["attention_weight"])   # Σ a_j = 그룹 어텐션 합
    assert _close(sum(t["av"] for t in toks), m["av_norm"])           # Σ av_j = 그룹 av 합
    assert _close(sum(t["v"] for t in toks) / len(toks), m["v_norm"])  # mean v_j = 그룹 v 평균


def test_detail_only_for_requested_spans():
    out = span_metrics(_ATTN, _VNORM, group_size=2,
                       spans={"A": [0], "B": [1]}, detail=("A",))
    assert "tokens" in out["A"] and "tokens" not in out["B"]
