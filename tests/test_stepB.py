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
