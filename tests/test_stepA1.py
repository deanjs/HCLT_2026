"""step A-1(선행 distinct) 단위 테스트 — 모델 없이 선행 구성·생성 경로 검증."""

import re

import pytest

from harness import (
    DISTINCT_TASKS,
    GENERATION_TASKS,
    build_preceding_code,
    classify_name,
    run,
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


def _cond(n_compliant=2, seed=0):
    return Condition(
        model=MODEL,
        preceding=PrecedingCode(n_compliant=n_compliant, composition=Composition.DISTINCT),
        instruction=Instruction(form=InstructionForm.POSITIVE, target_notation=Notation.CAMEL),
        seed=seed,
    )


def test_distinct_pool_is_valid():
    assert len(DISTINCT_TASKS) >= 12
    bases = [t.words for t in DISTINCT_TASKS]
    assert len(set(bases)) == len(bases)              # 서로 다른 과제
    assert all(len(t.words) >= 2 for t in DISTINCT_TASKS)  # 두 단어 이상(표기 구분)
    # 생성 3개와 겹치지 않는다
    gen = {t.words for t in GENERATION_TASKS}
    assert gen.isdisjoint(set(bases))


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_build_preceding_distinct_counts(n):
    code = build_preceding_code(_cond(n_compliant=n))
    names = re.findall(r"def (\w+)\(", code)
    assert len(names) == 12
    assert len(set(names)) == 12                      # 이름이 모두 다르다(distinct)
    assert sum(classify_name(x) == "camel" for x in names) == n
    assert sum(classify_name(x) == "snake" for x in names) == 12 - n


def test_distinct_differs_from_clone():
    d = build_preceding_code(_cond(n_compliant=2, seed=0))
    clone = build_preceding_code(
        Condition(model=MODEL,
                  preceding=PrecedingCode(n_compliant=2, composition=Composition.CLONE),
                  instruction=Instruction(form=InstructionForm.POSITIVE,
                                          target_notation=Notation.CAMEL),
                  seed=0)
    )
    assert d != clone
    # clone은 같은 base(scale_value) 반복, distinct는 서로 다른 base
    assert "scale_value" in clone and "scale_value" not in d


def _fake_gen(notation):
    def gen(messages):
        task = next(t for t in GENERATION_TASKS if t.description in messages[-1]["content"])
        return f"def {task.name(notation)}({', '.join(task.params)}):\n    ..."
    return gen


def test_run_distinct_generation_path():
    out = run(_cond(n_compliant=4), generate_fn=_fake_gen(Notation.CAMEL))
    assert out.metrics.compliance_rate == 1.0
    assert out.metrics.extra["turn_notations"] == ["camel", "camel", "camel"]
    assert len(out.metrics.extra["turn_texts"]) == 3   # 원문 자동 저장
