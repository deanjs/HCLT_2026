"""step A(생성 경로) 단위 테스트 — 모델 없이 프롬프트·판정·집계를 검증한다.

실제 200회 생성은 Colab 노트북에서. 여기서는 generate_fn seam으로 파이프라인을
끝까지 돌려 준수율·자기증폭 집계까지 확인한다.
"""

import re

import pytest

from harness import (
    CLONE_TASK,
    GENERATION_TASKS,
    build_instruction_text,
    build_preceding_code,
    classify_name,
    first_def_name,
    render_camel,
    render_snake,
    run,
)
from harness.conditions import (
    Composition,
    Condition,
    Instruction,
    InstructionForm,
    Intervention,
    InterventionKind,
    ModelSpec,
    Notation,
    PrecedingCode,
)
from harness.runner import StageNotImplemented

MODEL = ModelSpec(name="Qwen/Qwen2.5-Coder-3B-Instruct", family="qwen")


def _cond(n_compliant=2, seed=0, kind=InterventionKind.NONE):
    intervention = (
        Intervention() if kind is InterventionKind.NONE
        else Intervention(kind=kind, layers=[25])
    )
    return Condition(
        model=MODEL,
        preceding=PrecedingCode(n_compliant=n_compliant, composition=Composition.CLONE),
        instruction=Instruction(form=InstructionForm.POSITIVE, target_notation=Notation.CAMEL),
        intervention=intervention,
        seed=seed,
    )


# ── 렌더링·판정 ────────────────────────────────────────────────────────────

def test_render_camel_snake():
    assert render_camel(["scale", "value"], idx=2) == "scaleValue2"
    assert render_snake(["scale", "value"], idx=3) == "scale_value_3"
    assert render_camel(["clamp", "number"]) == "clampNumber"
    assert render_snake(["clamp", "number"]) == "clamp_number"


def test_classify_name():
    assert classify_name("scaleValue2") == "camel"
    assert classify_name("scale_value_1") == "snake"
    assert classify_name("clampNumber") == "camel"
    assert classify_name("clamp_number") == "snake"
    assert classify_name("clamp") == "other"      # 한 단어는 구분 불가
    assert classify_name("ClampNumber") == "other"  # PascalCase는 후보 밖


def test_first_def_name():
    assert first_def_name("here:\ndef clampNumber(x, lo, hi):\n    return x") == "clampNumber"
    assert first_def_name("no function here") is None


# ── 선행 코드 구성 ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_build_preceding_counts(n):
    code = build_preceding_code(_cond(n_compliant=n))
    names = re.findall(r"def (\w+)\(", code)
    assert len(names) == 12
    camels = [x for x in names if classify_name(x) == "camel"]
    snakes = [x for x in names if classify_name(x) == "snake"]
    assert len(camels) == n          # 목표(camel) = 준수 개수
    assert len(snakes) == 12 - n     # 나머지는 위반(snake)


def test_build_preceding_shuffle_is_seeded():
    a = build_preceding_code(_cond(n_compliant=2, seed=0))
    b = build_preceding_code(_cond(n_compliant=2, seed=0))
    c = build_preceding_code(_cond(n_compliant=2, seed=1))
    assert a == b          # 같은 seed → 동일 배치
    assert a != c          # 다른 seed → 다른 셔플(거의 확실)


def test_instruction_weak_positive_camel():
    text = build_instruction_text(_cond())
    assert "generally" in text          # 약한 어조
    assert "camelCase" in text
    assert "camelCase or snake_case" in text  # 닫힌 후보


# ── 생성 경로 end-to-end (fake generate_fn) ────────────────────────────────

def _fake_gen(notation):
    """모든 생성 과제를 주어진 표기로 구현하는 가짜 생성기."""
    def gen(messages):
        # 마지막 user 요청에 해당하는 과제를 찾아 그 표기로 함수 이름을 만든다
        asked = messages[-1]["content"]
        task = next(t for t in GENERATION_TASKS if t.description in asked)
        name = task.name(notation)
        return f"```python\ndef {name}({', '.join(task.params)}):\n    ...\n```"
    return gen


def test_run_all_compliant():
    out = run(_cond(n_compliant=4), generate_fn=_fake_gen(Notation.CAMEL))
    e = out.metrics.extra
    assert out.metrics.compliance_rate == 1.0
    assert e["turn_notations"] == ["camel", "camel", "camel"]
    assert e["first_violated"] is False
    assert e["subsequent_violation_rate"] == 0.0
    # 생성 원문·함수명이 함께 보존되는가
    assert e["turn_names"] == ["removeDuplicates", "countVowels", "mergeDicts"]
    assert len(e["turn_texts"]) == 3 and all("def " in t for t in e["turn_texts"])


def test_run_all_violation_self_amplifies():
    out = run(_cond(n_compliant=0), generate_fn=_fake_gen(Notation.SNAKE))
    assert out.metrics.compliance_rate == 0.0
    assert out.metrics.extra["first_violated"] is True
    assert out.metrics.extra["subsequent_violation_rate"] == 1.0  # 첫 위반 → 연쇄


def test_intervention_path_needs_handle():
    # 개입 경로(step C 구현)는 모델 내부 접근이 필요 → handle 없으면 ValueError.
    with pytest.raises(ValueError):
        run(_cond(kind=InterventionKind.VALUE), generate_fn=_fake_gen(Notation.CAMEL))


def test_generation_needs_a_generator():
    with pytest.raises(ValueError):
        run(_cond())  # handle도 generate_fn도 없음
