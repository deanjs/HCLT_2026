"""step 0 골격 검증 — 모델 없이 도는 스모크 테스트.

확인 대상: 조건 스키마 검증 / 직렬화 왕복 / 슬러그 결정성 / 결과 저장 불변성 /
단일 진입점 라우팅.
"""

import json

import pytest

from harness import (
    Composition,
    Condition,
    Instruction,
    InstructionForm,
    Intervention,
    InterventionKind,
    Metrics,
    ModelSpec,
    Notation,
    PrecedingCode,
    ResultRecord,
    Source,
    run,
    save_result,
)
from harness.runner import StageNotImplemented, describe_pipeline


def make_condition(**over):
    base = dict(
        model=ModelSpec(name="Qwen/Qwen2.5-Coder-3B-Instruct", family="qwen"),
        preceding=PrecedingCode(n_compliant=0),
        instruction=Instruction(form=InstructionForm.POSITIVE, target_notation=Notation.CAMEL),
        seed=0,
    )
    base.update(over)
    return Condition(**base)


# ── 조건 스키마 검증 ───────────────────────────────────────────────────────

def test_preceding_range_validation():
    with pytest.raises(ValueError):
        PrecedingCode(n_compliant=13)  # n_functions=12 초과
    with pytest.raises(ValueError):
        PrecedingCode(n_compliant=-1)


def test_repo_source_requires_lang():
    with pytest.raises(ValueError):
        PrecedingCode(n_compliant=2, source=Source.REPO)  # repo_lang 없음
    PrecedingCode(n_compliant=2, source=Source.REPO, repo_lang="python")  # OK


def test_instruction_notation_axes():
    ins = Instruction(form=InstructionForm.NEGATIVE, target_notation=Notation.CAMEL)
    # 부정형: 목표는 camel, 위반은 snake, 지침 문장에 등장하는 토큰은 위반(snake)
    assert ins.violation_notation is Notation.SNAKE
    assert ins.token_notation is Notation.SNAKE
    pos = Instruction(form=InstructionForm.POSITIVE, target_notation=Notation.CAMEL)
    assert pos.token_notation is Notation.CAMEL  # 긍정형은 목표 표기를 담음


def test_intervention_validation():
    with pytest.raises(ValueError):
        Intervention(kind=InterventionKind.VALUE)  # layers 없음
    with pytest.raises(ValueError):
        Intervention(kind=InterventionKind.NONE, layers=[25])  # NONE에 파라미터
    with pytest.raises(ValueError):
        Intervention(kind=InterventionKind.ATTENTION_AMPLIFY, layers=[25])  # amplify 없음
    Intervention(kind=InterventionKind.KEY_VALUE, layers="sweep")  # OK
    assert Intervention(kind=InterventionKind.VALUE, layers="sweep").is_sweep


# ── 직렬화 왕복 ────────────────────────────────────────────────────────────

def test_condition_roundtrip():
    c = make_condition(
        preceding=PrecedingCode(n_compliant=2, composition=Composition.DISTINCT),
        instruction=Instruction(form=InstructionForm.NEGATIVE, target_notation=Notation.SNAKE),
        intervention=Intervention(kind=InterventionKind.VALUE, layers=[25], donor="unrelated_camel"),
        task_ids=("t1", "t2", "t3"),
        tag="holdout",
    )
    d = json.loads(json.dumps(c.to_dict()))
    c2 = Condition.from_dict(d)
    assert c2 == c


# ── 슬러그 결정성 ──────────────────────────────────────────────────────────

def test_slug_deterministic_and_informative():
    c = make_condition(
        preceding=PrecedingCode(n_compliant=2),
        instruction=Instruction(form=InstructionForm.NEGATIVE, target_notation=Notation.CAMEL),
        intervention=Intervention(kind=InterventionKind.VALUE, layers=[25]),
    )
    s = c.slug()
    assert c.slug() == s  # 결정적
    assert "c2of12" in s and "ins-neg-camel" in s and "int-value-L25" in s and "s0" in s


# ── 결과 저장 불변성 ───────────────────────────────────────────────────────

def test_save_result_immutable(tmp_path):
    c = make_condition()
    rec = ResultRecord(condition=c, metrics=Metrics(compliance_rate=0.3),
                       step="stepA", rq="RQ1", prediction="0개에서 절벽")
    p = save_result(rec, results_dir=tmp_path)
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["metrics"]["compliance_rate"] == 0.3
    assert data["prediction"] == "0개에서 절벽"
    with pytest.raises(FileExistsError):  # 덮어쓰기 금지(§6)
        save_result(rec, results_dir=tmp_path)


# ── 단일 진입점 라우팅 ─────────────────────────────────────────────────────

def test_run_routes_to_generation_stage():
    # 생성 경로는 step A에서 구현됨. 생성기 없이 부르면 명시적 ValueError.
    with pytest.raises(ValueError):
        run(make_condition())  # 개입 없음 → 생성 경로, 그러나 생성기 미제공


def test_run_routes_to_intervention_stage():
    c = make_condition(intervention=Intervention(kind=InterventionKind.KEY_VALUE, layers="sweep"))
    with pytest.raises(StageNotImplemented) as ei:
        run(c)  # 개입 있음 → 개입 경로
    assert ei.value.stage_key == "apply_intervention"


def test_pipeline_description_lists_all_stages():
    text = describe_pipeline()
    for key in ("build_preceding", "measure_generation", "apply_intervention"):
        assert key in text
