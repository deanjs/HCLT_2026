"""step6(처방) 순수 로직 테스트 — torch 없이 검증 가능한 부분.

torch 훅(값 더하기·Spotlight 어텐션)의 실제 forward는 Colab 첫 실행에서 검증한다.
여기서는 조건 스키마·slug·Spotlight 수식·라우팅·조향표본 구성을 검증한다.
"""
import numpy as np
import pytest

from harness.conditions import (
    Condition, ModelSpec, PrecedingCode, Instruction,
    Composition, InstructionForm, Notation, Intervention, InterventionKind,
)
from harness import steer
from harness.runner import run, _steer_samples


M = ModelSpec(name="Qwen/Qwen2.5-Coder-3B-Instruct", family="qwen")


def _cond(iv, tag=None):
    return Condition(
        model=M,
        preceding=PrecedingCode(n_compliant=0, n_functions=12,
                                composition=Composition.POOL, pool_block=3),
        instruction=Instruction(form=InstructionForm.POSITIVE, target_notation=Notation.CAMEL),
        intervention=iv, token_unit="mean", seed=42, tag=tag,
    )


def test_value_add_condition_and_slug():
    va = Intervention(kind=InterventionKind.VALUE_ADD, layers=[25], strength=2.0,
                      steer_source="code_contrast")
    c = _cond(va, tag="value_star-s2.0")
    assert "value_add" in c.slug() and "str2" in c.slug()
    # 세기가 다르면 파일이 갈린다(재개 충돌 방지)
    c4 = _cond(Intervention(kind=InterventionKind.VALUE_ADD, layers=[25], strength=4.0,
                            steer_source="code_contrast"), tag="value_star-s4.0")
    assert c.slug() != c4.slug()


def test_value_add_requires_strength_and_source():
    with pytest.raises(ValueError):
        Intervention(kind=InterventionKind.VALUE_ADD, layers=[25], steer_source="code_contrast")
    with pytest.raises(ValueError):
        Intervention(kind=InterventionKind.VALUE_ADD, layers=[25], strength=1.0)
    with pytest.raises(ValueError):  # 치환 계열엔 strength 금지
        Intervention(kind=InterventionKind.KEY_VALUE, layers=[25], strength=1.0)


def test_condition_roundtrip_keeps_steer_fields():
    va = Intervention(kind=InterventionKind.VALUE_ADD, layers=[18], strength=8.0,
                      steer_source="code_contrast")
    c = _cond(va)
    c2 = Condition.from_dict(c.to_dict())
    assert c2.intervention.strength == 8.0
    assert c2.intervention.steer_source == "code_contrast"
    assert c2.slug() == c.slug()


def test_spotlight_reweight_closed_form():
    # 스팬비중 0.1, 목표 0.3 → 언더슛 0.25 (= 0.3/(1+0.3-0.1))
    new, cur, newp = steer.spotlight_reweight_np(
        np.array([0.05, 0.05, 0.9]), [True, True, False], 0.3)
    assert abs(float(cur.ravel()[0]) - 0.1) < 1e-9
    assert abs(float(newp.ravel()[0]) - 0.25) < 1e-9
    assert abs(float(new.sum()) - 1.0) < 1e-9      # 행합 보존


def test_spotlight_gating_no_change_when_already_high():
    # 이미 0.5 ≥ 0.3 → 게이팅, 불변
    new, cur, newp = steer.spotlight_reweight_np(
        np.array([0.25, 0.25, 0.5]), [True, True, False], 0.3)
    assert abs(float(newp.ravel()[0]) - 0.5) < 1e-9
    assert np.allclose(new, [0.25, 0.25, 0.5])


def test_spotlight_reweight_batched_rows():
    probs = np.array([[0.05, 0.05, 0.9], [0.4, 0.1, 0.5]])
    new, cur, newp = steer.spotlight_reweight_np(probs, [True, True, False], 0.3)
    assert np.allclose(new.sum(1), 1.0)
    assert abs(newp.ravel()[0] - 0.25) < 1e-9      # 0.1 -> 0.25
    assert abs(newp.ravel()[1] - 0.5) < 1e-9       # 0.5 게이팅


def test_run_steer_dispatch_needs_handle():
    # mode='steer'가 _run_steer로 라우팅됨을 확인(handle 없으면 그 경로에서 에러).
    c = _cond(Intervention(kind=InterventionKind.NONE), tag="none")
    with pytest.raises(ValueError, match="handle"):
        run(c, handle=None, mode="steer")


def test_steer_samples_structure():
    c = _cond(Intervention(kind=InterventionKind.VALUE_ADD, layers=[25], strength=1.0,
                           steer_source="code_contrast"))
    samples = _steer_samples(c, n_blocks=3)
    assert len(samples) == 3
    s0 = samples[0]
    assert set(s0) == {"camel_messages", "snake_messages", "camel_names", "snake_names"}
    # 이름은 표기만 다르고 개수 같아야
    assert len(s0["camel_names"]) == len(s0["snake_names"]) == 12
    # 서로 다른 묶음은 이름 집합이 다르다(풀 커버)
    assert samples[0]["camel_names"] != samples[1]["camel_names"]
    # camel 이름엔 밑줄 없음, snake엔 있음(대표 검사)
    assert all("_" not in n for n in s0["camel_names"])
    assert all("_" in n for n in s0["snake_names"])
