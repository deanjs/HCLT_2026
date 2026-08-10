"""step 2(모델 다양성) 단위 테스트 — 마지막-토큰 정렬(옵션 B)과 token_unit 축.

다른 패밀리 토크나이저는 camel/snake 이름을 다른 토큰 수로 쪼개 all 모드가 전부
스킵된다(정렬 0). 옵션 B(마지막 토큰만)는 토큰 수가 달라도 1:1이라 항상 정렬된다.
여기서는 순수 정렬 로직과 조건 축(token_unit) 직렬화·슬러그만 본다(모델 불필요).
"""

import pytest

from harness.intervention import align_name_tokens
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

MODEL = ModelSpec(name="stabilityai/stable-code-instruct-3b", family="stability")


# ── 마지막-토큰 정렬 (옵션 B) ──────────────────────────────────────────────

def test_all_mode_skips_length_mismatch():
    # snake 3토큰 vs camel 2토큰 → all 모드는 그 이름을 스킵(step2에서 전부 이 상황)
    viol = [[10, 11, 12], [20, 21, 22]]
    donor = [[30, 31], [40, 41]]
    pairs, skipped = align_name_tokens(viol, donor, mode="all")
    assert pairs == []
    assert skipped == [0, 1]


def test_last_mode_aligns_despite_mismatch():
    # 같은 입력이라도 last 모드는 각 이름의 마지막 토큰만 1:1 → 스킵 없음
    viol = [[10, 11, 12], [20, 21, 22]]
    donor = [[30, 31], [40, 41]]
    pairs, skipped = align_name_tokens(viol, donor, mode="last")
    assert pairs == [(12, 31), (22, 41)]       # 각 이름의 마지막 토큰
    assert skipped == []


def test_last_mode_skips_empty_side():
    # 한쪽 이름을 못 찾아 빈 리스트면 last 모드에서도 정렬 불가로 스킵
    viol = [[10, 11], []]
    donor = [[30, 31], [40, 41]]
    pairs, skipped = align_name_tokens(viol, donor, mode="last")
    assert pairs == [(11, 31)]                  # name0 마지막쌍, name1은 빈쪽이라 스킵
    assert skipped == [1]


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        align_name_tokens([[1]], [[2]], mode="middle")


# ── token_unit 조건 축 ─────────────────────────────────────────────────────

def _cond(token_unit="all"):
    return Condition(
        model=MODEL,
        preceding=PrecedingCode(n_compliant=0, n_functions=12, composition=Composition.POOL),
        instruction=Instruction(form=InstructionForm.POSITIVE, target_notation=Notation.CAMEL),
        intervention=Intervention(kind=InterventionKind.KEY_VALUE, layers="sweep", donor="compliant"),
        seed=0,
        token_unit=token_unit,
    )


def test_token_unit_validation():
    with pytest.raises(ValueError):
        _cond(token_unit="first")


def test_token_unit_slug_only_when_not_all():
    # all은 슬러그에 안 드러남(기존 step1 슬러그 불변), last만 tok-last로 표기
    assert "tok-" not in _cond("all").slug()
    assert "tok-last" in _cond("last").slug()


def test_token_unit_roundtrip():
    c = _cond("last")
    assert Condition.from_dict(c.to_dict()).token_unit == "last"
    # 기존 결과(토큰 필드 없음)는 기본 all로 복원 — 하위호환
    d = c.to_dict(); del d["token_unit"]
    assert Condition.from_dict(d).token_unit == "all"
