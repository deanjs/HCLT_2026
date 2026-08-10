"""step C(개입) 단위 테스트 — 모델 없이 정렬·회복률·채점 헬퍼 검증.

KV 캐시 편집 등 모델 내부는 GPU가 필요하므로 노트북(Colab)에서 검증한다. 여기서는
순수 로직만 본다.
"""

import pytest

from harness.intervention import (
    align_name_tokens,
    preference_score,
    recovery_rate,
    sum_logprob,
)


# ── 정렬 ──────────────────────────────────────────────────────────────────

def test_align_role_wise_2to2():
    # 위반 이름 2개(각 2토큰) ↔ 공여 이름 2개(각 2토큰) → 역할별 4쌍
    viol = [[10, 11], [20, 21]]
    donor = [[10, 11], [20, 21]]
    pairs, skipped = align_name_tokens(viol, donor)
    assert pairs == [(10, 10), (11, 11), (20, 20), (21, 21)]
    assert skipped == []


def test_align_skips_length_mismatch():
    # 둘째 이름의 토큰 수가 다르면 그 이름만 제외(기록)
    viol = [[10, 11], [20, 21, 22]]
    donor = [[30, 31], [40, 41]]
    pairs, skipped = align_name_tokens(viol, donor)
    assert pairs == [(10, 30), (11, 31)]      # 첫 이름만 정렬
    assert skipped == [1]


def test_align_rejects_name_count_mismatch():
    with pytest.raises(ValueError):
        align_name_tokens([[1, 2]], [[1, 2], [3, 4]])


# ── 회복률 ────────────────────────────────────────────────────────────────

def test_recovery_full_and_none():
    # 개입이 깨끗 수준까지 → 1.0
    assert recovery_rate(s_clean=2.0, s_base=0.0, s_int=2.0) == pytest.approx(1.0)
    # 개입 효과 없음 → 0.0
    assert recovery_rate(s_clean=2.0, s_base=0.0, s_int=0.0) == pytest.approx(0.0)
    # 79% 회복
    assert recovery_rate(s_clean=1.0, s_base=0.0, s_int=0.79) == pytest.approx(0.79)


def test_recovery_partial_negative_base():
    # 위반 baseline이 음수(위반 선호), 깨끗은 양수인 전형적 상황
    r = recovery_rate(s_clean=1.5, s_base=-0.5, s_int=0.5)  # (0.5-(-0.5))/(1.5-(-0.5))=1/2
    assert r == pytest.approx(0.5)


def test_recovery_degenerate_denominator_is_none():
    assert recovery_rate(s_clean=1.0, s_base=1.0, s_int=1.2) is None


# ── 점수 ──────────────────────────────────────────────────────────────────

def test_preference_and_sum():
    assert preference_score(-1.0, -3.0) == pytest.approx(2.0)   # 준수가 위반보다 확률↑
    assert sum_logprob([-0.5, -1.5, -0.2]) == pytest.approx(-2.2)
