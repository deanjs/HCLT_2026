"""step 1(RQ2 층 스윕 + K/V 분해) 단위 테스트 — 모델 없이 순수 로직·라우팅 검증.

KV 캐시 층 편집·value forward 등 모델 내부는 GPU가 필요하므로 노트북(Colab)에서
검증한다. 여기서는 (1) v 코사인의 순수 계산, (2) 피크 층 요약, (3) 단일 진입점이
스윕/코사인 경로로 라우팅하며 per_layer를 규약대로 평평하게 담는지만 본다.
"""

import math

import pytest

from harness import (
    Composition,
    Condition,
    Instruction,
    InstructionForm,
    Intervention,
    InterventionKind,
    ModelSpec,
    Notation,
    PrecedingCode,
    run,
)
from harness.attention_probe import cosine
from harness.intervention import peak_layer

MODEL = ModelSpec(name="Qwen/Qwen2.5-Coder-3B-Instruct", family="qwen")


# ── v 코사인 (축 B 방향) ────────────────────────────────────────────────────

def test_cosine_identical_orthogonal_opposite():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)     # 같은 방향
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)     # 직교(다른 정보)
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)   # 반대


def test_cosine_ignores_magnitude():
    # 크기가 달라도 방향이 같으면 1.0 — ‖v‖와 무관함을 확인(step B 배율 1.002 맥락).
    assert cosine([2.0, 0.0], [10.0, 0.0]) == pytest.approx(1.0)


def test_cosine_hand_value():
    # (1,1)·(1,0)/(√2·1) = 1/√2
    assert cosine([1.0, 1.0], [1.0, 0.0]) == pytest.approx(1 / math.sqrt(2))


def test_cosine_zero_vector_is_none():
    assert cosine([0.0, 0.0], [1.0, 2.0]) is None                   # 방향 미정의


def test_cosine_length_mismatch_raises():
    with pytest.raises(ValueError):
        cosine([1.0, 0.0], [1.0])


# ── 피크 층 요약 ────────────────────────────────────────────────────────────

def test_peak_layer_finds_max_and_skips_none():
    got = peak_layer({0: 0.1, 10: None, 25: 0.67, 30: 0.4})
    assert got == (25, 0.67)


def test_peak_layer_all_none_is_none():
    assert peak_layer({0: None, 1: None}) is None


# ── 라우팅: 스윕/코사인 (fake handle, 모델 없이) ────────────────────────────

def _pool_cond(intervention=None, seed=0):
    return Condition(
        model=MODEL,
        preceding=PrecedingCode(n_compliant=0, n_functions=12, composition=Composition.POOL),
        instruction=Instruction(form=InstructionForm.POSITIVE, target_notation=Notation.CAMEL),
        intervention=intervention or Intervention(),
        seed=seed,
    )


class _FakeSweepHandle:
    """intervene_preference_sweep만 흉내내 runner의 평탄화·라우팅을 검증한다."""

    def __init__(self):
        self.called_with = None

    def intervene_preference_sweep(self, viol_messages, comp_messages, **kw):
        self.called_with = kw
        return {
            "S_clean": 3.52,
            "S_base": -2.91,
            "per_layer": {
                0: {"key": {"S_int": -2.8, "recovery": 0.02},
                    "value": {"S_int": -2.7, "recovery": 0.03},
                    "key_value": {"S_int": -2.6, "recovery": 0.05}},
                25: {"key": {"S_int": 0.5, "recovery": 0.53},
                     "value": {"S_int": -1.0, "recovery": 0.30},
                     "key_value": {"S_int": 1.40, "recovery": 0.67}},
            },
            "layers": [0, 25],
            "kinds": ["key", "value", "key_value"],
            "n_substituted_tokens": 24,
            "skipped_names": [],
        }


def test_run_routes_sweep_and_flattens_per_layer():
    c = _pool_cond(Intervention(kind=InterventionKind.KEY_VALUE, layers="sweep", donor="compliant"))
    handle = _FakeSweepHandle()
    out = run(c, handle=handle)

    # 층별로 kind×지표가 평평한 키로 담긴다.
    assert out.metrics.per_layer[25]["key_value__recovery"] == pytest.approx(0.67)
    assert out.metrics.per_layer[25]["key__recovery"] == pytest.approx(0.53)
    assert out.metrics.per_layer[25]["value__recovery"] == pytest.approx(0.30)
    assert out.metrics.per_layer[0]["key__S_int"] == pytest.approx(-2.8)
    # 층-무관 값은 extra에.
    assert out.metrics.extra["mode"] == "intervene_sweep"
    assert out.metrics.extra["S_clean"] == pytest.approx(3.52)
    assert out.metrics.extra["donor"] == "compliant"
    # 스윕은 전 층 → layers=None으로 넘겨 모델이 num_layers를 쓰게 한다.
    assert handle.called_with["layers"] is None
    assert tuple(handle.called_with["kinds"]) == ("key", "value", "key_value")


class _FakeCosineHandle:
    def name_v_cosine_sweep(self, camel_messages, snake_messages, **kw):
        return {
            "per_layer": {0: {"v_cosine": 0.95, "n": 48},
                          25: {"v_cosine": 0.20, "n": 48},
                          30: {"v_cosine": None, "n": 0}},   # 미정의 층은 걸러져야
            "n_pairs": 24,
            "skipped_names": [],
        }


def test_run_routes_vcosine_and_drops_none_layers():
    c = _pool_cond()
    out = run(c, handle=_FakeCosineHandle(), mode="vcosine")
    assert out.metrics.per_layer[0]["v_cosine"] == pytest.approx(0.95)
    assert out.metrics.per_layer[25]["v_cosine"] == pytest.approx(0.20)
    assert 30 not in out.metrics.per_layer                  # None은 저장하지 않는다
    assert out.metrics.extra["mode"] == "vcosine"
    assert out.metrics.extra["n_pairs"] == 24
