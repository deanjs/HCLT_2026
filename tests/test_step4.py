"""step 4(RQ3 인과 — 지침 지시어 KV 치환) 단위 테스트 — 모델 없이 조립·위치만 검증.

KV 캐시 편집 자체는 GPU가 필요하므로 여기서 다루지 않고, step4가 stepC 개입 구조를
**지침 지시어 토큰 + 반대 지침 donor**로 재배선한 부분의 건전성만 본다:
  - 개입 타깃 축(conditions.Intervention.target) 검증·직렬화·슬러그
  - _preference_setup_instruction 이 방향별로 치환 대상/공여 지시어를 맞게 교차하는가
  - literal span_kind 가 지침 rule 문장의 지시어 토큰(첫 등장)을 집는가
  - pool_block(504 커버)과 함께 동작하는가
"""

import pytest

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
from harness.model import _locate_target_tokens
from harness.prompt import notation_word
from harness.runner import _preference_setup_instruction

MODEL = ModelSpec(name="Qwen/Qwen2.5-Coder-3B-Instruct", family="qwen")


def _cond(target=Notation.CAMEL, n_compliant=6, block=0, seed=0):
    return Condition(
        model=MODEL,
        preceding=PrecedingCode(n_compliant=n_compliant, n_functions=12,
                                composition=Composition.POOL, pool_block=block),
        instruction=Instruction(form=InstructionForm.POSITIVE, target_notation=target),
        intervention=Intervention(kind=InterventionKind.KEY_VALUE, layers="sweep",
                                  target="instruction"),
        seed=seed,
        token_unit="last",
    )


# ── 개입 타깃 축 ───────────────────────────────────────────────────────────

def test_intervention_target_validates():
    with pytest.raises(ValueError):
        Intervention(kind=InterventionKind.VALUE, layers="sweep", target="bogus")


def test_none_intervention_forbids_nondefault_target():
    with pytest.raises(ValueError):
        Intervention(kind=InterventionKind.NONE, target="instruction")


def test_target_roundtrips_through_dict():
    c = _cond()
    assert Condition.from_dict(c.to_dict()).intervention.target == "instruction"


def test_slug_reveals_instruction_target_and_block():
    s = _cond(Notation.CAMEL, block=7).slug()
    assert "-instr" in s                 # 지침 타깃 표시
    assert "ins-pos-camel" in s          # 방향(camel->snake)
    assert "-b7" in s                    # 블록 커버(504)


# ── 반대 지침 donor 구성 ───────────────────────────────────────────────────

@pytest.mark.parametrize("target,other", [
    (Notation.CAMEL, Notation.SNAKE),
    (Notation.SNAKE, Notation.CAMEL),
])
def test_setup_crosses_directive_words(target, other):
    s = _preference_setup_instruction(_cond(target))
    assert s["viol_names"] == [notation_word(target)]
    assert s["donor_names"] == [notation_word(other)]
    assert s["candidate_compliant"].endswith("Duplicates") == (target is Notation.CAMEL)
    assert s["donor_kind"] == "opposite_instruction"


def test_base_and_opposite_instruction_differ_and_share_preceding():
    s = _preference_setup_instruction(_cond(Notation.CAMEL))
    base_sys = s["viol_messages"][0]["content"]
    opp_sys = s["comp_messages"][0]["content"]
    assert base_sys != opp_sys
    assert "camelCase" in base_sys and "snake_case" in opp_sys
    assert s["viol_messages"][1]["content"] == s["comp_messages"][1]["content"]


def test_blocks_draw_different_preceding():
    # 다른 블록은 다른 이름(504 커버). 치환 대상 지시어는 블록 무관.
    s0 = _preference_setup_instruction(_cond(Notation.CAMEL, block=0))
    s41 = _preference_setup_instruction(_cond(Notation.CAMEL, block=41))
    assert s0["viol_names"] == s41["viol_names"] == ["camelCase"]
    assert s0["viol_messages"][1]["content"] != s41["viol_messages"][1]["content"]


def test_all_violation_preceding_supported():
    s6 = _preference_setup_instruction(_cond(Notation.CAMEL, n_compliant=6))
    s0 = _preference_setup_instruction(_cond(Notation.CAMEL, n_compliant=0))
    assert s6["viol_names"] == s0["viol_names"] == ["camelCase"]
    assert s6["viol_messages"][1]["content"] != s0["viol_messages"][1]["content"]


# ── literal span_kind: 지시어 토큰 위치 ────────────────────────────────────

def test_literal_span_targets_first_occurrence():
    text = "write names in camelCase. styles: camelCase or snake_case."
    off = [(15, 24), (34, 43), (47, 57)]
    got = _locate_target_tokens(text, off, ["camelCase"], span_kind="literal")
    assert got == [[0]]                     # rule 토큰만(첫 등장)


def test_literal_span_missing_word_is_empty():
    got = _locate_target_tokens("no directive here", [(0, 2)], ["camelCase"],
                                span_kind="literal")
    assert got == [[]]


def test_def_name_span_unchanged_by_new_param():
    text = "def parseHeader(v): return v"
    off = [(0, 3), (4, 9), (9, 15), (15, 16)]
    got = _locate_target_tokens(text, off, ["parseHeader"])  # default def_name
    assert got == [[1, 2]]
