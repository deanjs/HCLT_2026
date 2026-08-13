"""step 4(RQ3 인과 — 지침 지시어 KV 치환) 단위 테스트 — 모델 없이 조립·위치만 검증.

KV 캐시 편집 자체는 GPU가 필요하므로 여기서 다루지 않고, step4가 stepC 개입 구조를
**지침 지시어 토큰 + 반대 지침 donor**로 재배선한 부분의 건전성만 본다:
  - 개입 타깃 축(conditions.Intervention.target) 검증·직렬화·슬러그
  - _preference_setup_instruction 이 방향별로 치환 대상/공여 지시어를 맞게 교차하는가
  - literal span_kind 가 지침 rule 문장의 지시어 토큰(첫 등장)을 집는가
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
from harness.prompt import build_instruction_text, notation_word
from harness.runner import _preference_setup_instruction

MODEL = ModelSpec(name="Qwen/Qwen2.5-Coder-3B-Instruct", family="qwen")


def _cond(target=Notation.CAMEL, n_compliant=6, seed=0):
    return Condition(
        model=MODEL,
        preceding=PrecedingCode(n_compliant=n_compliant, n_functions=12,
                                composition=Composition.POOL),
        instruction=Instruction(form=InstructionForm.POSITIVE, target_notation=target),
        intervention=Intervention(kind=InterventionKind.VALUE, layers="sweep",
                                  target="instruction"),
        seed=seed,
        token_unit="last",
    )


# ── 개입 타깃 축 ───────────────────────────────────────────────────────────

def test_intervention_target_validates():
    with pytest.raises(ValueError):
        Intervention(kind=InterventionKind.VALUE, layers="sweep", target="bogus")


def test_none_intervention_forbids_nondefault_target():
    # kind=NONE에는 지침 타깃을 두지 않는다(개입 없음인데 대상만 지정 = 모순).
    with pytest.raises(ValueError):
        Intervention(kind=InterventionKind.NONE, target="instruction")


def test_target_roundtrips_through_dict():
    c = _cond()
    assert Condition.from_dict(c.to_dict()).intervention.target == "instruction"


def test_slug_reveals_instruction_target_and_direction():
    s = _cond(Notation.CAMEL).slug()
    assert "-instr" in s                 # 지침 타깃 표시(코드 타깃 파일과 구분)
    assert "ins-pos-camel" in s          # 방향(camel→snake 치환)


# ── 반대 지침 donor 구성 ───────────────────────────────────────────────────

@pytest.mark.parametrize("target,other", [
    (Notation.CAMEL, Notation.SNAKE),
    (Notation.SNAKE, Notation.CAMEL),
])
def test_setup_crosses_directive_words(target, other):
    s = _preference_setup_instruction(_cond(target))
    # 치환 대상 = 조건 지침의 지시어, 공여 = 반대 지침의 지시어.
    assert s["viol_names"] == [notation_word(target)]
    assert s["donor_names"] == [notation_word(other)]
    # 후보 두 이름은 방향에 맞게(target 준수판 / 위반판).
    assert s["candidate_compliant"].endswith("Duplicates") == (target is Notation.CAMEL)
    assert s["donor_kind"] == "opposite_instruction"


def test_base_and_opposite_instruction_differ_and_share_preceding():
    s = _preference_setup_instruction(_cond(Notation.CAMEL))
    base_sys = s["viol_messages"][0]["content"]
    opp_sys = s["comp_messages"][0]["content"]
    assert base_sys != opp_sys                       # 지침만 반대로
    # rule 문장 지시어가 교차(base=camel rule, opp=snake rule)
    assert "camelCase" in base_sys and "snake_case" in opp_sys
    # 선행(user 메시지)은 두 프롬프트가 공유 = 같은 코드, 반대 지침의 통제된 대조
    assert s["viol_messages"][1]["content"] == s["comp_messages"][1]["content"]


def test_all_violation_preceding_supported():
    # 선행 축을 6/6 균형과 전부위반(0 준수) 둘 다 태울 수 있어야 한다(대조 칸).
    s6 = _preference_setup_instruction(_cond(Notation.CAMEL, n_compliant=6))
    s0 = _preference_setup_instruction(_cond(Notation.CAMEL, n_compliant=0))
    # 지시어 치환 대상은 선행 개수와 무관(지침만 바뀜)
    assert s6["viol_names"] == s0["viol_names"] == ["camelCase"]
    # 선행 코드 내용은 달라야(6/6 vs 전부 snake)
    assert s6["viol_messages"][1]["content"] != s0["viol_messages"][1]["content"]


# ── literal span_kind: 지시어 토큰 위치 ────────────────────────────────────

def test_literal_span_targets_first_occurrence():
    # rule 문장이 closed 문장보다 앞서므로 첫 등장이 실제 지시어(rule) 위치.
    text = "write names in camelCase. styles: camelCase or snake_case."
    #        0123456789...          ^15                ^34
    off = [(15, 24), (34, 43), (47, 57)]   # camelCase(rule), camelCase(closed), snake_case
    got = _locate_target_tokens(text, off, ["camelCase"], span_kind="literal")
    assert got == [[0]]                     # rule 토큰만(첫 등장), closed 미포함


def test_literal_span_missing_word_is_empty():
    got = _locate_target_tokens("no directive here", [(0, 2)], ["camelCase"],
                                span_kind="literal")
    assert got == [[]]                      # 못 찾으면 빈 리스트 → align이 스킵


def test_def_name_span_unchanged_by_new_param():
    # 기존 코드 이름 경로(default span_kind)는 그대로 작동해야 한다(회귀 방지).
    text = "def parseHeader(v): return v"
    off = [(0, 3), (4, 9), (9, 15), (15, 16)]  # 'def' 'parse' 'Header' '('
    got = _locate_target_tokens(text, off, ["parseHeader"])  # default def_name
    assert got == [[1, 2]]                   # 이름 서브토큰만('def '·'(' 제외)
