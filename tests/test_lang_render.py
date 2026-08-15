"""step1 언어 다양성 — 합성 코드 파이썬/JS 렌더·파싱·슬러그·생성 턴 수 검증(모델 없이).

step1(준수율 절벽)은 같은 504 이름을 파이썬과 JS 두 언어로 렌더해 언어 다양성을 본다(§3).
이름·본문은 언어 무관, 껍데기(def↔function)만 바뀌고, 생성물에서 첫 함수명을 뽑아 표기 판정한다.
"""

from dataclasses import replace

from harness.conditions import (
    Composition, Condition, Instruction, InstructionForm,
    ModelSpec, Notation, PrecedingCode,
)
from harness.prompt import build_preceding_code, first_user_message
from harness.tasks import NAME_PAIR_POOL
from harness.naming import first_function_name, classify_name
from harness.runner import run

MODEL = ModelSpec(name="Qwen/Qwen2.5-Coder-3B-Instruct", family="qwen")


def _cond(lang):
    return Condition(
        model=MODEL,
        preceding=PrecedingCode(n_compliant=3, n_functions=12,
                                composition=Composition.POOL, pool_block=0, lang=lang),
        instruction=Instruction(form=InstructionForm.POSITIVE, target_notation=Notation.CAMEL),
        seed=42,
    )


def test_pool_render_python_and_js():
    py = build_preceding_code(_cond(None))       # 기본 = python
    js = build_preceding_code(_cond("javascript"))
    assert "def " in py and "function " not in py
    assert "function " in js and "{" in js and "}" in js
    assert "def " not in js
    # 이름은 두 언어에서 동일(껍데기만 다름) — 첫 함수 이름이 py/js에 똑같이 등장
    py_first = first_function_name(py, "python")
    js_first = first_function_name(js, "javascript")
    assert py_first and py_first == js_first


def test_js_render_body_valid():
    js = build_preceding_code(_cond("javascript"))
    first = js.split("\n\n")[0]
    assert first.startswith("function ")
    assert first.rstrip().endswith("}")
    assert "return value;" in first   # 본문에 세미콜론


def test_slug_separates_language():
    assert "-javascript" in _cond("javascript").slug()
    assert "-javascript" not in _cond(None).slug()   # python은 슬러그에 안 붙음(기존 불변)


def test_lang_roundtrip_serialization():
    c = _cond("javascript")
    assert Condition.from_dict(c.to_dict()).preceding.lang == "javascript"


def test_lang_validation_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        PrecedingCode(n_compliant=3, composition=Composition.POOL, lang="ruby")


def test_js_fence_in_prompt():
    assert "```javascript" in first_user_message(_cond("javascript"))
    assert "```python" in first_user_message(_cond(None))


def test_generation_parses_js_name_and_compliance():
    c = _cond("javascript")
    fake = lambda msgs: "Sure! function removeDuplicates(items) { return items; }"
    out = run(c, generate_fn=fake, max_turns=1)
    ex = out.metrics.extra
    assert ex["turn_names"] == ["removeDuplicates"]
    assert ex["turn_notations"] == ["camel"]
    assert ex["first_compliant"] is True
    assert ex["subsequent_violation_rate"] is None   # 1턴이라 자기증폭 없음
    assert "function removeDuplicates" in ex["turn_texts"][0]


def test_max_turns_limits_turns():
    c = _cond(None)
    fake = lambda msgs: "def parseHeader(x): return x"
    assert len(run(c, generate_fn=fake, max_turns=1).metrics.extra["turn_names"]) == 1
    assert len(run(c, generate_fn=fake, max_turns=2).metrics.extra["turn_names"]) == 2


def test_js_name_parser_handles_arrow_and_function():
    assert first_function_name("const parseHeader = (x) => x", "javascript") == "parseHeader"
    assert first_function_name("function parse_header(x) { return x; }", "javascript") == "parse_header"
