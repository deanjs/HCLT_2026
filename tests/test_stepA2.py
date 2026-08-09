"""step A-2(실제 저장소 코드) 단위 테스트 — 스키마·실파일 로드·언어 인식."""

import pytest

from harness import (
    GENERATION_TASKS,
    build_instruction_text,
    build_preceding_code,
    classify_name,
    first_function_name,
    run,
)
from harness.conditions import (
    Condition,
    Instruction,
    InstructionForm,
    ModelSpec,
    Notation,
    PrecedingCode,
    Source,
)
from harness.prompt import first_user_message

MODEL = ModelSpec(name="Qwen/Qwen2.5-Coder-3B-Instruct", family="qwen")


def _repo_cond(lang, repo_file, target=Notation.CAMEL):
    return Condition(
        model=MODEL,
        preceding=PrecedingCode(n_compliant=0, source=Source.REPO,
                                repo_lang=lang, repo_file=repo_file),
        instruction=Instruction(form=InstructionForm.POSITIVE, target_notation=target),
        seed=0,
    )


# ── 스키마 ────────────────────────────────────────────────────────────────

def test_repo_requires_lang_and_file():
    with pytest.raises(ValueError):
        PrecedingCode(n_compliant=0, source=Source.REPO, repo_lang="python")  # file 없음
    with pytest.raises(ValueError):
        PrecedingCode(n_compliant=0, source=Source.REPO, repo_file="python/x.py")  # lang 없음


def test_synthetic_forbids_repo_fields():
    with pytest.raises(ValueError):
        PrecedingCode(n_compliant=2, source=Source.SYNTHETIC, repo_lang="python")


def test_repo_slug_shows_lang_and_file():
    slug = _repo_cond("python", "python/textwrap.py").slug()
    assert "pre-repo-python-textwrap" in slug


# ── 실파일 로드 ────────────────────────────────────────────────────────────

def test_build_preceding_loads_real_python():
    code = build_preceding_code(_repo_cond("python", "python/textwrap.py"))
    assert "def " in code and "snake" not in code[:0]  # 실제 파일 내용
    assert len(code) > 500


def test_build_preceding_loads_real_js():
    code = build_preceding_code(_repo_cond("javascript", "javascript/utils.js"))
    assert "function" in code or "=>" in code
    assert len(code) > 500


# ── 언어 인식 프롬프트 ─────────────────────────────────────────────────────

def test_instruction_language_label():
    assert "Python module" in build_instruction_text(_repo_cond("python", "python/textwrap.py"))
    assert "JavaScript module" in build_instruction_text(_repo_cond("javascript", "javascript/utils.js"))


def test_user_message_fence_matches_language():
    assert "```javascript" in first_user_message(_repo_cond("javascript", "javascript/utils.js"))
    assert "```python" in first_user_message(_repo_cond("python", "python/textwrap.py"))


# ── 생성 경로 (fake JS 생성) ──────────────────────────────────────────────

def _fake_js_gen(notation):
    def gen(messages):
        task = next(t for t in GENERATION_TASKS if t.description in messages[-1]["content"])
        return f"function {task.name(notation)}({', '.join(task.params)}) {{ return null; }}"
    return gen


def test_run_repo_js_uses_js_parser():
    out = run(_repo_cond("javascript", "javascript/utils.js"),
              generate_fn=_fake_js_gen(Notation.CAMEL))
    assert out.metrics.extra["turn_notations"] == ["camel", "camel", "camel"]
    assert out.metrics.compliance_rate == 1.0
    assert all("function" in t for t in out.metrics.extra["turn_texts"])  # 원문 저장


def test_run_repo_js_snake_detected():
    out = run(_repo_cond("javascript", "javascript/utils.js", target=Notation.CAMEL),
              generate_fn=_fake_js_gen(Notation.SNAKE))
    assert out.metrics.extra["turn_notations"] == ["snake", "snake", "snake"]
    assert out.metrics.compliance_rate == 0.0   # 목표 camel인데 snake 생성 → 위반
