"""프롬프트 조립 — 선행 코드, 지침 문장, 순차 생성 메시지.

생성 경로(개입 없음)의 build_preceding / build_instruction / build_prompt 단계가
여기 로직을 부른다. 모델 의존성 없이 순수 문자열을 만든다(테스트 가능).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from .conditions import Composition, Condition, InstructionForm, Notation, Source
from .tasks import CLONE_TASK, DISTINCT_TASKS, GENERATION_TASKS, NAME_PAIR_POOL, TaskSpec

_STYLE = {Notation.CAMEL: "camelCase", Notation.SNAKE: "snake_case"}

# 번들 실파일 위치: <repo_root>/data/repo_files/ (src/harness/prompt.py → parents[2])
_REPO_FILES_DIR = Path(__file__).resolve().parents[2] / "data" / "repo_files"
_LANG_LABEL = {"python": "Python", "javascript": "JavaScript", "js": "JavaScript"}


def _lang(condition: Condition) -> str:
    """이 조건의 코드 언어. 합성은 python, 실코드는 repo_lang."""
    p = condition.preceding
    return p.repo_lang if p.source is Source.REPO else "python"


def _load_repo_file(repo_file: str) -> str:
    return (_REPO_FILES_DIR / repo_file).read_text(encoding="utf-8")


def build_instruction_text(condition: Condition) -> str:
    """지침(system) 문장. 형식(긍정/부정)·목표 표기·어조·닫힌 후보를 반영한다."""
    ins = condition.instruction
    target = _STYLE[ins.target_notation]
    strong = ins.strength.value == "strong"
    if ins.form is InstructionForm.POSITIVE:
        rule = (f"In this project, always write function names in {target}."
                if strong else
                f"In this project we generally write function names in {target}.")
    else:
        viol = _STYLE[ins.violation_notation]
        rule = (f"In this project, never write function names in {viol}."
                if strong else
                f"In this project, please avoid writing function names in {viol}.")
    c0, c1 = (_STYLE[c] for c in ins.candidates)
    closed = f"Every function name uses one of two styles only: {c0} or {c1}."
    lang = _LANG_LABEL.get(_lang(condition), "Python")
    return f"You are helping extend an existing {lang} module.\n" + rule + "\n" + closed


def build_preceding_code(condition: Condition) -> str:
    """선행 코드. 합성이면 12개 블록(clone/distinct), 실코드면 번들 파일 원문."""
    p = condition.preceding
    if p.source is Source.REPO:
        return _load_repo_file(p.repo_file)   # 실코드는 조작 없이 원문 그대로
    target = condition.instruction.target_notation
    violation = condition.instruction.violation_notation
    notations = [target] * p.n_compliant + [violation] * (p.n_functions - p.n_compliant)
    random.Random(condition.seed).shuffle(notations)

    if p.composition is Composition.CLONE:
        # 같은 과제를 인덱스만 바꿔 12복제. 표기는 notations[i].
        funcs = [CLONE_TASK.render(nt, idx=i + 1) for i, nt in enumerate(notations)]
    elif p.composition is Composition.DISTINCT:
        # 서로 다른 과제 12개(고정 풀 앞에서부터). 표기는 notations[i], 인덱스 없음.
        if p.n_functions > len(DISTINCT_TASKS):
            raise ValueError(
                f"distinct 과제 풀({len(DISTINCT_TASKS)})이 n_functions({p.n_functions})보다 작다"
            )
        funcs = [DISTINCT_TASKS[i].render(nt) for i, nt in enumerate(notations)]
    else:  # POOL — stepB 균형 관측: 이름 짝 풀(~80)에서 seed로 서로 다른 n개를 뽑는다.
        if p.n_functions > len(NAME_PAIR_POOL):
            raise ValueError(
                f"이름 풀({len(NAME_PAIR_POOL)})이 n_functions({p.n_functions})보다 작다"
            )
        idxs = random.Random(condition.seed).sample(range(len(NAME_PAIR_POOL)), p.n_functions)
        funcs = [NAME_PAIR_POOL[j].render(nt) for j, nt in zip(idxs, notations)]
    return "\n\n".join(funcs)


def first_user_message(condition: Condition) -> str:
    """첫 사용자 턴: 선행 코드 + 첫 생성 과제 요청."""
    preceding = build_preceding_code(condition)
    task = GENERATION_TASKS[0]
    fence = "javascript" if _lang(condition) in ("js", "javascript") else "python"
    return (
        "Here is the current module:\n\n"
        f"```{fence}\n{preceding}\n```\n\n"
        f"Add a function that {task.description}."
    )


def next_user_message(turn: int) -> str:
    """이후 사용자 턴: 다음 생성 과제만 요청."""
    task = GENERATION_TASKS[turn]
    return f"Add a function that {task.description}."


def initial_messages(condition: Condition) -> list[dict[str, str]]:
    """system(지침) + 첫 user 턴까지의 초기 메시지."""
    return [
        {"role": "system", "content": build_instruction_text(condition)},
        {"role": "user", "content": first_user_message(condition)},
    ]
