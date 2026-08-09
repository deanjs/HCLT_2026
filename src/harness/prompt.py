"""프롬프트 조립 — 선행 코드, 지침 문장, 순차 생성 메시지.

생성 경로(개입 없음)의 build_preceding / build_instruction / build_prompt 단계가
여기 로직을 부른다. 모델 의존성 없이 순수 문자열을 만든다(테스트 가능).
"""

from __future__ import annotations

import random
from typing import Optional

from .conditions import Composition, Condition, InstructionForm, Notation
from .tasks import CLONE_TASK, DISTINCT_TASKS, GENERATION_TASKS, TaskSpec

_STYLE = {Notation.CAMEL: "camelCase", Notation.SNAKE: "snake_case"}


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
    return "You are helping extend an existing Python module.\n" + rule + "\n" + closed


def build_preceding_code(condition: Condition) -> str:
    """선행 코드 12개 블록. n_compliant개는 목표 표기, 나머지는 위반 표기, 위치 셔플."""
    p = condition.preceding
    target = condition.instruction.target_notation
    violation = condition.instruction.violation_notation
    notations = [target] * p.n_compliant + [violation] * (p.n_functions - p.n_compliant)
    random.Random(condition.seed).shuffle(notations)

    if p.composition is Composition.CLONE:
        # 같은 과제를 인덱스만 바꿔 12복제. 표기는 notations[i].
        funcs = [CLONE_TASK.render(nt, idx=i + 1) for i, nt in enumerate(notations)]
    else:  # DISTINCT — 서로 다른 과제 12개. 표기는 notations[i], 인덱스 없음.
        if p.n_functions > len(DISTINCT_TASKS):
            raise ValueError(
                f"distinct 과제 풀({len(DISTINCT_TASKS)})이 n_functions({p.n_functions})보다 작다"
            )
        funcs = [DISTINCT_TASKS[i].render(nt) for i, nt in enumerate(notations)]
    return "\n\n".join(funcs)


def first_user_message(condition: Condition) -> str:
    """첫 사용자 턴: 선행 코드 + 첫 생성 과제 요청."""
    preceding = build_preceding_code(condition)
    task = GENERATION_TASKS[0]
    return (
        "Here is the current module:\n\n"
        f"```python\n{preceding}\n```\n\n"
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
