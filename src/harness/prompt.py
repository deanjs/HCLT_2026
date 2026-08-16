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


def notation_word(notation: Notation) -> str:
    """지침 문장에 등장하는 표기 지시어 단어('camelCase'/'snake_case').

    step4(지침 지시어 KV 치환)가 치환 대상 문자열을 지목할 때 쓴다. 지침 조립과
    지시어 문자열을 한 곳에서 관리해 중복을 막는다(CLAUDE.md §3).
    """
    return _STYLE[notation]

# 번들 실파일 위치: <repo_root>/data/repo_files/ (src/harness/prompt.py → parents[2])
_REPO_FILES_DIR = Path(__file__).resolve().parents[2] / "data" / "repo_files"
_LANG_LABEL = {"python": "Python", "javascript": "JavaScript", "js": "JavaScript"}


def _lang(condition: Condition) -> str:
    """이 조건의 코드 언어. 실코드는 repo_lang, 합성은 preceding.lang(없으면 python).

    step1 언어 다양성(파이썬+JS)에서 합성 코드도 언어를 가질 수 있다(§3).
    """
    p = condition.preceding
    if p.source is Source.REPO:
        return p.repo_lang
    return p.lang or "python"


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


# 후보열거 문장의 고정 앵커(build_instruction_text의 closed 문장). 이 앞=규칙문, 뒤=후보열거.
_CANDIDATE_ANCHOR = "one of two styles only:"


def instruction_notation_spans(condition: Condition) -> dict[str, list[tuple[int, int]]]:
    """지침 문장 안 표기어(camelCase/snake_case)를 **역할별로 분리**해 위치를 돌려준다.

    지침은 표기어를 두 곳에서 언급한다:
      · 규칙문   : "...write function names in camelCase." — 실제 지시(요구/금지) **1개**
      · 후보열거 : "...one of two styles only: camelCase or snake_case." — 대칭 나열(**둘 다 1개씩**)

    그래서 요구 표기어는 규칙문+후보열거로 **2회**, 반대 표기어는 후보열거로 **1회** 나온다.
    그냥 합쳐 재면(step4 초판) 요구어가 "많이 나와서" 부풀려져 불공정하다. 이 함수가 규칙문
    지시어와 후보열거를 나눠, 공정한 비교(규칙문 지시어 vs 코드이름, 후보 camel vs snake)를 가능케 한다.

    반환(키 → 문자 구간 목록, **지침 문장 내부 offset 기준**. model.py가 프롬프트 전체 기준으로 재기준):
      instr_rule_word  : 규칙문의 지시어(요구 또는 금지 표기어) — "지침을 보는가"의 핵심 신호
      instr_cand_camel : 후보열거의 camelCase
      instr_cand_snake : 후보열거의 snake_case
    """
    text = build_instruction_text(condition)
    camel, snake = _STYLE[Notation.CAMEL], _STYLE[Notation.SNAKE]
    ci = text.find(_CANDIDATE_ANCHOR)   # 후보열거 시작(그 앞은 규칙문). 못 찾으면 -1.

    def occ(word: str) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        s = 0
        while True:
            i = text.find(word, s)
            if i < 0:
                break
            out.append((i, i + len(word)))
            s = i + len(word)
        return out

    rule_word: list[tuple[int, int]] = []
    cand_camel: list[tuple[int, int]] = []
    cand_snake: list[tuple[int, int]] = []
    for a, b in occ(camel):
        (cand_camel if ci >= 0 and a >= ci else rule_word).append((a, b))
    for a, b in occ(snake):
        (cand_snake if ci >= 0 and a >= ci else rule_word).append((a, b))
    return {
        "instr_rule_word": rule_word,
        "instr_cand_camel": cand_camel,
        "instr_cand_snake": cand_snake,
    }


def _pool_indices(condition: Condition) -> list[int]:
    """POOL 선행이 쓸 이름 풀 인덱스 n_functions개(블록 커버리지).

    풀을 n_functions개씩 나눠 블록 인덱스로 고른다: 블록 b → [b*n, (b+1)*n).
    여러 블록으로 풀 전체(504개)를 **중복 없이** 덮는다(§3 500 커버리지).
    seed는 수량이 아니라 블록 내 **순서(위치·이웃)** 를 섞어 문맥 강건성 변주를 준다.
    (build_preceding_code·preceding_specs·unrelated_specs가 이 하나를 공유해 어긋나지 않는다.)
    """
    p = condition.preceding
    n = p.n_functions
    start = p.pool_block * n
    if start + n > len(NAME_PAIR_POOL):
        raise ValueError(
            f"pool_block({p.pool_block})가 이름 풀 범위를 넘는다: "
            f"[{start}, {start + n}) > 풀 {len(NAME_PAIR_POOL)}"
        )
    idxs = list(range(start, start + n))
    random.Random(condition.seed).shuffle(idxs)   # 위치·이웃 변주
    return idxs


def build_preceding_code(condition: Condition) -> str:
    """선행 코드. 합성이면 12개 블록(clone/distinct), 실코드면 번들 파일 원문."""
    p = condition.preceding
    if p.source is Source.REPO:
        return _load_repo_file(p.repo_file)   # 실코드는 조작 없이 원문 그대로
    target = condition.instruction.target_notation
    violation = condition.instruction.violation_notation
    notations = [target] * p.n_compliant + [violation] * (p.n_functions - p.n_compliant)
    random.Random(condition.seed).shuffle(notations)
    lang = _lang(condition)   # 합성 코드 렌더 언어(python/javascript) — step1 언어 다양성

    if p.composition is Composition.CLONE:
        # 같은 과제를 인덱스만 바꿔 12복제. 표기는 notations[i].
        funcs = [CLONE_TASK.render(nt, idx=i + 1, lang=lang) for i, nt in enumerate(notations)]
    elif p.composition is Composition.DISTINCT:
        # 서로 다른 과제 12개(고정 풀 앞에서부터). 표기는 notations[i], 인덱스 없음.
        if p.n_functions > len(DISTINCT_TASKS):
            raise ValueError(
                f"distinct 과제 풀({len(DISTINCT_TASKS)})이 n_functions({p.n_functions})보다 작다"
            )
        funcs = [DISTINCT_TASKS[i].render(nt, lang=lang) for i, nt in enumerate(notations)]
    else:  # POOL — stepB 균형 관측: 이름 풀을 블록으로 덮는다(전체 504). seed는 표기·위치 변주.
        idxs = _pool_indices(condition)
        funcs = [NAME_PAIR_POOL[j].render(nt, lang=lang) for j, nt in zip(idxs, notations)]
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


# ── step C(개입)용 헬퍼 ────────────────────────────────────────────────────

def preceding_specs(condition: Condition) -> list[tuple[TaskSpec, Notation]]:
    """선행 코드의 (TaskSpec, 표기) 목록을 build_preceding_code와 **동일 규칙**으로 재구성.

    개입에서 위반 이름과 그 준수판을 알아야 하므로 구조를 노출한다. 합성 POOL/DISTINCT만
    지원(CLONE은 idx가 필요하고 step C에서 쓰지 않는다).
    """
    p = condition.preceding
    if p.source is Source.REPO:
        raise ValueError("preceding_specs는 합성 선행에만 쓴다")
    target = condition.instruction.target_notation
    violation = condition.instruction.violation_notation
    notations = [target] * p.n_compliant + [violation] * (p.n_functions - p.n_compliant)
    random.Random(condition.seed).shuffle(notations)
    if p.composition is Composition.POOL:
        idxs = _pool_indices(condition)
        specs = [NAME_PAIR_POOL[j] for j in idxs]
    elif p.composition is Composition.DISTINCT:
        specs = [DISTINCT_TASKS[i] for i in range(p.n_functions)]
    else:
        raise ValueError("preceding_specs는 POOL/DISTINCT만 지원(CLONE은 idx 필요)")
    return list(zip(specs, notations))


def render_preceding(specs_notations: list[tuple[TaskSpec, Notation]]) -> str:
    """(TaskSpec, 표기) 목록을 def 블록들로 렌더링(선행 코드 문자열)."""
    return "\n\n".join(spec.render(nt) for spec, nt in specs_notations)


def user_message_with_preceding(condition: Condition, preceding_text: str) -> str:
    """임의의 선행 코드 문자열 + 첫 생성 요청으로 user 메시지를 만든다(개입용)."""
    task = GENERATION_TASKS[0]
    fence = "javascript" if _lang(condition) in ("js", "javascript") else "python"
    return (
        "Here is the current module:\n\n"
        f"```{fence}\n{preceding_text}\n```\n\n"
        f"Add a function that {task.description}."
    )


def unrelated_specs(condition: Condition, n: int) -> list[TaskSpec]:
    """조건의 선행에 쓰이지 않은 풀 이름 n개(무관 코드 통제용). POOL 전용."""
    if condition.preceding.composition is not Composition.POOL:
        raise ValueError("unrelated_specs는 POOL 선행에만 쓴다")
    used = set(_pool_indices(condition))
    avail = [j for j in range(len(NAME_PAIR_POOL)) if j not in used]
    if n > len(avail):
        raise ValueError(f"무관 이름 부족: 요청 {n}, 가용 {len(avail)}")
    pick = random.Random(condition.seed + 10007).sample(avail, n)
    return [NAME_PAIR_POOL[j] for j in pick]


def initial_messages(condition: Condition) -> list[dict[str, str]]:
    """system(지침) + 첫 user 턴까지의 초기 메시지."""
    return [
        {"role": "system", "content": build_instruction_text(condition)},
        {"role": "user", "content": first_user_message(condition)},
    ]
