"""과제 스펙 — 선행 코드와 생성 요청에 쓰는 합성 함수 정의.

각 과제는 단어 목록으로 이름을 camel/snake 렌더링한다. 표기가 구분되도록
모든 과제는 **두 단어 이상**이다.

- CLONE_TASK      : step A 선행용. 같은 일을 하는 12개를 인덱스로 찍어낸다.
- GENERATION_TASKS: 모델이 새로 작성하는 3개(서로 다르고 선행과도 다르다). 측정 대상.

step A-1(선행 distinct)용 과제 풀은 그 step에서 이 파일에 추가한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .conditions import Notation
from .naming import render_name


@dataclass(frozen=True)
class TaskSpec:
    words: tuple[str, ...]     # 이름을 이루는 단어들 (예: ("scale","value"))
    description: str           # 자연어 설명 (생성 요청 문장에 그대로 들어간다)
    params: tuple[str, ...]    # 매개변수 이름
    body: tuple[str, ...]      # 본문 줄 (들여쓰기 없이)

    def name(self, notation: Notation, idx: Optional[int] = None) -> str:
        return render_name(self.words, notation, idx)

    def render(self, notation: Notation, idx: Optional[int] = None) -> str:
        """완성된 def 블록 문자열."""
        sig = ", ".join(self.params)
        head = f"def {self.name(notation, idx)}({sig}):"
        return "\n".join([head] + [f"    {ln}" for ln in self.body])


# 선행용 clone 과제 1개 — 같은 일(값 × 배수)을 12번 복제해 이름만 바꾼다.
CLONE_TASK = TaskSpec(
    words=("scale", "value"),
    description="returns the value multiplied by a factor",
    params=("value", "factor"),
    body=("return value * factor",),
)

# 생성용 과제 3개 — 서로 다르고 선행과도 다르다. 모델이 이름을 직접 선택한다.
GENERATION_TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        words=("clamp", "number"),
        description="clamps a number between a low and high bound",
        params=("number", "low", "high"),
        body=("return max(low, min(number, high))",),
    ),
    TaskSpec(
        words=("count", "vowels"),
        description="counts the vowels in a string",
        params=("text",),
        body=('return sum(c in "aeiou" for c in text.lower())',),
    ),
    TaskSpec(
        words=("merge", "dicts"),
        description="merges two dictionaries",
        params=("a", "b"),
        body=("return {**a, **b}",),
    ),
)
