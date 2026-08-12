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

# 선행용 distinct 과제 12개 (step A-1) — 서로 다른 작업. 생성 3개와 겹치지 않는다.
# 모두 두 단어 이상이라 camel/snake가 구분된다. 인덱스 없이 이름 그대로 렌더링.
DISTINCT_TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(("sum", "list"), "returns the sum of a list", ("items",),
             ("return sum(items)",)),
    TaskSpec(("max", "value"), "returns the largest value in a list", ("items",),
             ("return max(items)",)),
    TaskSpec(("reverse", "string"), "reverses a string", ("text",),
             ("return text[::-1]",)),
    TaskSpec(("is", "even"), "checks whether a number is even", ("number",),
             ("return number % 2 == 0",)),
    TaskSpec(("to", "upper"), "converts a string to upper case", ("text",),
             ("return text.upper()",)),
    TaskSpec(("first", "item"), "returns the first item of a list", ("items",),
             ("return items[0]",)),
    TaskSpec(("last", "item"), "returns the last item of a list", ("items",),
             ("return items[-1]",)),
    TaskSpec(("square", "number"), "returns the square of a number", ("number",),
             ("return number * number",)),
    TaskSpec(("join", "words"), "joins a list of words with spaces", ("words",),
             ('return " ".join(words)',)),
    TaskSpec(("strip", "spaces"), "strips leading and trailing spaces", ("text",),
             ("return text.strip()",)),
    TaskSpec(("double", "value"), "returns the value doubled", ("value",),
             ("return value * 2",)),
    TaskSpec(("abs", "diff"), "returns the absolute difference of two numbers", ("a", "b"),
             ("return abs(a - b)",)),
)

# ── stepB 관측용 이름 짝 풀 ────────────────────────────────────────────────
# RQ2 관측(stepB)은 선행에 camel/snake 이름을 6/6으로 균형 배치하고, 두 표기 그룹의
# 어텐션·‖v‖·‖av‖를 비교한다(docs/stepB/plan.md 표본 설계). seed만 늘리면 배치 순열만
# 바뀌고 이름 정체성은 고정되므로, "서로 다른 이름"을 충분히 확보해야 효과가 특정 단어의
# 우연이 아님이 드러난다. 그래서 두 단어(동사×명사) 조합으로 ~80개 짝 풀을 만든다.
#
# 본문·매개변수는 **모든 항목이 동일**하다(값을 그대로 반환). 이는 의도된 통제다 —
# 선행 함수들 사이에서 변하는 것을 오직 **이름 표기**로 한정해, 어텐션 차이가 내용이
# 아니라 형태에서 온다는 stepB의 비교를 깨끗하게 만든다.

# 주의: 생성 과제(GENERATION_TASKS: remove/count/merge)의 단어와 겹치지 않게 한다.
# 선행에 생성 대상과 같은 이름이 있으면 모델이 베끼는 혼입이 생긴다. (merge 제외 → expand)
# 동사·명사 각 50개(50×50 어휘 확장). **앞 20개는 원래 순서 그대로** 두고 30개씩 덧붙인다.
# → 원래 80개 이름을 새 풀의 앞부분에 그대로 보존(부분집합, 연속성). 생성 과제 단어
# (remove/count/merge, duplicates/vowels/dicts)와 겹치지 않는다.
_POOL_VERBS: tuple[str, ...] = (
    # 원래 20 (순서 불변)
    "parse", "build", "fetch", "render", "encode", "decode", "expand", "split",
    "filter", "format", "resolve", "compute", "extract", "validate", "normalize",
    "serialize", "append", "compress", "flatten", "collect",
    # 추가 30
    "transform", "aggregate", "dispatch", "register", "allocate", "initialize",
    "schedule", "convert", "generate", "inspect", "traverse", "partition",
    "sanitize", "hydrate", "marshal", "deserialize", "reconcile", "prefetch",
    "rollback", "synchronize", "throttle", "escalate", "propagate", "materialize",
    "reindex", "rebalance", "tokenize", "annotate", "checkpoint", "bootstrap",
)
_POOL_NOUNS: tuple[str, ...] = (
    # 원래 20 (순서 불변)
    "header", "payload", "token", "buffer", "record", "node", "entry", "config",
    "session", "request", "packet", "matrix", "vector", "index", "cursor", "stream",
    "column", "segment", "handle", "frame",
    # 추가 30
    "batch", "queue", "channel", "socket", "worker", "task", "job", "shard",
    "replica", "cluster", "region", "bucket", "object", "blob", "chunk", "offset",
    "cache", "ticket", "event", "signal", "snapshot", "manifest", "registry",
    "endpoint", "adapter", "provider", "resource", "policy", "schema", "context",
)

_POOL_SIZE = 504   # 42블록 × 12 = 504 (서로 다른 이름 ≥500, 블록으로 정확히 덮임)


def _build_name_pool(size: int = _POOL_SIZE) -> tuple[TaskSpec, ...]:
    """동사×명사 조합으로 서로 다른 두 단어 이름 짝 `size`개를 결정적으로 생성한다.

    구성:
      1) **원래 80개** — 앞 20동사 × 20명사, shift 0~3 (원래 공식·순서 그대로). 재현·연속성.
      2) 나머지 — 50×50 공간에서 shift를 늘리며 (원래 80과) 중복 없이 `size`까지 채운다.
    본문·매개변수는 전 항목 공통(값 반환)이라 이름 표기만 변수로 남는다.
    """
    v, n = _POOL_VERBS, _POOL_NOUNS
    seen: set[tuple[str, str]] = set()
    pool: list[TaskSpec] = []

    def add(vb: str, nn: str) -> None:
        if (vb, nn) in seen:
            return
        seen.add((vb, nn))
        pool.append(TaskSpec(
            words=(vb, nn),
            description=f"{vb}s the {nn} and returns it",
            params=("value",),
            body=("return value",),
        ))

    # 1) 원래 80개 (앞 20×20, shift 0~3) — 정체성·순서 보존
    for shift in range(4):
        for i in range(20):
            add(v[i], n[(i + shift) % 20])

    # 2) 50×50 공간에서 size까지 확장(중복 제외)
    shift = 0
    while len(pool) < size:
        shift += 1
        for i in range(len(v)):
            if len(pool) >= size:
                break
            add(v[i], n[(i + shift) % len(n)])

    return tuple(pool[:size])


# stepB 선행 6/6 균형 배치와 관측 대상 이름을 여기서 뽑는다(Composition.POOL).
# 앞 80개는 원래 풀과 동일(파일럿 이름 보존), 전체 504개(블록 커버리지용).
NAME_PAIR_POOL: tuple[TaskSpec, ...] = _build_name_pool()


# 생성용 과제 3개 — 서로 다르고 선행과도 다르다. 모델이 이름을 직접 선택한다.
# 과제 1은 원래 clamp였으나, 모델이 한 단어 `clamp`로 축약해 표기 판정 불가("other")가
# 됐다(stepA-1·stepA-2 결과). 그 권고대로 **두 단어가 강제되는 과제**로 교체한다.
GENERATION_TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        words=("remove", "duplicates"),
        description="removes duplicate items from a list, preserving order",
        params=("items",),
        body=("return list(dict.fromkeys(items))",),
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
