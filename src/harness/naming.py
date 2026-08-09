"""이름 표기 렌더링과 판정.

- render_camel / render_snake : 단어 목록을 camelCase / snake_case 식별자로.
  표기가 구분되려면 단어가 둘 이상이어야 한다(한 단어면 camel==snake).
- classify_name : 식별자가 camel / snake / other 중 무엇인가.
- first_def_name : 생성 텍스트에서 첫 `def <name>(` 의 이름을 뽑는다.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from .conditions import Notation


def render_camel(words: Sequence[str], idx: Optional[int] = None) -> str:
    first, *rest = words
    name = first.lower() + "".join(w.capitalize() for w in rest)
    return name if idx is None else f"{name}{idx}"


def render_snake(words: Sequence[str], idx: Optional[int] = None) -> str:
    name = "_".join(w.lower() for w in words)
    return name if idx is None else f"{name}_{idx}"


def render_name(words: Sequence[str], notation: Notation, idx: Optional[int] = None) -> str:
    if notation is Notation.CAMEL:
        return render_camel(words, idx)
    return render_snake(words, idx)


# camelCase: 소문자로 시작 + 내부 대문자 경계 하나 이상 (예: scaleValue2, clampNumber)
_CAMEL_RE = re.compile(r"^[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+$")
# snake_case: 밑줄 경계 하나 이상 (예: scale_value_1, clamp_number)
_SNAKE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")


def classify_name(name: str) -> str:
    """식별자를 'camel' / 'snake' / 'other'로 분류한다."""
    if _SNAKE_RE.match(name):
        return "snake"
    if _CAMEL_RE.match(name):
        return "camel"
    return "other"


_DEF_RE = re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def first_def_name(text: str) -> Optional[str]:
    """Python 생성 텍스트에서 첫 `def <name>(`의 이름을 반환한다(없으면 None)."""
    m = _DEF_RE.search(text)
    return m.group(1) if m else None


# JS 함수 정의: function foo( / const foo = (…) => / const foo = async (…) => /
#              const foo = function / let|var 도 동일. 이름만 캡처.
_JS_RES = (
    re.compile(r"\bfunction\s*\*?\s+([A-Za-z_$][\w$]*)\s*\("),
    re.compile(
        r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
        r"(?:async\s+)?(?:function\b|\(|[A-Za-z_$][\w$]*\s*=>)"
    ),
)


def first_js_name(text: str) -> Optional[str]:
    """JavaScript 생성 텍스트에서 첫 함수 정의의 이름을 반환한다(없으면 None)."""
    best: Optional[tuple[int, str]] = None
    for rgx in _JS_RES:
        m = rgx.search(text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), m.group(1))
    return best[1] if best else None


def first_function_name(text: str, lang: str = "python") -> Optional[str]:
    """언어에 맞춰 첫 함수 이름을 뽑는다. camel/snake 판정은 언어 무관(classify_name)."""
    if lang in ("js", "javascript"):
        return first_js_name(text)
    return first_def_name(text)
