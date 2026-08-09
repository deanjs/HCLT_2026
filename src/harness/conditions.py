"""조건 스키마 — 실험의 단일 진실 공급원.

RQ1·RQ2·RQ3은 스크립트가 아니라 이 스키마의 값 조합으로 구분된다(CLAUDE.md §3).
새 실험은 새 파일이 아니라 여기 Enum/필드에 값을 추가하는 방식으로 확장한다.

조건 축(§3):
    선행 코드 : 준수 개수 / 구성(클론·개별과제) / 출처(합성·실제저장소)
    지침      : 형식(긍정·부정) / 목표 표기(camel·snake)
    개입      : 없음 / Key / Value / Key+Value / 어텐션 증폭
    층        : 전 층 스윕
    모델      : 패밀리별
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence, Union


# ── 열거형 (조건 축의 이산 값) ─────────────────────────────────────────────

class Composition(str, Enum):
    """선행 코드 구성 — 12개 함수가 같은 일을 하는가(클론), 다른 일을 하는가."""
    CLONE = "clone"        # 동일 작업 12개 (기본, A1 재현용)
    DISTINCT = "distinct"  # 서로 다른 작업 12개 (step 3 ablation)


class Source(str, Enum):
    """선행 코드 출처 — 합성인가 실제 저장소 파일인가."""
    SYNTHETIC = "synthetic"  # 합성 코드 (기본)
    REPO = "repo"            # 실제 저장소 파일 (step 4)


class InstructionForm(str, Enum):
    """지침 형식 — 긍정형인가 부정형인가(RQ3의 핵심 축)."""
    POSITIVE = "positive"  # "camelCase로 작성하라"
    NEGATIVE = "negative"  # "snake_case로 작성하지 마라"


class Notation(str, Enum):
    """표기 규약 — 준수/위반 및 목표 표기를 이 두 값으로 닫는다."""
    CAMEL = "camel"
    SNAKE = "snake"


class Strength(str, Enum):
    """지침 어조 — 강한 어조는 준수율이 포화되어 조건 차이가 사라진다(계획서 §3 통제)."""
    WEAK = "weak"
    STRONG = "strong"


class InterventionKind(str, Enum):
    """개입 종류 — 무엇을 어떻게 조작하는가."""
    NONE = "none"                     # 개입 없음 (생성 실험, baseline)
    KEY = "key"                       # 선행 코드 Key만 치환 (축 A 경로)
    VALUE = "value"                   # 선행 코드 Value만 치환 (축 B 경로)
    KEY_VALUE = "key_value"           # Key+Value 동시 치환 (현재 확보 결과)
    ATTENTION_AMPLIFY = "attn_amplify"  # 지침 구간 어텐션 증폭/반감


# 층 지정: 정수 층 리스트, 또는 전 층 스윕을 뜻하는 "sweep"
LayerSpec = Union[Sequence[int], str]


# ── 조건 축 데이터클래스 ───────────────────────────────────────────────────

@dataclass(frozen=True)
class PrecedingCode:
    """선행 코드 축.

    n_compliant : 규약(camelCase)을 지킨 함수 개수. 계획서는 0~4를 배정한다.
    n_functions : 선행 코드 함수 총 개수(기본 12).
    """
    n_compliant: int
    n_functions: int = 12
    composition: Composition = Composition.CLONE
    source: Source = Source.SYNTHETIC
    repo_lang: Optional[str] = None  # source=REPO일 때 언어(python/js/java 등)

    def __post_init__(self) -> None:
        if self.n_functions <= 0:
            raise ValueError("n_functions는 양수여야 한다")
        if not (0 <= self.n_compliant <= self.n_functions):
            raise ValueError(
                f"n_compliant는 0..{self.n_functions} 범위여야 한다 (받음: {self.n_compliant})"
            )
        if self.source is Source.REPO and not self.repo_lang:
            raise ValueError("source=REPO일 때 repo_lang을 지정해야 한다")
        if self.source is Source.SYNTHETIC and self.repo_lang is not None:
            raise ValueError("source=SYNTHETIC에는 repo_lang을 두지 않는다")


@dataclass(frozen=True)
class Instruction:
    """지침 축.

    candidates : 생성 실험에서 표기 후보를 두 개로 못 박기 위한 닫힌 선택지.
                 부정형이 긍정형과 논리적 등가가 되려면 필수(계획서 RQ3 주의사항).
    """
    form: InstructionForm
    target_notation: Notation
    strength: Strength = Strength.WEAK
    candidates: tuple[Notation, Notation] = (Notation.CAMEL, Notation.SNAKE)

    def __post_init__(self) -> None:
        if self.target_notation not in self.candidates:
            raise ValueError("target_notation은 candidates 안에 있어야 한다")
        if len(set(self.candidates)) != 2:
            raise ValueError("candidates는 서로 다른 두 표기여야 한다")

    @property
    def violation_notation(self) -> Notation:
        """목표 표기가 아닌 나머지 후보(= 위반 표기)."""
        return next(c for c in self.candidates if c != self.target_notation)

    @property
    def token_notation(self) -> Notation:
        """지침 문장에 실제로 등장하는 표기 토큰.

        긍정형은 목표 표기를, 부정형은 위반 표기를 문장 안에 담는다(RQ3 착안점).
        """
        if self.form is InstructionForm.POSITIVE:
            return self.target_notation
        return self.violation_notation


@dataclass(frozen=True)
class Intervention:
    """개입 축.

    layers    : 개입할 층. 정수 리스트 또는 "sweep"(전 층). NONE이면 무시.
    donor     : 치환에 쓸 공여 표현의 출처. 무관 코드 통제 조건을 여기서 표현한다
                (예: "unrelated_camel", "unrelated_snake", "compliant").
    amplify   : ATTENTION_AMPLIFY에서 곱할 배율(>1 증폭, <1 반감).
    """
    kind: InterventionKind = InterventionKind.NONE
    layers: Optional[LayerSpec] = None
    donor: Optional[str] = None
    amplify: Optional[float] = None

    def __post_init__(self) -> None:
        if self.kind is InterventionKind.NONE:
            if self.layers or self.donor or self.amplify is not None:
                raise ValueError("kind=NONE에는 개입 파라미터를 두지 않는다")
            return
        if self.layers is None:
            raise ValueError(f"kind={self.kind.value}에는 layers가 필요하다")
        if isinstance(self.layers, str) and self.layers != "sweep":
            raise ValueError('layers 문자열은 "sweep"만 허용한다')
        if self.kind is InterventionKind.ATTENTION_AMPLIFY:
            if self.amplify is None:
                raise ValueError("ATTENTION_AMPLIFY에는 amplify 배율이 필요하다")
        elif self.amplify is not None:
            raise ValueError("치환 계열 개입에는 amplify를 두지 않는다")

    @property
    def is_sweep(self) -> bool:
        return isinstance(self.layers, str) and self.layers == "sweep"


@dataclass(frozen=True)
class ModelSpec:
    """모델 축."""
    name: str            # HF 허브 ID (예: "Qwen/Qwen2.5-Coder-3B-Instruct")
    family: str          # 패밀리 라벨 (예: "qwen", "deepseek", "starcoder")
    dtype: str = "float16"
    quantization: Optional[str] = None  # None / "8bit" / "4bit"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("model name은 비어 있을 수 없다")


# ── 상위 조건 객체 ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Condition:
    """실험 하나를 완전히 규정하는 조건 조합.

    단일 진입점 run(condition)이 받는 유일한 입력. 이 객체 하나로 실험이 재현된다.
    """
    model: ModelSpec
    preceding: PrecedingCode
    instruction: Instruction
    intervention: Intervention = field(default_factory=Intervention)
    seed: int = 0
    task_ids: tuple[str, ...] = ()  # 생성시킬 함수 과제 식별자(순서 = 생성 순서)
    tag: Optional[str] = None       # 자유 라벨 (예: "holdout", "pilot")

    # ── 직렬화 ────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Condition":
        return Condition(
            model=ModelSpec(**d["model"]),
            preceding=PrecedingCode(
                **{**d["preceding"],
                   "composition": Composition(d["preceding"]["composition"]),
                   "source": Source(d["preceding"]["source"])}
            ),
            instruction=Instruction(
                form=InstructionForm(d["instruction"]["form"]),
                target_notation=Notation(d["instruction"]["target_notation"]),
                strength=Strength(d["instruction"]["strength"]),
                candidates=tuple(Notation(c) for c in d["instruction"]["candidates"]),
            ),
            intervention=Intervention(
                kind=InterventionKind(d["intervention"]["kind"]),
                layers=(list(d["intervention"]["layers"])
                        if isinstance(d["intervention"]["layers"], list)
                        else d["intervention"]["layers"]),
                donor=d["intervention"]["donor"],
                amplify=d["intervention"]["amplify"],
            ),
            seed=d["seed"],
            task_ids=tuple(d.get("task_ids", ())),
            tag=d.get("tag"),
        )

    # ── 파일명 슬러그 ─────────────────────────────────────────────────────

    def slug(self) -> str:
        """결과 파일명에 조건이 드러나도록 하는 결정적 슬러그(§6).

        예: qwen2p5-coder-3b__pre-c2of12-clone-syn__ins-neg-camel-weak__int-value-L25__s0
        """
        m = _slugify(self.model.name.split("/")[-1])
        p = self.preceding
        pre = f"pre-c{p.n_compliant}of{p.n_functions}-{p.composition.value}-{p.source.value[:3]}"
        if p.repo_lang:
            pre += f"-{_slugify(p.repo_lang)}"
        ins = (f"ins-{self.instruction.form.value[:3]}-"
               f"{self.instruction.target_notation.value}-{self.instruction.strength.value[:1]}")
        iv = self.intervention
        if iv.kind is InterventionKind.NONE:
            intr = "int-none"
        else:
            intr = f"int-{iv.kind.value}-{_layer_tag(iv.layers)}"
            if iv.donor:
                intr += f"-{_slugify(iv.donor)}"
            if iv.amplify is not None:
                intr += f"-x{iv.amplify:g}".replace(".", "p")
        parts = [m, pre, ins, intr, f"s{self.seed}"]
        if self.tag:
            parts.append(_slugify(self.tag))
        return "__".join(parts)


# ── 슬러그 헬퍼 ────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _layer_tag(layers: LayerSpec) -> str:
    if isinstance(layers, str):
        return layers  # "sweep"
    layers = list(layers)
    if len(layers) == 1:
        return f"L{layers[0]}"
    return f"L{layers[0]}-{layers[-1]}" if layers else "L?"
