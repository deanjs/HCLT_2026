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
    POOL = "pool"          # 이름 짝 풀(~80)에서 seed로 뽑은 서로 다른 이름 (stepB 균형 관측)


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
    ATTENTION_AMPLIFY = "attn_amplify"  # 지침 스팬 어텐션 증폭 (step6 Spotlight 재구현)
    VALUE_ADD = "value_add"             # 잔차에 조향 방향 더하기 (step6 값 조향, CAA식)


# 층 지정: 정수 층 리스트 / "sweep"(층을 하나씩 순회) / "all"(전 층에 동시 적용)
#   Spotlight는 전 층·전 헤드에 한꺼번에 거는 기법이라 "all"을 쓴다.
LayerSpec = Union[Sequence[int], str]


# ── 조건 축 데이터클래스 ───────────────────────────────────────────────────

@dataclass(frozen=True)
class PrecedingCode:
    """선행 코드 축.

    n_compliant : 규약(camelCase)을 지킨 함수 개수. 계획서는 0~4를 배정한다.
    n_functions : 선행 코드 함수 총 개수(기본 12).
    pool_block  : POOL 전용. 이름 풀을 n_functions개씩 나눈 블록 인덱스. 블록 b는
                  풀의 [b*n_functions, (b+1)*n_functions) 이름을 덮는다(seed는 표기·위치 변주).
                  → 여러 블록으로 풀 전체(504개)를 중복 없이 커버. POOL이 아니면 무시(0).
    """
    n_compliant: int
    n_functions: int = 12
    composition: Composition = Composition.CLONE
    source: Source = Source.SYNTHETIC
    repo_lang: Optional[str] = None  # source=REPO일 때 언어(python/javascript)
    repo_file: Optional[str] = None  # source=REPO일 때 data/repo_files/ 하위 상대경로
    pool_block: int = 0              # POOL 전용 이름 블록 인덱스(§3 500 커버리지)
    lang: Optional[str] = None       # 합성 코드 렌더/파싱 언어(python/javascript). step1 언어 다양성용.

    def __post_init__(self) -> None:
        if self.n_functions <= 0:
            raise ValueError("n_functions는 양수여야 한다")
        if self.pool_block < 0:
            raise ValueError("pool_block은 0 이상이어야 한다")
        if not (0 <= self.n_compliant <= self.n_functions):
            raise ValueError(
                f"n_compliant는 0..{self.n_functions} 범위여야 한다 (받음: {self.n_compliant})"
            )
        if self.lang is not None and self.lang not in ("python", "javascript", "js"):
            raise ValueError("lang은 python/javascript만 허용한다")
        if self.source is Source.REPO:
            # 실코드는 관습이 고정되어 n_compliant/composition을 쓰지 않는다.
            if not self.repo_lang:
                raise ValueError("source=REPO일 때 repo_lang을 지정해야 한다")
            if not self.repo_file:
                raise ValueError("source=REPO일 때 repo_file을 지정해야 한다")
        else:  # SYNTHETIC
            if self.repo_lang is not None or self.repo_file is not None:
                raise ValueError("source=SYNTHETIC에는 repo_lang·repo_file을 두지 않는다")


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
                target="instruction"이면 공여는 "반대 지침"(runner가 자동 구성).
    amplify   : ATTENTION_AMPLIFY에서 곱할 배율(>1 증폭, <1 반감).
    target    : 치환 대상 토큰의 종류. "code"=선행 코드 이름(stepC/step1),
                "instruction"=지침의 표기 지시어 단어(step4, RQ3 인과). 같은 KV 치환
                로직을 대상 토큰만 바꿔 재사용한다(CLAUDE.md §3).
    """
    kind: InterventionKind = InterventionKind.NONE
    layers: Optional[LayerSpec] = None
    donor: Optional[str] = None
    amplify: Optional[float] = None
    # 한 번에 **여러 방식**을 재고 싶을 때 쓴다(예: 한 층에서 key/value/key_value 전부).
    # 지정하면 kind 대신 이 목록이 실행을 규정한다 — 전에는 스윕이 세 방식을 하드코딩해
    # 조건 객체가 실행을 규정하지 못했다(CLAUDE.md §7 위반).
    kinds: Optional[tuple[str, ...]] = None
    # ── step6 처방 전용 ───────────────────────────────────────────────
    strength: Optional[float] = None      # VALUE_ADD: 잔차에 더할 방향의 배율(세기)
    steer_source: Optional[str] = None    # VALUE_ADD: 방향 출처. "code_contrast"(step3 camel−snake)
    steer_layer: Optional[int] = None     # VALUE_ADD: **방향을 뽑을 층**. None이면 주입 층과 같다.
                                          #   다르게 주면 "방향이 층 특이적인가"를 묻는 대조가 된다
                                          #   (맞는 층에서 뽑은 방향을 엉뚱한 층에 주입).
    span: Optional[str] = None            # ATTENTION_AMPLIFY: 밀어 올릴 구간
                                          #   "rule_word"  = 규칙문 지시어만(step5와 스팬 일치)
                                          #   "instruction"= 지침 문장 전체(원 논문 정의)
    target: str = "code"          # 개입 대상: "code"(선행 이름, stepC/1) | "instruction"(지침 지시어, step4)

    def __post_init__(self) -> None:
        if self.target not in ("code", "instruction"):
            raise ValueError('intervention target은 "code" 또는 "instruction"만 허용한다')
        if self.kind is InterventionKind.NONE:
            if (self.layers or self.donor or self.amplify is not None
                    or self.target != "code" or self.kinds
                    or self.strength is not None or self.steer_source or self.span):
                raise ValueError("kind=NONE에는 개입 파라미터를 두지 않는다")
            return
        if self.kinds is not None:
            allowed = {"key", "value", "key_value"}
            bad = [k for k in self.kinds if k not in allowed]
            if bad:
                raise ValueError(f"잴 수 없는 방식: {bad} (key/value/key_value)")
            if len(set(self.kinds)) != len(self.kinds):
                raise ValueError("kinds에 같은 방식을 두 번 넣지 않는다")
        if self.layers is None:
            raise ValueError(f"kind={self.kind.value}에는 layers가 필요하다")
        if isinstance(self.layers, str) and self.layers not in ("sweep", "all"):
            raise ValueError(
                'layers 문자열은 "sweep"(층을 하나씩 순회) 또는 '
                '"all"(전 층에 동시 적용, Spotlight)만 허용한다'
            )
        if self.kind is InterventionKind.ATTENTION_AMPLIFY:
            if self.amplify is None:
                raise ValueError("ATTENTION_AMPLIFY에는 amplify(목표 비중 ψ_target)가 필요하다")
            if not 0.0 < self.amplify < 1.0:
                raise ValueError(f"ψ_target은 0과 1 사이여야 한다 (받음: {self.amplify})")
            if self.span not in ("rule_word", "instruction"):
                raise ValueError(
                    'ATTENTION_AMPLIFY에는 span이 필요하다 ("rule_word" 또는 "instruction"). '
                    "어디를 밀어 올릴지가 결과를 좌우하므로 조건에 명시한다."
                )
        elif self.amplify is not None:
            raise ValueError("치환/조향 계열 개입에는 amplify를 두지 않는다")

        if self.kind is InterventionKind.VALUE_ADD:
            if self.strength is None:
                raise ValueError("VALUE_ADD에는 strength(조향 세기)가 필요하다")
            if self.steer_source is None:
                raise ValueError("VALUE_ADD에는 steer_source(방향 출처)가 필요하다")
        elif (self.strength is not None or self.steer_source is not None
              or self.steer_layer is not None):
            raise ValueError("VALUE_ADD 외에는 strength/steer_source/steer_layer를 두지 않는다")
        if self.kind is not InterventionKind.ATTENTION_AMPLIFY and self.span is not None:
            raise ValueError("span은 ATTENTION_AMPLIFY 전용이다")

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
    # 이름/지시어 토큰 정렬 단위(개입 치환·v 코사인 공통):
    #   "all"  = 전체 토큰 1:1(step1, camel/snake 2:2일 때). 개수 다르면 스킵.
    #   "last" = 마지막 토큰만 1:1(step2 옵션 B; 밑줄을 다르게 쪼개 all이 전부
    #            스킵될 때. 항상 1:1이라 정렬 실패가 없지만 신호가 약할 수 있다)
    #   "mean" = mean-pool(step4 모델다양성). 공여 토큰들을 평균 내 위반 이름의
    #            **모든 토큰 자리**에 넣는다. 개수 불일치를 허용하면서(스킵 없음)
    #            단어 전체를 덮어 last보다 신호가 강하다. v 코사인엔 쓰지 않는다.
    token_unit: str = "all"

    def __post_init__(self) -> None:
        if self.token_unit not in ("all", "last", "mean"):
            raise ValueError('token_unit은 "all"·"last"·"mean"만 허용한다')

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
                target=d["intervention"].get("target", "code"),
                kinds=(tuple(d["intervention"]["kinds"])
                       if d["intervention"].get("kinds") else None),
                strength=d["intervention"].get("strength"),
                steer_source=d["intervention"].get("steer_source"),
                steer_layer=d["intervention"].get("steer_layer"),
                span=d["intervention"].get("span"),
            ),
            seed=d["seed"],
            task_ids=tuple(d.get("task_ids", ())),
            tag=d.get("tag"),
            token_unit=d.get("token_unit", "all"),
        )

    # ── 파일명 슬러그 ─────────────────────────────────────────────────────

    def slug(self) -> str:
        """결과 파일명에 조건이 드러나도록 하는 결정적 슬러그(§6).

        예: qwen2p5-coder-3b__pre-c2of12-clone-syn__ins-neg-camel-weak__int-value-L25__s0
        """
        m = _slugify(self.model.name.split("/")[-1])
        p = self.preceding
        if p.source is Source.REPO:
            stem = p.repo_file.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            pre = f"pre-repo-{_slugify(p.repo_lang)}-{_slugify(stem)}"
        else:
            pre = f"pre-c{p.n_compliant}of{p.n_functions}-{p.composition.value}-syn"
            if p.composition is Composition.POOL:
                pre += f"-b{p.pool_block}"   # 블록별 결과 파일 분리(§6)
            if p.lang and p.lang != "python":
                pre += f"-{_slugify(p.lang)}"   # 언어(js) 결과 파일 분리 (python은 생략)
        ins = (f"ins-{self.instruction.form.value[:3]}-"
               f"{self.instruction.target_notation.value}-{self.instruction.strength.value[:1]}")
        iv = self.intervention
        if iv.kind is InterventionKind.NONE:
            intr = "int-none"
        else:
            head = "+".join(iv.kinds) if iv.kinds else iv.kind.value
            intr = f"int-{head}-{_layer_tag(iv.layers)}"
            if iv.target == "instruction":
                intr += "-instr"          # 지침 지시어 타깃(step4) — 코드 타깃과 파일 구분
            if iv.donor:
                intr += f"-{_slugify(iv.donor)}"
            if iv.amplify is not None:
                # 숫자만 치환한다 — 접두사의 하이픈까지 바꾸면 파일명이 깨진다
                intr += "-psi" + f"{iv.amplify:g}".replace(".", "p")
            if iv.span:
                intr += f"-{_slugify(iv.span)}"
            if iv.strength is not None:
                intr += "-str" + f"{iv.strength:g}".replace(".", "p").replace("-", "m")
            if iv.steer_source:
                intr += f"-{_slugify(iv.steer_source)}"
            if iv.steer_layer is not None:
                intr += f"-from{iv.steer_layer}"   # 방향을 뽑은 층이 주입 층과 다를 때
        parts = [m, pre, ins, intr]
        if self.token_unit != "all":          # all은 생략(기존 슬러그 불변), last만 표기
            parts.append(f"tok-{self.token_unit}")
        parts.append(f"s{self.seed}")
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
