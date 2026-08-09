"""단일 진입점 — run(condition).

모든 실험이 이 함수 하나를 통과한다. RQ별 분기는 여기 없고, 조건 축의 값에 따라
각 단계가 다르게 동작할 뿐이다(CLAUDE.md §3).

파이프라인 단계는 아래 PIPELINE에 고정한다. step 0(하네스 골격)에서는 각 단계의
**인터페이스만** 확정하고, 실제 로직은 대응하는 step에서 채운다. 아직 채워지지
않은 단계는 어느 step이 구현하는지를 담아 NotImplementedError를 던진다 —
"조건 스키마 → 단계 → 채움 주체"의 대응이 코드에서 바로 드러나도록.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .conditions import Condition, InterventionKind
from .metrics import Metrics
from .model import ModelHandle

# 파이프라인 단계와 그것을 채우는 step. 골격에서 이 대응이 계약이다.
PIPELINE: tuple[tuple[str, str, str], ...] = (
    # (단계 키, 설명, 채우는 step)
    ("build_preceding", "선행 코드 구성 (준수 개수·클론/개별·합성/실제)", "step A / step 3·4"),
    ("build_instruction", "지침 문장 구성 (긍정/부정·목표 표기·닫힌 후보)", "step A / step 5"),
    ("build_prompt", "선행 코드 + 지침 + 생성 과제를 프롬프트로 조립", "step A"),
    ("measure_generation", "생성 후 첫 함수 표기 측정 → 준수율", "step A / step 5"),
    ("measure_attention", "구간별 어텐션·‖v‖·‖av‖·v 코사인 관측", "step B / step 6"),
    ("apply_intervention", "Key/Value 치환 또는 어텐션 증폭 훅 적용", "step C / step 1"),
    ("measure_preference", "고정 후보 두 이름의 준수 선호 점수", "step C / step 1"),
)


@dataclass
class RunOutput:
    condition: Condition
    metrics: Metrics


class StageNotImplemented(NotImplementedError):
    """아직 채워지지 않은 파이프라인 단계. 어느 step이 구현하는지 메시지에 담는다."""

    def __init__(self, stage_key: str) -> None:
        owner = next((o for k, _d, o in PIPELINE if k == stage_key), "?")
        super().__init__(
            f"파이프라인 단계 '{stage_key}'는 아직 골격(step 0)이다. "
            f"이 단계는 {owner}에서 구현한다."
        )
        self.stage_key = stage_key


def run(condition: Condition, handle: Optional[ModelHandle] = None) -> RunOutput:
    """조건 하나를 실행한다.

    step 0에서는 조건을 검증(생성자에서 이미 수행)하고, 어떤 단계가 필요한지
    조건 축으로부터 결정하는 라우팅까지만 확정한다. 필요한 측정 단계는 대응 step이
    채우기 전까지 StageNotImplemented를 던진다.
    """
    needs_intervention = condition.intervention.kind is not InterventionKind.NONE

    # 라우팅: 조건 축 → 필요한 측정. (스크립트 분기가 아니라 조건 기반 분기)
    if needs_intervention:
        # 개입 실험 경로: 치환/증폭 후 준수 선호 점수
        raise StageNotImplemented("apply_intervention")
    else:
        # 생성 실험 경로: 프롬프트 조립 후 준수율
        raise StageNotImplemented("measure_generation")


def describe_pipeline() -> str:
    """골격이 고정한 단계-담당 step 대응을 사람이 읽을 수 있게 출력."""
    lines = ["파이프라인 단계        | 채우는 step", "-" * 48]
    for key, desc, owner in PIPELINE:
        lines.append(f"{key:22s} | {owner}    — {desc}")
    return "\n".join(lines)
