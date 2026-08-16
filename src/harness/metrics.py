"""측정 지표 컨테이너.

계획서가 요구하는 지표(§3, 계획서 §2.4)를 한 곳에 담는다. 값 계산은 각 step에서
채우며, 여기서는 자리와 의미만 고정한다.

    준수율            : 생성 실험 — 새 함수가 목표 표기를 따르는 비율
    준수 선호 점수    : 개입 실험 — 고정 후보 두 이름 중 준수 표기를 선호하는 정도
    어텐션 가중치     : 축 A — 특정 토큰/구간을 얼마나 보는가
    ‖v‖              : 축 B(크기) — 토큰이 실어 나르는 표현의 크기
    ‖av‖             : 축 A×B — Σ a·‖v‖ (상쇄 무시·W_O 이전 = 기여량 상한 대리치)
    v 코사인 유사도   : 축 B(방향) — camel/snake 표현이 다른 정보를 싣는가

층 스윕처럼 층별로 값이 나오는 경우 per_layer에 {층: {지표: 값}}으로 담는다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Metrics:
    # 스칼라 지표 (해당 실험이 산출하는 것만 채운다)
    compliance_rate: Optional[float] = None          # 생성
    compliance_preference: Optional[float] = None    # 개입
    attention_weight: Optional[float] = None         # 축 A
    v_norm: Optional[float] = None                   # 축 B 크기
    av_norm: Optional[float] = None                  # 축 A×B
    v_cosine: Optional[float] = None                 # 축 B 방향

    # 층별 지표 (전 층 스윕 산출물). 키는 층 인덱스(int).
    per_layer: dict[int, dict[str, float]] = field(default_factory=dict)

    # 자유 형식 부가 관측 (예: 회복률, 자기증폭 연쇄 길이 등)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # per_layer 키를 JSON 호환(str)으로
        d["per_layer"] = {str(k): v for k, v in self.per_layer.items()}
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Metrics":
        per_layer = {int(k): v for k, v in d.get("per_layer", {}).items()}
        return Metrics(
            compliance_rate=d.get("compliance_rate"),
            compliance_preference=d.get("compliance_preference"),
            attention_weight=d.get("attention_weight"),
            v_norm=d.get("v_norm"),
            av_norm=d.get("av_norm"),
            v_cosine=d.get("v_cosine"),
            per_layer=per_layer,
            extra=d.get("extra", {}),
        )
