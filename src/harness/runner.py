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
from typing import Callable, Optional

from .conditions import Condition, InterventionKind
from .metrics import Metrics
from .model import ModelHandle
import re

from .naming import classify_name, first_function_name
from .prompt import (
    build_instruction_text,
    build_preceding_code,
    first_user_message,
    next_user_message,
)
from .tasks import GENERATION_TASKS

# 생성 함수 타입: 메시지 목록 → 새로 생성된 텍스트.
GenerateFn = Callable[[list], str]

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


def run(
    condition: Condition,
    handle: Optional[ModelHandle] = None,
    *,
    mode: str = "generate",
    generate_fn: Optional[GenerateFn] = None,
    max_new_tokens: int = 256,
) -> RunOutput:
    """조건 하나를 실행한다.

    라우팅은 조건 축과 측정 모드에서 결정한다(스크립트 분기 아님).
      - mode='generate' : 생성 후 표기 측정(step A). 개입이 있으면 개입 경로(미구현).
      - mode='observe'  : 개입 없이 이름 생성 시점 query를 구간별로 관측(step B).

    생성은 handle.chat_generate 또는 주입한 generate_fn을 쓴다
    (후자는 모델 없이 파이프라인을 테스트하기 위한 seam).
    """
    if mode == "observe":
        return _run_observation(condition, handle)
    if mode != "generate":
        raise ValueError(f"알 수 없는 mode: {mode!r} (generate|observe)")
    if condition.intervention.kind is not InterventionKind.NONE:
        # 개입 실험 경로: step C·step 1에서 구현
        raise StageNotImplemented("apply_intervention")
    return _run_generation(condition, handle, generate_fn, max_new_tokens)


def _run_observation(condition: Condition, handle: Optional[ModelHandle]) -> RunOutput:
    """관측 경로(step B) — 이름 생성 시점 query의 구간별 어텐션·‖v‖·‖av‖.

    선행 코드의 camel/snake 이름을 두 그룹으로 나눠, 지침을 생성하는 바로 그 query가
    어느 그룹을 더 보는지(축 A)와 각 그룹이 잔차에 기여하는 양(‖av‖, 축 A×B)을 잰다.
    개입은 없다(관측). 인과는 step C.
    """
    if handle is None:
        raise ValueError("관측에는 handle이 필요하다 (모델 내부 접근)")

    # 선행 코드에서 함수 이름을 뽑아 표기별로 그룹화(POOL 배치면 camel 6 / snake 6).
    preceding = build_preceding_code(condition)
    names = re.findall(r"def (\w+)\(", preceding)
    groups: dict[str, list[str]] = {"code_camel": [], "code_snake": []}
    for nm in names:
        cls = classify_name(nm)
        if cls == "camel":
            groups["code_camel"].append(nm)
        elif cls == "snake":
            groups["code_snake"].append(nm)

    instruction_text = build_instruction_text(condition)
    messages = [
        {"role": "system", "content": instruction_text},
        {"role": "user", "content": first_user_message(condition)},
    ]

    obs = handle.observe_generation_query(
        messages, groups=groups, instruction_text=instruction_text
    )

    # per_layer를 Metrics 스키마에 담는다. 구간·지표를 평평한 키로 편다.
    per_layer: dict[int, dict[str, float]] = {}
    for layer, by_span in obs["per_layer"].items():
        flat: dict[str, float] = {}
        for span_name, vals in by_span.items():
            for metric_name, v in vals.items():
                if metric_name == "n_tokens" or v is None:
                    continue
                flat[f"{span_name}__{metric_name}"] = v
        per_layer[int(layer)] = flat

    metrics = Metrics(
        per_layer=per_layer,
        extra={
            "mode": "observe",
            "target": condition.instruction.target_notation.value,
            "instruction_form": condition.instruction.form.value,
            "group_names": {"code_camel": groups["code_camel"],
                            "code_snake": groups["code_snake"]},
            "span_token_counts": obs["spans"],
            "seq_len": obs["seq_len"],
            "gqa": obs["gqa"],
            # per-token 상세(code_camel/code_snake) — 밑줄/형태 마커 분석용(§6 불변 저장)
            "token_detail": obs["token_detail"],
        },
    )
    return RunOutput(condition=condition, metrics=metrics)


def _run_generation(
    condition: Condition,
    handle: Optional[ModelHandle],
    generate_fn: Optional[GenerateFn],
    max_new_tokens: int,
) -> RunOutput:
    """생성 경로 — 선행 12 + 생성 3을 순차 생성하고 표기를 측정한다(step A)."""
    if generate_fn is None:
        if handle is None:
            raise ValueError("생성에는 handle 또는 generate_fn 중 하나가 필요하다")
        generate_fn = lambda msgs: handle.chat_generate(
            msgs, max_new_tokens=max_new_tokens, seed=condition.seed
        )

    # system(지침) + 순차 3턴. 모델의 이전 답이 히스토리에 쌓여 자기증폭이 누적된다.
    # 코드 언어(합성=python, 실코드=repo_lang) → 이름 추출기 선택
    lang = condition.preceding.repo_lang or "python"

    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_instruction_text(condition)}
    ]
    notations: list[str] = []
    names: list[Optional[str]] = []
    texts: list[str] = []
    for turn in range(len(GENERATION_TASKS)):
        user = first_user_message(condition) if turn == 0 else next_user_message(turn)
        messages.append({"role": "user", "content": user})
        text = generate_fn(messages)
        messages.append({"role": "assistant", "content": text})
        name = first_function_name(text, lang)
        names.append(name)
        notations.append(classify_name(name) if name else "other")
        texts.append(text)

    target = condition.instruction.target_notation.value
    first_compliant = notations[0] == target
    subsequent = notations[1:]
    subsequent_violation = (
        sum(n != target for n in subsequent) / len(subsequent) if subsequent else None
    )
    metrics = Metrics(
        compliance_rate=1.0 if first_compliant else 0.0,  # 조건별 평균은 seed 집계에서
        extra={
            "turn_notations": notations,
            "turn_names": names,   # 모델이 실제 고른 함수명 (표기 판정의 근거)
            "turn_texts": texts,   # 턴별 생성 원문 (눈으로 검증)
            "target": target,
            "first_compliant": first_compliant,
            "first_violated": not first_compliant,
            # 자기증폭: 첫 함수가 위반일 때 뒤 함수도 위반인 비율
            "subsequent_violation_rate": subsequent_violation,
        },
    )
    return RunOutput(condition=condition, metrics=metrics)


def describe_pipeline() -> str:
    """골격이 고정한 단계-담당 step 대응을 사람이 읽을 수 있게 출력."""
    lines = ["파이프라인 단계        | 채우는 step", "-" * 48]
    for key, desc, owner in PIPELINE:
        lines.append(f"{key:22s} | {owner}    — {desc}")
    return "\n".join(lines)
