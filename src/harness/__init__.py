"""표기 규약 준수 실패 Probing 연구 — 실험 하네스.

RQ1·RQ2·RQ3은 별도 스크립트가 아니라 **하나의 파이프라인**에 조건 축만
다르게 넣어 돌린다(CLAUDE.md §3). 이 패키지는 그 조건 축을 고정하고,
단일 진입점 `run()`으로 실험을 실행하며, 지표를 `results/`에 불변으로 저장한다.

- conditions : 조건 스키마 (5개 축) — 단일 진실 공급원
- metrics    : 측정 지표 컨테이너
- results    : 결과 저장 규약 (불변, 조건이 파일명에 드러남)
- model      : 모델 로딩 래퍼 (GQA group 확인 포함)
- runner     : 단일 진입점 run(condition)
"""

from .conditions import (
    Composition,
    Condition,
    Instruction,
    InstructionForm,
    Intervention,
    InterventionKind,
    ModelSpec,
    Notation,
    PrecedingCode,
    Source,
    Strength,
)
from .metrics import Metrics
from .naming import classify_name, first_def_name, render_camel, render_snake
from .prompt import (
    build_instruction_text,
    build_preceding_code,
    first_user_message,
    next_user_message,
)
from .results import ResultRecord, result_path, save_result
from .runner import PIPELINE, run
from .tasks import CLONE_TASK, DISTINCT_TASKS, GENERATION_TASKS, TaskSpec

__all__ = [
    "Composition",
    "Condition",
    "Instruction",
    "InstructionForm",
    "Intervention",
    "InterventionKind",
    "ModelSpec",
    "Notation",
    "PrecedingCode",
    "Source",
    "Strength",
    "Metrics",
    "classify_name",
    "first_def_name",
    "render_camel",
    "render_snake",
    "build_instruction_text",
    "build_preceding_code",
    "first_user_message",
    "next_user_message",
    "CLONE_TASK",
    "DISTINCT_TASKS",
    "GENERATION_TASKS",
    "TaskSpec",
    "ResultRecord",
    "result_path",
    "save_result",
    "PIPELINE",
    "run",
]
