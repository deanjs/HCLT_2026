"""결과 저장 규약 — 불변, 조건이 파일명에 드러남(CLAUDE.md §6).

- 저장 위치: results/ 하위
- 파일명: Condition.slug() 기반 → 조건이 파일명만 봐도 드러난다
- 불변: 이미 있는 파일은 덮어쓰지 않는다(덮어쓰기 시도는 예외)
- 실행 전 예측(prediction)을 함께 보존할 수 있다
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .conditions import Condition
from .metrics import Metrics

RESULTS_DIR = Path("results")


@dataclass
class ResultRecord:
    condition: Condition
    metrics: Metrics
    step: str                              # 어느 step이 만든 결과인지 (예: "step0", "stepA")
    rq: str                                # 대응 RQ (예: "RQ1")
    prediction: Optional[str] = None       # 실행 전 예측 (있으면 결과와 함께 보존)
    meta: dict[str, Any] = field(default_factory=dict)  # git sha, 실행 환경 등
    created_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "rq": self.rq,
            "prediction": self.prediction,
            "created_at": self.created_at,
            "condition": self.condition.to_dict(),
            "metrics": self.metrics.to_dict(),
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ResultRecord":
        return ResultRecord(
            condition=Condition.from_dict(d["condition"]),
            metrics=Metrics.from_dict(d["metrics"]),
            step=d["step"],
            rq=d["rq"],
            prediction=d.get("prediction"),
            meta=d.get("meta", {}),
            created_at=d.get("created_at"),
        )


def result_path(condition: Condition, step: str, results_dir: Path = RESULTS_DIR) -> Path:
    """조건·step으로부터 결정적 결과 경로를 만든다: results/<step>/<slug>.json"""
    return Path(results_dir) / step / f"{condition.slug()}.json"


def save_result(
    record: ResultRecord,
    results_dir: Path = RESULTS_DIR,
    *,
    overwrite: bool = False,
) -> Path:
    """결과를 불변으로 저장한다. 이미 존재하면(overwrite=False) 예외를 던진다."""
    path = result_path(record.condition, record.step, results_dir)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"결과가 이미 존재한다(불변 규약, §6): {path}. "
            f"덮어쓰려면 명시적으로 overwrite=True."
        )
    if record.created_at is None:
        record.created_at = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_result(path: Path) -> ResultRecord:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ResultRecord.from_dict(data)
