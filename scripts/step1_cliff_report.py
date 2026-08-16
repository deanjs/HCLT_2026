"""step1(준수율 절벽) 결과 집계 + 그래프.

results/step1_cliff_results_*/ 의 불변 산출물을 읽어
  - docs/step1/summary.csv   (조건별 준수율)
  - docs/step1/figs/*.png    (그래프)
를 만든다. results/ 는 절대 건드리지 않는다(§6).

실행: python scripts/step1_cliff_report.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "step1"
FIGS = OUT / "figs"

# 결과 폴더 → 그래프에 쓸 모델 표시 이름
MODEL_DIRS = {
    "step1_cliff_results_qwen": "Qwen2.5-Coder-3B",
    "step1_cliff_results_stability": "stable-code-3b",
}

# dataviz 검증 통과 팔레트(light, categorical slot 1·2)
COLORS = {"Qwen2.5-Coder-3B": "#2a78d6", "stable-code-3b": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd6"

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load() -> list[dict]:
    """결과 JSON 전부를 평평한 레코드로 읽는다."""
    rows = []
    for dirname, label in MODEL_DIRS.items():
        for path in sorted((ROOT / "results" / dirname).glob("*.json")):
            r = json.load(open(path, encoding="utf-8"))
            cond, met = r["condition"], r["metrics"]
            pre = cond["preceding"]
            rows.append({
                "model": label,
                "target": cond["instruction"]["target_notation"],
                "n_violations": pre["n_functions"] - pre["n_compliant"],
                "block": pre["pool_block"],
                "generated": met["extra"]["turn_notations"][0],
                "compliant": met["compliance_rate"],
            })
    return rows


def rate(vals: list[float]) -> tuple[float, float]:
    """준수율과 95% 오차폭(정규 근사)."""
    n = len(vals)
    p = sum(vals) / n
    return p, 1.96 * math.sqrt(max(p * (1 - p), 0.0) / n)


def summarize(rows: list[dict]) -> dict:
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["model"], r["target"], r["n_violations"])].append(r["compliant"])
    return {k: (rate(v)[0], rate(v)[1], len(v)) for k, v in buckets.items()}


def write_csv(summary: dict) -> None:
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "instructed_notation", "n_violations",
                    "compliance_rate", "ci95", "n_samples"])
        for (m, t, v), (p, ci, n) in sorted(summary.items()):
            w.writerow([m, t, v, f"{p:.4f}", f"{ci:.4f}", n])


def fig_cliff(summary: dict) -> None:
    """그림 1 — 위반 개수에 따른 준수율 곡선(지침 방향별 2패널)."""
    xs = list(range(13))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    for ax, target, title in zip(
        axes, ["camel", "snake"],
        ["Instruction: camelCase (against Python prior)",
         "Instruction: snake_case (with Python prior)"],
    ):
        for model, color in COLORS.items():
            ys = [summary[(model, target, v)][0] for v in xs]
            es = [summary[(model, target, v)][1] for v in xs]
            ax.errorbar(xs, ys, yerr=es, color=color, lw=2, marker="o", ms=4,
                        capsize=2, elinewidth=1, label=model, zorder=3)
        ax.set_title(title, fontsize=9, color=INK, pad=8)
        ax.set_xlabel("Violating functions in context (of 12)")
        ax.set_xticks(xs)
        ax.set_ylim(-0.05, 1.08)
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Compliance rate")
    # 절벽 위치 표시(camel 패널)
    axes[0].axvspan(4.5, 5.5, color="#0b0b0b", alpha=0.05, zorder=1)
    axes[0].annotate("cliff", xy=(5, 0.55), xytext=(6.6, 0.72), color=INK2,
                     arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Compliance collapses once context violations pass a threshold",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "cliff_curve.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_drop(summary: dict) -> None:
    """그림 2 — camel 지침에서 위반 1개가 추가될 때마다의 준수율 감소폭."""
    xs = list(range(1, 13))
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    width = 0.4
    for i, (model, color) in enumerate(COLORS.items()):
        ys = [summary[(model, "camel", v - 1)][0] - summary[(model, "camel", v)][0]
              for v in xs]
        ax.bar([x + (i - 0.5) * width for x in xs], ys, width * 0.92, color=color,
               label=model, zorder=3)
    ax.set_xlabel("Violating functions in context (of 12)")
    ax.set_ylabel("Drop in compliance vs. one fewer")
    ax.set_xticks(xs)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Where compliance is lost (camelCase instruction)", fontsize=10, pad=8)
    fig.tight_layout()
    fig.savefig(FIGS / "marginal_drop.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    rows = load()
    summary = summarize(rows)
    write_csv(summary)
    fig_cliff(summary)
    fig_drop(summary)

    print(f"읽은 결과 파일 수: {len(rows)}")
    for model in COLORS:
        sub = [r for r in rows if r["model"] == model]
        snake = sum(r["generated"] == "snake" for r in sub) / len(sub)
        print(f"{model}: 묶음당 표본 {summary[(model, 'camel', 0)][2]}개, "
              f"snake로 생성한 비율 {snake:.3f}")
    print(f"저장: {OUT/'summary.csv'}, {FIGS}/*.png")


if __name__ == "__main__":
    main()
