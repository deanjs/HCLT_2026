"""step1(준수율 절벽) 재집계 — 준수 개수별 준수율과, 측정되지 않은 지표 확인.

무엇을 확인하나
---------------
① **준수율 절벽**: 앞선 코드에 규약을 지킨 함수가 몇 개냐(0~12)에 따라 모델이 첫 함수를
   규약대로 쓰는 비율이 어떻게 달라지나. RQ1의 본체다.
② **자기증폭 지표가 실제로 측정됐나**: `subsequent_violation_rate`는 "첫 함수가 위반일 때
   뒤 함수도 위반인 비율"이라고 주석에 적혀 있지만, 구현은 첫 결과와 무관한 무조건부다.
   그리고 원자료(`turn_notations`)에 턴이 몇 개나 들어 있는지 확인해야 재계산이 가능한지 알 수 있다.

쓰는 법
-------
    python scripts/step1_reaggregate.py results/step1_cliff [--figs docs/step1/figs]
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def load(root: str | Path):
    rows = []
    for p in sorted(Path(root).rglob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        ex = r["metrics"]["extra"]
        rows.append({
            "model": r["condition"]["model"]["family"],
            "target": r["condition"]["instruction"]["target_notation"],
            "n_compliant": r["condition"]["preceding"]["n_compliant"],
            "n_functions": r["condition"]["preceding"]["n_functions"],
            "compliant": bool(ex["first_compliant"]),
            "notations": ex["turn_notations"],
            "sub_rate": ex.get("subsequent_violation_rate"),
        })
    return rows


def ci95(xs):
    if len(xs) < 2:
        return (xs[0] if xs else float("nan")), float("nan")
    return st.mean(xs), 1.96 * st.stdev(xs) / math.sqrt(len(xs))


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    rows = load(sys.argv[1])
    fig_dir = None
    if "--figs" in sys.argv:
        fig_dir = Path(sys.argv[sys.argv.index("--figs") + 1])
        fig_dir.mkdir(parents=True, exist_ok=True)
    models = sorted({r["model"] for r in rows})
    levels = sorted({r["n_compliant"] for r in rows})

    print("=" * 88)
    print("① 준수율 절벽 — 앞선 코드의 준수 개수(0~12)별 첫 함수 준수율")
    print("=" * 88)
    for target in ("camel", "snake"):
        print(f"\n[지침 목표 = {target}]")
        print(f"{'준수개수':<9}" + "".join(f"{m:>22}" for m in models))
        for n in levels:
            line = f"{n:<9}"
            for m in models:
                xs = [1.0 if r["compliant"] else 0.0 for r in rows
                      if r["model"] == m and r["target"] == target and r["n_compliant"] == n]
                mu, ci = ci95(xs)
                line += f"{mu:>15.3f}±{ci:<6.3f}"
            print(line)

    print("\n" + "=" * 88)
    print("② 원자료 점검 — 자기증폭 지표를 재계산할 수 있나")
    print("=" * 88)
    turn_counts = defaultdict(int)
    none_sub = 0
    for r in rows:
        turn_counts[len(r["notations"])] += 1
        if r["sub_rate"] is None:
            none_sub += 1
    print(f"조건 수                     : {len(rows)}")
    print(f"생성 턴 수 분포             : {dict(turn_counts)}")
    print(f"subsequent_violation_rate가 비어 있는 조건: {none_sub} / {len(rows)}")
    if set(turn_counts) == {1}:
        print("\n  → 모든 조건이 **1턴만** 생성했다. 뒤따르는 함수가 없으므로")
        print("     '첫 함수가 위반일 때 뒤 함수도 위반인 비율'(자기증폭)은")
        print("     **재계산이 아니라 재실행이 필요하다.** 현재 결과에는 원자료 자체가 없다.")
    else:
        print("\n  → 2턴 이상인 조건이 있으므로 조건부 위반율을 재계산한다:")
        for m in models:
            cond = [r for r in rows if r["model"] == m and len(r["notations"]) > 1]
            if not cond:
                continue
            viol_first = [r for r in cond if not r["compliant"]]
            def rate(rs):
                out = []
                for r in rs:
                    sub = r["notations"][1:]
                    out.append(sum(x != r["target"] for x in sub) / len(sub))
                return ci95(out)
            a, ac = rate(cond)
            b, bc = rate(viol_first)
            print(f"    {m:<10} 무조건부 {a:.3f}±{ac:.3f}   "
                  f"첫 함수가 위반일 때만 {b:.3f}±{bc:.3f}  (n={len(viol_first)})")

    if fig_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, target in zip(axes, ("camel", "snake")):
            for m, color in zip(models, ("tab:blue", "tab:orange")):
                pts = []
                for n in levels:
                    xs = [1.0 if r["compliant"] else 0.0 for r in rows
                          if r["model"] == m and r["target"] == target and r["n_compliant"] == n]
                    pts.append(ci95(xs))
                mu = [a for a, _ in pts]
                ax.plot(levels, mu, marker="o", color=color, label=m, linewidth=1.6)
                ax.fill_between(levels, [a - b for a, b in pts], [a + b for a, b in pts],
                                color=color, alpha=0.2, linewidth=0)
            ax.set_xlabel("Compliant functions in preceding code (out of 12)")
            ax.set_title(f"instruction target = {target}")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Compliance rate of the generated name")
        axes[0].legend(fontsize=9)
        fig.suptitle("step1 compliance cliff (42 name blocks, mean ± 95% CI)", fontsize=11)
        fig.tight_layout()
        fig.savefig(fig_dir / "cliff_reaggregated.png", dpi=150)
        plt.close(fig)
        print(f"\n그림 → {fig_dir}/cliff_reaggregated.png")


if __name__ == "__main__":
    main()
