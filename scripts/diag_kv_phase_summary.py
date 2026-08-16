"""A 진단 집계 — 평균 덮어쓰기가 어텐션 쪽(Key) 값을 불리하게 망가뜨렸나.

재는 값
------
    줄어듦 = ‖조각들의 평균‖ ÷ 조각 노름들의 평균     (1이면 손해 없음)

Key에는 위치 회전(RoPE)이 박혀 있고 Value에는 없다. 회전이 엇갈리면 평균이 서로 지워져
Key만 더 작아진다 — 그러면 "어텐션을 바꿔도 안 돌아온다"가 방법의 산물일 수 있다.
**Value가 기준선**이다. 두 값이 비슷하면 회전 탓의 추가 손해가 없다는 뜻.

쓰는 법
------
    python scripts/diag_kv_phase_summary.py results/diag_kv-phase_results [--figs docs/diag/figures]
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
            "target": ex["intervention_target"],
            "offset": ex["position_offset_abs_mean"],
            "donor_pieces": ex["donor_piece_counts"],
            "target_pieces": ex["target_piece_counts"],
            "per_layer": {int(L): v for L, v in r["metrics"]["per_layer"].items()},
        })
    return rows


def ci95(xs):
    if len(xs) < 2:
        return (xs[0] if xs else float("nan")), float("nan")
    return st.mean(xs), 1.96 * st.stdev(xs) / math.sqrt(len(xs))


def curves(rows, model, target, key):
    out = defaultdict(list)
    for r in rows:
        if r["model"] == model and r["target"] == target:
            for L, v in r["per_layer"].items():
                out[L].append(v[key])
    return out


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
    targets = sorted({r["target"] for r in rows})

    print("=" * 96)
    print("① 덮을 자리와 공여 자리의 위치가 얼마나 어긋나 있나 (칸 수)")
    print("=" * 96)
    print(f"{'모델':<11}" + "".join(f"{t:>16}" for t in targets) + "   조각 수(공여/대상)")
    for m in models:
        line = f"{m:<11}"
        pieces = ""
        for t in targets:
            xs = [r["offset"] for r in rows if r["model"] == m and r["target"] == t]
            line += f"{st.mean(xs):>16.1f}"
            if t == targets[0]:
                dp = [n for r in rows if r["model"] == m and r["target"] == t for n in r["donor_pieces"]]
                tp = [n for r in rows if r["model"] == m and r["target"] == t for n in r["target_pieces"]]
                pieces = f"   {st.mean(dp):.1f} / {st.mean(tp):.1f}"
        print(line + pieces)

    print("\n" + "=" * 96)
    print("② 평균 낼 때 얼마나 줄어드나 — 어텐션(Key) vs 내용(Value, 기준선)")
    print("   1에 가까울수록 손해 없음. Key가 Value보다 뚜렷이 작으면 회전 탓 편향.")
    print("=" * 96)
    print(f"{'모델':<11}{'대상':<13}{'어텐션 줄어듦':>16}{'내용 줄어듦':>15}{'차이(K−V)':>13}{'판정':>22}")
    verdicts = {}
    for m in models:
        for t in targets:
            k = curves(rows, m, t, "key_shrink")
            v = curves(rows, m, t, "value_shrink")
            layers = sorted(k)
            kk = [x for L in layers for x in k[L]]
            vv = [x for L in layers for x in v[L]]
            km, kc = ci95(kk)
            vm, vc = ci95(vv)
            diff = km - vm
            verdict = ("문제 없음" if diff >= -0.05 else
                       "약한 편향" if diff >= -0.15 else "**Key 불리**")
            verdicts[(m, t)] = (km, vm, diff, verdict)
            print(f"{m:<11}{t:<13}{km:>10.3f}±{kc:<5.3f}{vm:>9.3f}±{vc:<5.3f}{diff:>13.3f}{verdict:>22}")

    print("\n" + "=" * 96)
    print("③ 조각들이 같은 방향을 보고 있나 (평균 코사인) — 1이면 완전히 같은 방향")
    print("=" * 96)
    print(f"{'모델':<11}{'대상':<13}{'어텐션 조각 코사인':>20}{'내용 조각 코사인':>18}")
    for m in models:
        for t in targets:
            kc_ = [x for L, xs in curves(rows, m, t, "key_piece_cosine").items() for x in xs]
            vc_ = [x for L, xs in curves(rows, m, t, "value_piece_cosine").items() for x in xs]
            print(f"{m:<11}{t:<13}{st.mean(kc_):>20.3f}{st.mean(vc_):>18.3f}")

    print("\n" + "=" * 96)
    worst = min(verdicts.values(), key=lambda x: x[2])
    if worst[2] >= -0.05:
        print("결론: 어텐션 쪽 값이 내용 쪽보다 더 망가지지 않았다.")
        print("      → 회전(RoPE) 때문에 Key가 불리했다는 우려는 데이터로 기각된다.")
        print("      → '어텐션 경로는 표기 정보를 나르지 않는다'를 그대로 주장할 수 있다.")
    else:
        print("결론: 어텐션 쪽 값이 내용 쪽보다 더 줄어드는 구간이 있다.")
        print(f"      가장 나쁜 차이 {worst[2]:.3f} → 위치를 맞춰 어텐션 조건만 재측정할 것.")
    print("=" * 96)

    if fig_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(targets), len(models),
                                 figsize=(4.2 * len(models), 3.6 * len(targets)), squeeze=False)
        for i, t in enumerate(targets):
            for j, m in enumerate(models):
                ax = axes[i][j]
                k = curves(rows, m, t, "key_shrink")
                v = curves(rows, m, t, "value_shrink")
                layers = sorted(k)
                ax.plot(layers, [st.mean(k[L]) for L in layers],
                        color="tab:gray", label="Key (has rotation)")
                ax.plot(layers, [st.mean(v[L]) for L in layers],
                        color="tab:blue", label="Value (no rotation)")
                ax.axhline(1.0, color="black", linewidth=0.7, linestyle=":")
                ax.set_ylim(0, 1.05)
                ax.set_title(f"{m} — {t}", fontsize=10)
                ax.grid(alpha=0.25)
                if i == len(targets) - 1:
                    ax.set_xlabel("Layer")
                if j == 0:
                    ax.set_ylabel("Shrink ratio")
                if i == 0 and j == 0:
                    ax.legend(fontsize=8)
        fig.suptitle("Diagnostic: does mean-pooling damage Key more than Value?", fontsize=12)
        fig.tight_layout()
        fig.savefig(fig_dir / "kv_phase_shrink.png", dpi=150)
        plt.close(fig)
        print(f"\n그림 → {fig_dir}/kv_phase_shrink.png")


if __name__ == "__main__":
    main()
