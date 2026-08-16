"""step5(지침 인과) 재집계 — 방향을 나누고, 평균의 함정을 걷어낸다.

왜 필요한가
-----------
지금까지의 step5 보고에는 세 가지 문제가 있었다.
  ① camel 지침 조건과 snake 지침 조건을 **하나로 평균**했다. StableCode는 두 방향이 3배 차이라
     평균 하나로는 그 사실이 보이지 않는다.
  ② 전이율은 위가 막혀 있지 않은 비율이라 **1을 넘는 값**이 평균을 부풀린다. 몇 개가 넘는지
     밝힌 적이 없다.
  ③ "판정 불가" 기준선(|S_반대 − S_기준| < 1.0)이 노트북 안 상수로만 있고, 그 기준을 옮기면
     결론이 어떻게 되는지 확인한 기록이 없다. Llama는 기준선이 분포 한가운데를 지난다.

이 스크립트는 재실험 없이 ①②③을 전부 다시 계산한다. 결과 파일은 읽기만 한다(§6 불변).

쓰는 법
-------
    python scripts/step5_reaggregate.py results/step5_instr-cause [--figs docs/step5/figures]
"""

from __future__ import annotations

import math
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

KINDS = ("key", "value", "key_value")
THRESHOLDS = (0.5, 1.0, 2.0)


def load(root: str | Path) -> list[dict]:
    """조건마다 (모델, 지침방향, 영향력, 층별 전이율)을 뽑아 평평한 목록으로."""
    rows = []
    for p in sorted(Path(root).rglob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        ex = r["metrics"]["extra"]
        rows.append({
            "model": r["condition"]["model"]["family"],
            "direction": r["condition"]["instruction"]["target_notation"],
            "block": r["condition"]["preceding"]["pool_block"],
            "gap": abs(ex["S_clean"] - ex["S_base"]),   # 영향력 = 지침이 실제로 흔든 폭
            "S_base": ex["S_base"], "S_clean": ex["S_clean"],
            "per_layer": {int(L): v for L, v in r["metrics"]["per_layer"].items()},
        })
    return rows


def ci95(xs):
    if len(xs) < 2:
        return (xs[0] if xs else float("nan")), float("nan")
    return st.mean(xs), 1.96 * st.stdev(xs) / math.sqrt(len(xs))


def curve(rows, model, kind, direction=None, gap_min=0.0):
    """층 → 전이율 목록. direction=None이면 두 방향 합침."""
    out = defaultdict(list)
    for r in rows:
        if r["model"] != model or r["gap"] < gap_min:
            continue
        if direction and r["direction"] != direction:
            continue
        for L, v in r["per_layer"].items():
            x = v.get(f"{kind}__recovery")
            if x is not None:
                out[L].append(x)
    return out


def peak(c):
    return max(c, key=lambda L: st.mean(c[L]))


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

    print("=" * 96)
    print("① 지침 방향을 나눠서 본다 (봉우리 층, 내용만(Value) 기준)")
    print("=" * 96)
    print(f"{'모델':<11}{'봉우리':>7}{'합침 평균':>12}{'합침 중앙값':>13}"
          f"{'camel 방향':>13}{'snake 방향':>13}{'전이율>1':>10}")
    peaks = {}
    for m in models:
        c = curve(rows, m, "value")
        L = peak(c)
        peaks[m] = L
        both = c[L]
        cm = st.mean(curve(rows, m, "value", "camel")[L])
        sn = st.mean(curve(rows, m, "value", "snake")[L])
        over = sum(1 for x in both if x > 1)
        mu, ci = ci95(both)
        print(f"{m:<11}{L:>7}{mu:>7.2f}±{ci:<4.2f}{st.median(both):>13.2f}"
              f"{cm:>13.2f}{sn:>13.2f}{over:>7}/{len(both):<3}")
    print("\n평균과 중앙값이 크게 다르거나 '전이율>1'이 많으면, 평균 하나로 보고하면 안 된다.")

    print("\n" + "=" * 96)
    print("② 어텐션만(Key) vs 내용만(Value) — 방향별")
    print("=" * 96)
    print(f"{'모델':<11}{'방향':<8}{'내용만(Value)':>16}{'어텐션만(Key)':>16}{'Key/Value':>11}")
    for m in models:
        L = peaks[m]
        for d in ("camel", "snake"):
            v = st.mean(curve(rows, m, "value", d)[L])
            k = st.mean(curve(rows, m, "key", d)[L])
            print(f"{m:<11}{d:<8}{v:>16.3f}{k:>16.3f}{(k / v if v else float('nan')):>11.2f}")
        print()

    print("=" * 96)
    print("③ 판정 불가 기준선을 옮기면 결론이 바뀌나 (영향력 = |S_반대 − S_기준|)")
    print("=" * 96)
    print(f"{'모델':<11}{'영향력 최소':>11}{'중앙':>8}{'최대':>8}" +
          "".join(f"{f'기준 {t}':>22}" for t in THRESHOLDS))
    print(f"{'':<11}{'':>11}{'':>8}{'':>8}" +
          "".join(f"{'남는 조건 / 전이율':>22}" for _ in THRESHOLDS))
    for m in models:
        gs = [r["gap"] for r in rows if r["model"] == m]
        line = f"{m:<11}{min(gs):>11.2f}{st.median(gs):>8.2f}{max(gs):>8.2f}"
        for t in THRESHOLDS:
            c = curve(rows, m, "value", gap_min=t)
            if not c:
                line += f"{'전부 제외':>22}"
                continue
            L = peak(c)
            n = len(c[L])
            line += f"{f'{n}개 / {st.mean(c[L]):.2f} (L{L})':>22}"
        print(line)
    print("\n남는 조건 수가 기준선에 따라 크게 흔들리면, 그 모델의 결론은 기준선에 민감하다는 뜻이다.")

    if fig_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for m in models:
            layers = sorted(curve(rows, m, "value"))
            fig, ax = plt.subplots(figsize=(6.5, 4))
            for d, color in (("camel", "tab:blue"), ("snake", "tab:red")):
                c = curve(rows, m, "value", d)
                pts = [ci95(c[L]) for L in layers]
                mu = [a for a, _ in pts]
                ax.plot(layers, mu, color=color, label=f"target = {d}", linewidth=1.6)
                ax.fill_between(layers, [a - b for a, b in pts], [a + b for a, b in pts],
                                color=color, alpha=0.2, linewidth=0)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.axhline(1, color="gray", linewidth=0.7, linestyle=":")
            ax.set_xlabel("Intervened layer")
            ax.set_ylabel("Transfer rate (Value only)")
            ax.set_title(f"step5 instruction causality — {m} (by instruction direction)")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(fig_dir / f"direction_{m}.png", dpi=150)
            plt.close(fig)
        print(f"\n방향별 그림 → {fig_dir}/direction_<모델>.png")


if __name__ == "__main__":
    main()
