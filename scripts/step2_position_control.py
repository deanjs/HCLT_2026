"""step2 자리 통제 — 표기 격차가 **표기 때문인가 자리 때문인가.**

무엇이 문제였나
---------------
`scripts/step2_token_bars.py`가 드러낸 대로, 어텐션은 **프롬프트에 나온 순서를 따라 떨어지는데**
42묶음이 전부 같은 배치라 **위반이 항상 1번 자리**다(1번은 나머지 평균의 2.5배).

    시드 42 :  S C C S S S S C C C C S      ← 42묶음 전부 동일

표기 배치는 묶음이 아니라 **시드**가 정한다(`prompt.py:151`의 seeded shuffle).
그래서 "위반을 더 본다"와 "1번 자리를 더 본다"가 100% 겹쳐 가를 수 없다.

어떻게 가르나
-------------
시드 67이 시드 42의 **정확한 반대 배치**다.

    시드 67 :  C S S C C C C S S S S C      ← 모든 자리가 뒤집힌다

두 시드를 합치면 모든 자리에서 위반이 정확히 절반이 되어 **자리가 수학적으로 소거된다.**

    자리 효과만 있다면      두 시드 평균 위반 = 12자리÷2 = 준수      →  차이 0
    진짜 위반 효과 Δ가 있다면  두 시드 평균 차이 = Δ×6              →  Δ만 남는다

읽는 법
-------
`두 시드 평균` 칸이 답이다. 0을 물면 격차는 자리에서 온 것이고,
양수로 남으면 그게 자리를 지운 진짜 위반 효과다.

만드는 그림
-----------
    position_control          4모델 한 판 — 층별로 시드42·시드67·두 시드 평균 세 곡선
    explain_position_control  이해용 — 봉우리 층에서 두 시드가 어떻게 상쇄되는지 막대

쓰는 법
-------
    python scripts/step2_position_control.py [--out docs/step2/figures]
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "figure.dpi": 300,
})

STEP = "step2_code-observe"
MODELS = ["qwen", "deepseek", "llama", "stability"]
TITLE = {"qwen": "Qwen2.5-Coder-3B", "deepseek": "DeepSeek-Coder-6.7B",
         "llama": "Llama-3.2-3B", "stability": "StableCode-3B"}
BASE_SEED, FLIP_SEED = 42, 67

# 지표 → (한글 이름, 토큰 수로 나누나)
# ‖v‖만 어텐션을 곱하지 않는다 → 자리의 영향을 받지 않아야 한다(음성 통제 노릇을 한다).
METRICS = {
    "attention_weight": ("어텐션", True),
    "av_norm": ("‖a·v‖", True),
    "v_norm": ("‖v‖", False),      # 이미 토큰당 평균이라 다시 나누지 않는다
}
MAIN = "attention_weight"


def ci95(xs):
    if len(xs) < 2:
        return (xs[0] if xs else float("nan")), float("nan")
    return st.mean(xs), 1.96 * st.stdev(xs) / math.sqrt(len(xs))


def _band(ax, layers, series, color, label, style="-"):
    pts = [ci95(series[L]) for L in layers]
    ax.plot(layers, [a for a, _ in pts], color=color, label=label, linestyle=style)
    ax.fill_between(layers, [a - b for a, b in pts], [a + b for a, b in pts],
                    color=color, alpha=0.15, linewidth=0)


def _paired(cube, m, L):
    """같은 묶음끼리 짝지어 두 시드를 평균 — 자리가 수학적으로 소거된다."""
    a, b = cube[m][BASE_SEED][L], cube[m][FLIP_SEED][L]
    return [(a[k] + b[k]) / 2 for k in set(a) & set(b)]


def figures(cube, out: Path) -> None:
    ready = [m for m in MODELS if BASE_SEED in cube[m] and FLIP_SEED in cube[m]]
    if not ready:
        return
    out.mkdir(parents=True, exist_ok=True)

    # ① 논문용 — 층별 세 곡선, 4모델 한 판
    fig, axes = plt.subplots(1, len(ready), figsize=(3.1 * len(ready), 2.6), sharex=False)
    axes = [axes] if len(ready) == 1 else list(axes)
    for ax, m in zip(axes, ready):
        Ls = sorted(set(cube[m][BASE_SEED]) & set(cube[m][FLIP_SEED]))
        _band(ax, Ls, {L: list(cube[m][BASE_SEED][L].values()) for L in Ls},
              "tab:blue", "seed 42 (violation first)", "--")
        _band(ax, Ls, {L: list(cube[m][FLIP_SEED][L].values()) for L in Ls},
              "tab:orange", "seed 67 (order flipped)", "--")
        _band(ax, Ls, {L: _paired(cube, m, L) for L in Ls},
              "black", "mean of both (position removed)", "-")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(TITLE[m], fontsize=8)
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.25, linewidth=0.4)
    axes[0].set_ylabel("Attention gap per token\n(violating − compliant)")
    fig.tight_layout(rect=(0, 0.10, 1, 1))          # 아래 10%를 범례 자리로 비운다
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.savefig(out / "position_control.pdf", bbox_inches="tight")
    fig.savefig(out / "position_control.png", bbox_inches="tight")
    plt.close(fig)

    # ② 이해용 — 봉우리 층에서 두 시드가 상쇄되는 모습
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    xs = range(len(ready))
    w = 0.26
    peaks = {}
    for m in ready:
        peaks[m] = max(cube[m][BASE_SEED],
                       key=lambda L: st.mean(list(cube[m][BASE_SEED][L].values())))
    for off, (s, col, lab) in enumerate((
            (BASE_SEED, "tab:blue", "seed 42 — what we reported before"),
            (FLIP_SEED, "tab:orange", "seed 67 — order flipped"))):
        vals = [ci95(list(cube[m][s][peaks[m]].values())) for m in ready]
        ax.bar([x + (off - 1) * w for x in xs], [a for a, _ in vals], w,
               yerr=[e for _, e in vals], color=col, label=lab, capsize=2)
    vals = [ci95(_paired(cube, m, peaks[m])) for m in ready]
    ax.bar([x + w for x in xs], [a for a, _ in vals], w, yerr=[e for _, e in vals],
           color="0.25", label="mean of both — position removed", capsize=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{TITLE[m]}\n(peak L{peaks[m]})" for m in ready], fontsize=7)
    ax.set_ylabel("Attention gap per token\n(violating − compliant)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    fig.savefig(out / "explain_position_control.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\n그림 → {out}/position_control.(pdf|png) · {out}/explain_position_control.png")


def load():
    """지표 → 모델 → 시드 → 층 → {묶음: snake−camel 격차}."""
    box = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for p in sorted(Path("results", STEP).glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        c = r["condition"]
        sd, m, b = c.get("seed"), c["model"]["family"], c["preceding"]["pool_block"]
        cnt = r["metrics"]["extra"]["span_token_counts"]
        for L, v in r["metrics"]["per_layer"].items():
            for met, (_, per_token) in METRICS.items():
                d = {}
                for sp in ("code_snake", "code_camel"):
                    x = v.get(f"{sp}__{met}")
                    if x is not None:
                        d[sp] = x / cnt[sp] if per_token else x
                if len(d) == 2:
                    box[met][m][sd][int(L)][b] = d["code_snake"] - d["code_camel"]
    return box


def by_metric(box) -> None:
    """세 지표를 나란히 — 어텐션을 곱하는 지표만 무너져야 한다."""
    ready = [m for m in MODELS
             if BASE_SEED in box[MAIN][m] and FLIP_SEED in box[MAIN][m]]
    if not ready:
        return
    print("\n" + "=" * 92)
    print("지표별 — 자리 교란은 **어텐션 경로로** 들어온다. ‖v‖는 어텐션을 곱하지 않아 그대로여야 한다")
    print("=" * 92)
    for met, (name, _) in METRICS.items():
        mul = "어텐션을 곱한다" if met != "v_norm" else "어텐션을 곱하지 않는다 ← 음성 통제"
        print(f"── {name}  ({mul})")
        print(f"{'모델':<11}{'봉우리층':>9}{'시드42':>15}{'시드67':>15}"
              f"{'두 시드 평균':>22}{'0을 무나':>10}{'남은%':>8}")
        for m in ready:
            C = box[met][m]
            if BASE_SEED not in C or FLIP_SEED not in C:
                continue
            Lp = max(C[BASE_SEED], key=lambda L: st.mean(list(C[BASE_SEED][L].values())))
            a42 = st.mean(list(C[BASE_SEED][Lp].values()))
            a67 = st.mean(list(C[FLIP_SEED][Lp].values()))
            ks = set(C[BASE_SEED][Lp]) & set(C[FLIP_SEED][Lp])
            mu, e = ci95([(C[BASE_SEED][Lp][k] + C[FLIP_SEED][Lp][k]) / 2 for k in ks])
            cross = "예" if mu - e <= 0 <= mu + e else "아니오"
            print(f"{m:<11}{f'L{Lp}':>9}{a42:>+15.5f}{a67:>+15.5f}"
                  f"{f'{mu:+.5f}±{e:.5f}':>22}{cross:>10}{mu / a42 * 100:>7.0f}%")
        print()
    print("어텐션·‖a·v‖는 무너지고 ‖v‖는 남으면, 격차가 **어텐션 쪽에서 왔다**는 직접 증거다.")


def main() -> None:
    box = load()
    cube = box[MAIN]

    print("=" * 92)
    print("step2 자리 통제 — 위반−준수 격차(토큰당). 두 시드 평균이 자리를 지운 값이다")
    print("=" * 92)

    missing = [m for m in MODELS if FLIP_SEED not in cube[m]]
    if missing:
        print(f"⚠️ 시드 {FLIP_SEED} 결과가 없는 모델: {', '.join(missing)}")
        print("   notebooks/step2_position-control.ipynb 를 먼저 돌린다.\n")

    print(f"{'모델':<11}{'봉우리층':>9}{'시드42':>18}{'시드67':>18}{'두 시드 평균':>20}{'0을 무나':>10}")
    for m in MODELS:
        if BASE_SEED not in cube[m]:
            continue
        seeds = [s for s in (BASE_SEED, FLIP_SEED) if s in cube[m]]
        layers = sorted(cube[m][BASE_SEED])
        # 봉우리 층은 **시드 42 기준**으로 잡는다(기존 보고와 같은 층에서 비교하려고).
        Lp = max(layers, key=lambda L: st.mean(list(cube[m][BASE_SEED][L].values())))
        cells = []
        for s in (BASE_SEED, FLIP_SEED):
            if s in cube[m]:
                a, e = ci95(list(cube[m][s][Lp].values()))
                cells.append(f"{a:+.5f}±{e:.5f}")
            else:
                cells.append("—")
        if len(seeds) == 2:
            # 같은 묶음끼리 짝지어 평균 — 묶음 난이도가 상쇄된다
            ks = set(cube[m][BASE_SEED][Lp]) & set(cube[m][FLIP_SEED][Lp])
            paired = [(cube[m][BASE_SEED][Lp][k] + cube[m][FLIP_SEED][Lp][k]) / 2 for k in ks]
            a, e = ci95(paired)
            avg, crosses = f"{a:+.5f}±{e:.5f}", "예" if (a - e) <= 0 <= (a + e) else "아니오"
        else:
            avg, crosses = "—", "—"
        print(f"{m:<11}{f'L{Lp}':>9}{cells[0]:>18}{cells[1]:>18}{avg:>20}{crosses:>10}")

    print("\n'0을 무나 = 예'  →  격차는 **자리** 때문이다. step2 §4-(1)을 철회한다.")
    print("'0을 무나 = 아니오' →  자리를 지우고도 남는 **진짜 위반 효과**다. 그 값으로 다시 쓴다.")

    # 봉우리 층 하나만 보면 **시드42로 층을 고른 편향**이 남는다. 전 층을 세어 본다.
    ready = [m for m in MODELS if BASE_SEED in cube[m] and FLIP_SEED in cube[m]]
    if ready:
        print("\n" + "=" * 92)
        print("전 층 점검 — 봉우리 층은 시드42로 고른 것이라 편향이 있다. 층마다 두 시드 평균을 센다")
        print("=" * 92)
        print(f"{'모델':<11}{'층수':>6}{'유의 +':>8}{'유의 −':>8}{'0 포함':>8}"
              f"{'평균 최대 층':>26}")
        for m in ready:
            Ls = sorted(set(cube[m][BASE_SEED]) & set(cube[m][FLIP_SEED]))
            rows = [(L, *ci95(_paired(cube, m, L))) for L in Ls]
            pos = sum(1 for _, a, e in rows if a - e > 0)
            neg = sum(1 for _, a, e in rows if a + e < 0)
            L, a, e = max(rows, key=lambda r: r[1])
            print(f"{m:<11}{len(rows):>6}{pos:>8}{neg:>8}{len(rows) - pos - neg:>8}"
                  f"{f'L{L}  {a:+.5f}±{e:.5f}':>26}")
        print("\n유의 −가 유의 +보다 많으면 '위반을 더 본다'는 층에 따라 **부호가 뒤집힌다** —")
        print("모델의 성질이 아니라 몇 개 층의 국소 현상이다.")

    by_metric(box)

    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv \
        else Path("docs/step2/figures")
    figures(cube, out)


if __name__ == "__main__":
    main()
