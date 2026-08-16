"""step3(코드 인과) 보조 그림 — **거품이 얼마나 큰지 층별로 보인다.**

용어는 `docs/개념_트랜스포머와_개입.md` §0에 정의돼 있다. 짧게 다시 적으면:

    공여(donor)   덮어넣을 값을 어디서 떠 왔는가
      compliant         같은 이름의 준수판 (`format_matrix` → `formatMatrix`)   ← 처치
      unrelated_camel   전혀 다른 이름의 camel판
      unrelated_snake   전혀 다른 이름의 snake판                                 ← 통제
    거품(bubble)  통제에서 나오는 값. "덮어쓰는 행위 자체"가 만든 가짜 효과
    순효과        처치 − 통제. 거품을 걷어낸 진짜 몫

왜 필요한가
-----------
논문용 `code_causality_*`는 **순효과만** 그린다. 그런데 그 값이 나오기까지
"원값이 얼마였고 거품이 얼마였나"가 보이지 않는다. step3의 가장 중요한 방법론적 발견이
**거품이 원값의 절반 이상**이라는 것인데, 그게 그림에 없다.

만드는 그림 (모델마다)
----------------------
    donors_<모델>   공여 3종의 **원값** 곡선 (Value 기준) — 거품의 크기가 층마다 보인다
    bubble_<모델>   원값 · 거품 · 순효과를 한 장에 (Value 기준)
                    회색 = 거품, 파랑 = 순효과, 점선 = 원값

쓰는 법
-------
    python scripts/step3_figures.py [--out docs/step3/figures]
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

STEP = "step3_code-cause"
MODELS = ["qwen", "deepseek", "llama", "stability"]
TREAT, FORM, NEG = "compliant", "unrelated_camel", "unrelated_snake"
DONOR_LABEL = {
    TREAT: "compliant (same name, camel)",
    FORM:  "unrelated camel name",
    NEG:   "unrelated snake name (control)",
}
DONOR_COLOR = {TREAT: "tab:blue", FORM: "tab:cyan", NEG: "0.45"}


def ci95(xs):
    if not xs:
        return float("nan"), float("nan")
    if len(xs) < 2:
        return xs[0], 0.0
    return st.mean(xs), 1.96 * st.stdev(xs) / math.sqrt(len(xs))


def load(kind="value"):
    """모델 → 공여 → 층 → {묶음: 되돌림}. 묶음 단위로 짝을 지어야 순효과를 낸다."""
    cube = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for p in sorted(Path("results", STEP).glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        c = r["condition"]; ex = r["metrics"]["extra"]
        m, b, d = c["model"]["family"], c["preceding"]["pool_block"], ex["donor"]
        for L, vals in r["metrics"]["per_layer"].items():
            x = vals.get(f"{kind}__recovery")
            if x is not None:
                cube[m][d][int(L)][b] = x
    return cube


def _band(ax, layers, series, color, label, style="-"):
    pts = [ci95(series[L]) for L in layers]
    ax.plot(layers, [a for a, _ in pts], color=color, label=label, linestyle=style)
    ax.fill_between(layers, [a - b for a, b in pts], [a + b for a, b in pts],
                    color=color, alpha=0.15, linewidth=0)


def _legend_above(ax, ncol=1):
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, frameon=False, ncol=ncol, handlelength=1.5, columnspacing=1.0,
              loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0.0)


def main() -> None:
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv \
        else Path("docs/step3/figures")
    out.mkdir(parents=True, exist_ok=True)
    cube = load("value")

    print("=" * 88)
    print("step3 보조 그림 — 공여별 원값과 거품 (내용만/Value 기준)")
    print("=" * 88)
    print(f"{'모델':<11}{'봉우리층':>9}{'처치 원값':>11}{'거품(통제)':>12}"
          f"{'순효과':>10}{'거품 비중':>11}")

    for m in MODELS:
        if not cube[m]:
            continue
        layers = sorted(cube[m][TREAT])

        # ① 공여 3종 원값
        fig, ax = plt.subplots(figsize=(3.3, 2.6))
        for d in (TREAT, FORM, NEG):
            _band(ax, layers, {L: list(cube[m][d][L].values()) for L in layers},
                  DONOR_COLOR[d], DONOR_LABEL[d], "-" if d != NEG else "--")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Layer"); ax.set_ylabel("Recovery (raw, not net)")
        ax.margins(y=0.10); ax.grid(alpha=0.25, linewidth=0.4)
        _legend_above(ax, ncol=1)
        fig.savefig(out / f"donors_{m}.pdf", bbox_inches="tight")
        fig.savefig(out / f"donors_{m}.png", bbox_inches="tight")
        plt.close(fig)

        # ② 원값 · 거품 · 순효과 — 순효과는 묶음끼리 짝지어 뺀다
        net, bub, rawv = {}, {}, {}
        for L in layers:
            blocks = set(cube[m][FORM][L]) & set(cube[m][NEG][L])
            net[L] = [cube[m][FORM][L][b] - cube[m][NEG][L][b] for b in blocks]
            bub[L] = list(cube[m][NEG][L].values())
            rawv[L] = list(cube[m][TREAT][L].values())
        fig, ax = plt.subplots(figsize=(3.3, 2.6))
        _band(ax, layers, rawv, "0.6", "raw (treatment)", ":")
        _band(ax, layers, bub, "tab:red", "bubble (control only)", "--")
        _band(ax, layers, net, "tab:blue", "net effect (used)", "-")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Layer"); ax.set_ylabel("Recovery")
        ax.margins(y=0.10); ax.grid(alpha=0.25, linewidth=0.4)
        _legend_above(ax, ncol=1)
        fig.savefig(out / f"bubble_{m}.pdf", bbox_inches="tight")
        fig.savefig(out / f"bubble_{m}.png", bbox_inches="tight")
        plt.close(fig)

        Lp = max(layers, key=lambda L: st.mean(net[L]))
        rw, bb, nt = st.mean(rawv[Lp]), st.mean(bub[Lp]), st.mean(net[Lp])
        print(f"{m:<11}{f'L{Lp}':>9}{rw:>11.3f}{bb:>12.3f}{nt:>10.3f}{bb / rw:>10.0%}")

    print(f"\n그림 → {out}/  (모델당 donors·bubble 2장)")
    print("거품 비중 = 통제 원값 ÷ 처치 원값. 이 몫은 표기 정보와 무관하게 나온다.")


if __name__ == "__main__":
    main()
