"""step5(지침 인과) 보조 그림 — **층 스윕과 두 통제를 그린다.**

용어는 `docs/개념_트랜스포머와_개입.md` §0. 짧게:

    공여(donor)   덮어넣을 값의 출처
      opposite         반대 지침의 지시어(`snake_case`)            ← 처치
      self             같은 지침의 같은 지시어(`camelCase`)         ← 자기 통제
      unrelated_word   지침 속 무관한 단어(`project`)               ← 음성 통제
    거품          통제에서 나오는 값 = 덮어쓰는 행위 자체가 만든 가짜 효과
    순효과        처치 − 통제 (묶음끼리 짝지어 뺀다)

왜 필요한가
-----------
step3에는 층별 곡선이 있는데 **step5에는 봉우리 층 막대(control.png)뿐**이었다.
같은 자로 비교하려면 층 스윕이 있어야 한다. 그리고 음성 통제(`unrelated_word`)는
표에만 있고 그림이 없었다.

⚠️ **통제는 봉우리 층에서만 돌았다.** 처치는 전 층 스윕(28~36층)인데 통제 스윕은
층이 **1개**뿐이다. 그래서 **층별 순효과 곡선은 만들 수 없다** — step3에는 있는데
step5에는 없는 비대칭이며, 이 스텝의 한계다.

    처치(opposite)        전 층      28~36개 층
    통제(self·무관어)      봉우리 층    1개 층      ← 여기서만 뺄 수 있다

⚠️⚠️ **"창 7층" 실행분도 7층이 아니다.** 조건에 `[14,15,16,17,18,19,20]`처럼 층 목록이
적혀 있지만, 그때 하네스는 목록을 받아도 **첫 층만 돌고 나머지를 조용히 버렸다**
(`extra.layer`가 전부 목록의 첫 값이다: DeepSeek 14 · Qwen 24 · Llama 21 · StableCode 16).
지금은 `runner.py:138`이 목록을 받으면 예외를 던져 같은 사고를 막는다.

    조건에 적힌 층   [14, 15, 16, 17, 18, 19, 20]
    실제로 잰 층     14 하나                        ← 파일명·조건만 보면 7층인 줄 안다

→ 그 672개는 **봉우리−3층 단일 측정**이다. 층 국소성 확인에 쓸 수 없다.

만드는 그림 (모델마다)
----------------------
    sweep_<모델>   층별 처치 곡선(Value·Key) + 봉우리 층 표시 + 통제 값 수평선
                   통제는 점 하나뿐이라 **수평선**으로 그린다

쓰는 법
-------
    python scripts/step5_figures.py [--out docs/step5/figures]
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

MODELS = ["qwen", "deepseek", "llama", "stability"]
TREAT, SELF, NEG = "opposite_instruction", "control_self", "control_unrelated_word"
LABEL = {TREAT: "opposite instruction (treatment)",
         SELF:  "self control (same word)",
         NEG:   "neutral word control ('project')"}
COLOR = {TREAT: "tab:blue", SELF: "tab:red", NEG: "tab:orange"}
GAP_MIN = 1.0


def ci95(xs):
    if not xs:
        return float("nan"), float("nan")
    if len(xs) < 2:
        return xs[0], 0.0
    return st.mean(xs), 1.96 * st.stdev(xs) / math.sqrt(len(xs))


def ctrl_folders() -> list[str]:
    """통제 폴더를 고른다 — 전 층 스윕이 있으면 그쪽(모델별로 갈려 있어도 받는다)."""
    one = Path("results/step5_instr-cause-control-sweep")
    if one.is_dir() and any(one.glob("*.json")):
        return [one.name]
    split = sorted(d.name for d in Path("results").glob("step5_control_sweep_*")
                   if d.is_dir() and any(d.glob("*.json")))
    return split or ["step5_instr-cause-control"]


def load():
    """모델 → 공여 → 방식 → 층 → {(묶음,방향): 전이율}. 스윕 실행분만."""
    cube = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for folder in ["step5_instr-cause", *ctrl_folders()]:
        for p in sorted(Path("results", folder).glob("*.json")):
            r = json.loads(p.read_text(encoding="utf-8"))
            ex = r["metrics"]["extra"]
            if ex.get("mode") != "intervene_sweep":
                continue
            gap = ex.get("gap")
            if gap is None:
                gap = abs(ex["S_clean"] - ex["S_base"])
            if gap < GAP_MIN:                      # 판정 불가는 뺀다
                continue
            c = r["condition"]
            k = (c["preceding"]["pool_block"], c["instruction"]["target_notation"])
            m, d = c["model"]["family"], ex["donor"]
            for L, vals in r["metrics"]["per_layer"].items():
                for kind in ("value", "key"):
                    x = vals.get(f"{kind}__recovery")
                    if x is not None:
                        cube[m][d][kind][int(L)][k] = x
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
        else Path("docs/step5/figures")
    out.mkdir(parents=True, exist_ok=True)
    cube = load()

    print("=" * 92)
    print("step5 보조 그림 — 층 스윕 · 두 통제 · 순효과 (판정 불가 제외)")
    print("=" * 92)
    print(f"{'모델':<11}{'봉우리층':>9}{'처치':>9}{'자기통제':>10}{'음성통제':>10}"
          f"{'순효과(자기)':>13}{'Key 순효과':>12}{'n':>5}")

    for m in MODELS:
        if not cube[m][TREAT]["value"]:
            continue
        layers = sorted(cube[m][TREAT]["value"])

        ctrl_layers = sorted(cube[m][SELF]["value"])
        full = len(ctrl_layers) > 1          # 통제가 전 층에 있는가

        # ① 처치 층 곡선 + 통제(전 층이면 곡선, 한 층뿐이면 수평선)
        Lc = ctrl_layers[0]
        fig, ax = plt.subplots(figsize=(3.3, 2.7))
        _band(ax, layers, {L: list(cube[m][TREAT]["value"][L].values()) for L in layers},
              COLOR[TREAT], LABEL[TREAT], "-")
        _band(ax, layers, {L: list(cube[m][TREAT]["key"][L].values()) for L in layers},
              "0.45", "opposite instr., Key only", ":")
        for d in (SELF, NEG):
            if full:
                Ls = sorted(cube[m][d]["value"])
                _band(ax, Ls, {L: list(cube[m][d]["value"][L].values()) for L in Ls},
                      COLOR[d], LABEL[d], "--")
            elif cube[m][d]["value"].get(Lc):
                v = st.mean(list(cube[m][d]["value"][Lc].values()))
                ax.axhline(v, color=COLOR[d], linestyle="--", linewidth=1.0,
                           label=f"{LABEL[d]} @L{Lc}")
        if not full:
            ax.axvline(Lc, color="tab:red", linewidth=0.8, linestyle=":")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Layer"); ax.set_ylabel("Transfer rate (raw)")
        ax.margins(y=0.10); ax.grid(alpha=0.25, linewidth=0.4)
        _legend_above(ax, ncol=1)
        fig.savefig(out / f"sweep_{m}.pdf", bbox_inches="tight")
        fig.savefig(out / f"sweep_{m}.png", bbox_inches="tight")
        plt.close(fig)

        # ② 층별 순효과 곡선 — 통제가 전 층에 있을 때만 만들 수 있다
        if full:
            common = [L for L in layers if L in cube[m][SELF]["value"]]
            fig, ax = plt.subplots(figsize=(3.3, 2.7))
            for kind, col, sty, lab in (("value", COLOR[TREAT], "-", "net effect · Value"),
                                        ("key", "0.45", ":", "net effect · Key")):
                series = {}
                for L in common:
                    tk, ck = cube[m][TREAT][kind][L], cube[m][SELF][kind][L]
                    ks = set(tk) & set(ck)
                    series[L] = [tk[k] - ck[k] for k in ks]
                _band(ax, common, series, col, lab, sty)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xlabel("Layer"); ax.set_ylabel("Net transfer rate")
            ax.margins(y=0.10); ax.grid(alpha=0.25, linewidth=0.4)
            _legend_above(ax, ncol=1)
            fig.savefig(out / f"net_{m}.pdf", bbox_inches="tight")
            fig.savefig(out / f"net_{m}.png", bbox_inches="tight")
            plt.close(fig)

        # 순효과를 낼 수 있는 층 — 전 층이면 순효과 최대 층, 아니면 통제가 있는 그 한 층
        if full:
            cand = [L for L in layers if L in cube[m][SELF]["value"]]
            def _net(L):
                tk, ck = cube[m][TREAT]["value"][L], cube[m][SELF]["value"][L]
                ks = set(tk) & set(ck)
                return st.mean([tk[k] - ck[k] for k in ks]) if ks else float("-inf")
            Lp = max(cand, key=_net)
        else:
            Lp = Lc
        nets = {}
        for kind in ("value", "key"):
            tk, ck = cube[m][TREAT][kind][Lp], cube[m][SELF][kind][Lp]
            keys = set(tk) & set(ck)
            nets[kind] = [tk[k] - ck[k] for k in keys]

        g = lambda d: (st.mean(list(cube[m][d]["value"][Lp].values()))
                       if cube[m][d]["value"].get(Lp) else float("nan"))   # noqa: E731
        print(f"{m:<11}{f'L{Lp}':>9}{g(TREAT):>9.3f}{g(SELF):>10.3f}{g(NEG):>10.3f}"
              f"{st.mean(nets['value']):>13.3f}{st.mean(nets['key']):>12.3f}"
              f"{len(nets['value']):>5}")

    folders = ctrl_folders()
    print(f"\n통제 폴더: {', '.join('results/' + f for f in folders)}")
    if any("sweep" in f for f in folders):
        print(f"그림 → {out}/  (모델당 sweep·net 2장)")
        print("통제가 전 층에 있으므로 봉우리 층은 **순효과 최대 층**이다(더 이상 유일한 후보가 아니다).")
    else:
        print(f"그림 → {out}/  (모델당 sweep 1장)")
        print("통제는 봉우리 층 한 곳에서만 돌아 수평선으로 그렸다 — 층별 순효과 곡선은 만들 수 없다.")


if __name__ == "__main__":
    main()
