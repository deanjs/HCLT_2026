# -*- coding: utf-8 -*-
"""두 자를 나란히 — 점수 회복률과 실제 준수율은 다른 것을 잰다.

왼쪽은 **선호 점수 회복률**(교사강제로 후보 두 개의 확률만 읽은 것),
오른쪽은 **실제 준수율**(조향을 건 채 진짜로 이름을 생성시켜 센 것)이다.
**같은 조건끼리 짝지어** 그리므로 두 자가 어긋나는 곳이 바로 보인다.

    StableCode  회복률 2.81 (4모델 중 1위)  →  준수율 0.095 (꼴찌)
    DeepSeek    회복률 0.24 (낮다)          →  준수율 0.810 (높다)
    Llama       Spotlight 회복률 0.90       →  준수율 0.190

조건은 모델마다 **실제 준수율이 가장 높은 것**으로 골랐다(방법별로 각각).
회복률이 가장 높은 조건을 고르면 우리 쪽에 유리하게 뽑는 셈이 된다.

    python scripts/fig_score_vs_compliance.py [--out docs/step6/figures]
"""
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

for _f in fm.findSystemFonts(fontpaths=[str(Path.home() / ".fonts")]):
    fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "NanumGothic", "font.size": 9,
                     "axes.unicode_minus": False})

MODELS = ["qwen", "deepseek", "llama", "stability"]
NAME = {"qwen": "Qwen2.5\nCoder-3B", "deepseek": "DeepSeek\nCoder-6.7B",
        "llama": "Llama-3.2-3B\n(general)", "stability": "StableCode\n3B"}
# 값 조향을 넣은 층 — 라벨에 반드시 적는다. "후반 층"이라고만 쓰면 어디인지 알 수 없다
LAYER = {"qwen": 25, "deepseek": 20, "llama": 15, "stability": 18}
TASK = ("remove", "duplicat", "dedup", "uniq", "distinct")
MUT, HAIR = "#777777", "#CCCCCC"
BAD, GOOD, NONE = "#B34A28", "#1F6FB4", "#BBBBBB"


def on_task(n):
    return bool(n) and any(w in (n or "").lower() for w in TASK)


def cond_key(ex):
    m = ex.get("method", "none")
    if m == "value_add":
        return f"값조향 세기{ex['strength']:g}"
    if m == "attn_amplify":
        return f"Spotlight {ex.get('span')} ψ{ex.get('psi_target')}"
    return "무개입"


def collect():
    """모델 → 조건 → (회복률, 준수율). 세기 전 구간을 그대로 남긴다."""
    rec, gen = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for f in glob.glob("results/step6_steer/*.json"):
        r = json.load(open(f, encoding="utf-8"))
        ex = r["metrics"]["extra"]
        v = r["metrics"].get("recovery", ex.get("recovery"))
        if v is not None:
            rec[r["condition"]["model"]["family"]][cond_key(ex)].append(v)
    for f in glob.glob("results/step6_steer-generate/*.json"):
        r = json.load(open(f, encoding="utf-8"))
        ex = r["metrics"]["extra"]
        m = r["condition"]["model"]["family"]
        for g in ex.get("generations", [ex]):
            gen[m][cond_key(ex)].append(
                1.0 if (g.get("compliant") and on_task(g.get("name"))) else 0.0)
    return rec, gen


def main() -> None:
    out_dir = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv \
        else Path("docs/step6/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    rec, gen = collect()

    STR = [1, 2, 4, 8]
    COL = {"qwen": "#1F6FB4", "deepseek": "#2CA02C",
           "llama": "#9467BD", "stability": "#D9822B"}

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    for ax, src, title, sub in (
            (axes[0], rec, "Preference-score recovery",
             "Log-prob of two fixed candidates. 1.0 = clean-context level"),
            (axes[1], gen, "Actual compliance rate",
             "Names generated under steering. Correct notation AND on-task")):
        for m in MODELS:
            ys = [st.mean(src[m][f"값조향 세기{a:g}"]) if src[m].get(f"값조향 세기{a:g}")
                  else float("nan") for a in STR]
            ax.plot(STR, ys, marker="o", ms=4.5, lw=1.6, color=COL[m],
                    label=f"{NAME[m].replace(chr(10), ' ')}  (L{LAYER[m]})")
            # Spotlight — 세기 축이 없으므로 오른쪽 끝 밖에 점 하나로 둔다
            sl = [st.mean(v) for k, v in src[m].items() if k.startswith("Spot")]
            if sl:
                ax.plot([13], [max(sl)], marker="s", ms=5, color=COL[m], alpha=0.75)
        ax.axhline(0, color="black", lw=0.8)
        if src is rec:
            ax.axhline(1.0, color=HAIR, lw=1, ls="--")
            ax.text(0.75, 1.06, "ceiling 1.0", fontsize=7, color=MUT)
        ax.set_xscale("log", base=2)
        ax.set_xticks(STR + [13])
        ax.set_xticklabels([str(a) for a in STR] + ["Spot-\nlight"], fontsize=8)
        ax.set_xlabel("Steering strength  α", fontsize=9)
        ax.axvline(10.5, color=HAIR, lw=1)
        ax.set_title(title, fontsize=10.5, pad=18)
        ax.text(0.5, 1.02, sub, transform=ax.transAxes, ha="center", fontsize=7.6, color=MUT)
        ax.grid(alpha=0.25, lw=0.4); ax.set_axisbelow(True)
    axes[1].set_ylim(-0.04, 1.08)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               fontsize=8)
    fig.text(0.5, 0.055,
             "Every strength is shown — no best-case selection. "
             "Spotlight (square) is the max over span x psi; it has no strength axis.",
             ha="center", fontsize=7.4, color=MUT)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"score_vs_compliance_paired.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"→ {out_dir}/score_vs_compliance_paired.(png|pdf)")


if __name__ == "__main__":
    main()
