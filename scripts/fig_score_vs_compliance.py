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
        "llama": "Llama-3.2\n3B (범용)", "stability": "StableCode\n3B"}
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

    out = {}
    for m in MODELS:
        common = [k for k in gen[m] if rec[m].get(k)]

        def pair(k):
            return st.mean(rec[m][k]), st.mean(gen[m][k])

        va = [k for k in common if k.startswith("값조향")]
        sl = [k for k in common if k.startswith("Spot")]
        bv = max(va, key=lambda k: pair(k)[1])
        bs = max(sl, key=lambda k: pair(k)[1])
        out[m] = {"none": pair("무개입"),
                  "value": (bv.replace("값조향 ", ""),) + pair(bv),
                  "spot": (bs.split()[-1],) + pair(bs)}
    return out


def main() -> None:
    out_dir = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv \
        else Path("docs/step6/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    D = collect()

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    xs = range(len(MODELS))
    w = 0.26
    panels = [
        (axes[0], 1, "왼쪽 — 선호 점수 회복률",
         "후보 두 개의 확률만 읽은 값. 1 = 오염이 없던 상태까지 회복"),
        (axes[1], 2, "오른쪽 — 실제 준수율",
         "조향을 건 채 이름을 생성시켜 센 값. 표기 맞음 × 과제 맞음"),
    ]
    for ax, idx, title, sub in panels:
        for off, (key, col, lab) in enumerate((
                ("none", NONE, "무개입"),
                ("value", GOOD, "값 조향 (우리)"),
                ("spot", BAD, "Spotlight (기존)"))):
            vals = [D[m][key][idx] if key != "none" else D[m][key][idx - 1] for m in MODELS]
            ax.bar([x + (off - 1) * w for x in xs], vals, w, color=col, label=lab)
            for x, v in zip(xs, vals):
                ax.text(x + (off - 1) * w, v + (0.06 if v >= 0 else -0.16), f"{v:.3f}",
                        ha="center", fontsize=6.8, color=col if key != "none" else MUT)
        ax.axhline(0, color="black", lw=0.8)
        if idx == 1:
            ax.axhline(1.0, color=HAIR, lw=1, ls="--")
            ax.text(-0.55, 1.06, "천장 1.0", fontsize=7, color=MUT)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([NAME[m] for m in MODELS], fontsize=8)
        ax.set_title(title, fontsize=10.5, pad=18)
        ax.text(0.5, 1.02, sub, transform=ax.transAxes, ha="center",
                fontsize=7.6, color=MUT)
        ax.grid(axis="y", alpha=0.25, lw=0.4)
        ax.set_axisbelow(True)
    axes[1].set_ylim(-0.03, 1.15)
    axes[0].set_ylim(-0.55, 3.25)

    # 어긋나는 곳 두 군데를 직접 가리킨다 — 이 그림의 요점이다
    axes[0].annotate("4모델 중 1위", xy=(3, D["stability"]["value"][1]), xytext=(1.55, 2.55),
                     fontsize=7.6, color=BAD,
                     arrowprops=dict(arrowstyle="->", color=BAD, lw=1))
    axes[1].annotate("그런데 꼴찌", xy=(3, D["stability"]["value"][2]), xytext=(2.05, 0.45),
                     fontsize=7.6, color=BAD,
                     arrowprops=dict(arrowstyle="->", color=BAD, lw=1))

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    lay = " · ".join(f"{NAME[m].split(chr(10))[0]} L{LAYER[m]}" for m in MODELS)
    fig.text(0.5, 0.055, f"값 조향을 넣은 층 — {lay}   ·   조건은 모델마다 준수율이 가장 높은 것",
             ha="center", fontsize=7.4, color=MUT)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"score_vs_compliance_paired.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"→ {out_dir}/score_vs_compliance_paired.(png|pdf)")


if __name__ == "__main__":
    main()
