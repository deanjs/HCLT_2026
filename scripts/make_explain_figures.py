"""이해용 그림 — 논문용과 별개로, 결과를 처음 보는 사람이 읽을 수 있게 크게 그린다.

논문용(`make_paper_figures.py`)은 2단 조판에 맞춰 3.3인치로 압축돼 있어 눈으로 읽기 어렵다.
이 스크립트는 같은 데이터를 **크게, 주석을 달아** 그린다. 논문에는 안 들어간다.

  docs/step1/figures/explain_cliff.png            절벽이 어디서 무너지나
  docs/step3/figures/explain_bubble.png           거품을 빼면 어떻게 되나
  docs/step5/figures/explain_control.png          처치 vs 통제, 방식별
  docs/step6/figures/explain_headline.png         처방 비교 (헤드라인)
  docs/step6/figures/explain_score_vs_real.png    점수와 실제 준수율이 어긋난다

쓰는 법:
    python scripts/make_explain_figures.py
"""

from __future__ import annotations

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 140,
})

MODELS = ["qwen", "deepseek", "llama", "stability"]
LABEL = {"qwen": "Qwen2.5-Coder-3B", "deepseek": "DeepSeek-Coder-6.7B",
         "llama": "Llama-3.2-3B (general)", "stability": "StableCode-3B"}
PEAK3 = {"qwen": 25, "deepseek": 20, "llama": 15, "stability": 18}
PEAK5 = {"qwen": 27, "deepseek": 17, "llama": 24, "stability": 19}


def load(step: str):
    d = Path("results") / step
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))] \
        if d.is_dir() else []


def ci95(xs):
    if not xs:
        return float("nan"), 0.0
    if len(xs) < 2:
        return xs[0], 0.0
    return st.mean(xs), 1.96 * st.stdev(xs) / math.sqrt(len(xs))


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("  ", path)


# ── step1: 절벽 ──────────────────────────────────────────────────────────
def explain_cliff():
    by = defaultdict(lambda: defaultdict(list))
    for r in load("step1_cliff"):
        c = r["condition"]
        tgt = c["instruction"]["target_notation"]
        n_viol = c["preceding"]["n_functions"] - c["preceding"]["n_compliant"]
        by[tgt][n_viol].append(1.0 if r["metrics"]["extra"]["first_compliant"] else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, tgt, title in zip(axes, ("camel", "snake"),
                              ("Instruction: camelCase\n(fights Python habit)",
                               "Instruction: snake_case\n(agrees with Python habit)")):
        xs = sorted(by[tgt])
        pts = [ci95(by[tgt][x]) for x in xs]
        ax.plot(xs, [a for a, _ in pts], marker="o", ms=7, color="tab:blue", linewidth=2.5)
        ax.fill_between(xs, [a - b for a, b in pts], [a + b for a, b in pts],
                        color="tab:blue", alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("Violating names in the context (out of 12)")
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.08, 1.15)
    axes[0].set_ylabel("Compliance rate")
    axes[0].axvspan(5, 12, color="tab:red", alpha=0.10)
    axes[0].annotate("collapses to exactly 0\nfrom 5 violations on",
                     xy=(7, 0.02), xytext=(6.2, 0.45), fontsize=10, color="tab:red",
                     arrowprops=dict(arrowstyle="->", color="tab:red"))
    axes[1].annotate("stays at the ceiling\n→ cannot be judged",
                     xy=(8, 0.99), xytext=(3.5, 0.62), fontsize=10, color="tab:green",
                     arrowprops=dict(arrowstyle="->", color="tab:green"))
    fig.suptitle("step1  Context violations destroy compliance — but only in one direction",
                 fontsize=13)
    save(fig, Path("docs/step1/figures/explain_cliff.png"))


# ── step3: 거품 ──────────────────────────────────────────────────────────
def explain_bubble():
    cube = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for r in load("step3_code-cause"):
        m = r["condition"]["model"]["family"]
        b = r["condition"]["preceding"]["pool_block"]
        d = r["metrics"]["extra"]["donor"]
        L = str(PEAK3[m])
        v = r["metrics"]["per_layer"].get(L, {})
        for k in ("key", "value"):
            x = v.get(f"{k}__recovery")
            if x is not None:
                cube[m][d][k][b] = x

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    width, xs = 0.35, range(len(MODELS))
    for ax, kind, title in zip(axes, ("value", "key"),
                               ("Changing the CONTENT (Value)",
                                "Changing the ATTENTION (Key)")):
        raw = [st.mean(list(cube[m]["unrelated_camel"][kind].values())) for m in MODELS]
        bub = [st.mean(list(cube[m]["unrelated_snake"][kind].values())) for m in MODELS]
        ax.bar([i - width / 2 for i in xs], raw, width, color="tab:blue",
               label="donor carries camel info")
        ax.bar([i + width / 2 for i in xs], bub, width, color="0.7",
               label="donor carries NO info (bubble)")
        for i, (a, b) in enumerate(zip(raw, bub)):
            ax.annotate("", xy=(i, a), xytext=(i, b),
                        arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.6))
            ax.text(i + 0.05, (a + b) / 2, f"{a - b:+.3f}", color="tab:red",
                    fontsize=10, fontweight="bold")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([LABEL[m].split("-")[0] for m in MODELS], fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0, 0.72)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Recovery at the peak layer")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[1].text(1.5, 0.60, "the two bars are the SAME height\n→ no information transferred",
                 ha="center", fontsize=10, color="tab:red")
    fig.suptitle("step3  Red arrow = what is actually carried. Attention carries nothing.",
                 fontsize=13)
    save(fig, Path("docs/step3/figures/explain_bubble.png"))


# ── step5: 처치 vs 통제 ─────────────────────────────────────────────────
def explain_control():
    ctrl = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in load("step5_instr-cause-control"):
        ex = r["metrics"]["extra"]
        if ex.get("mode") != "intervene_sweep" or ex.get("undecidable"):
            continue
        m = r["condition"]["model"]["family"]
        vals = r["metrics"]["per_layer"].get(str(PEAK5[m])) or {}
        for k in ("value", "key"):
            x = vals.get(f"{k}__recovery")
            if x is not None:
                ctrl[m][ex["donor"]][k].append(x)
    treat = defaultdict(lambda: defaultdict(list))
    for r in load("step5_instr-cause"):
        m = r["condition"]["model"]["family"]
        if r["metrics"]["extra"].get("undecidable"):
            continue
        vals = r["metrics"]["per_layer"].get(str(PEAK5[m]), {})
        for k in ("value", "key"):
            x = vals.get(f"{k}__recovery")
            if x is not None:
                treat[m][k].append(x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    width, xs = 0.35, range(len(MODELS))
    for ax, kind, title in zip(axes, ("value", "key"),
                               ("Changing the CONTENT (Value)",
                                "Changing the ATTENTION (Key)")):
        t = [st.mean(treat[m][kind]) if treat[m][kind] else float("nan") for m in MODELS]
        c = [st.mean(ctrl[m]["control_self"][kind]) if ctrl[m]["control_self"][kind]
             else float("nan") for m in MODELS]
        ax.bar([i - width / 2 for i in xs], t, width, color="tab:blue",
               label="opposite instruction (treatment)")
        ax.bar([i + width / 2 for i in xs], c, width, color="0.7",
               label="same instruction (control)")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([LABEL[m].split("-")[0] for m in MODELS], fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Transfer rate at the peak layer")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[1].text(1.5, 0.75, "treatment and control are identical\n→ nothing transferred",
                 ha="center", fontsize=10, color="tab:red")
    fig.suptitle("step5  Same picture for the instruction. Only content carries the signal.",
                 fontsize=13)
    save(fig, Path("docs/step5/figures/explain_control.png"))


# ── step6: 헤드라인 + 점수와 실제의 어긋남 ───────────────────────────────
def explain_step6():
    rows = load("step6_steer")
    rec = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ex = r["metrics"]["extra"]
        m = r["condition"]["model"]["family"]
        if ex.get("undecidable") or ex.get("recovery") is None:
            continue
        if ex["method"] == "none":
            rec[m]["none"].append(ex["recovery"])
        elif ex["method"] == "value_add" and ex["strength"] == 2.0 and ex["layer"] == PEAK3[m]:
            rec[m]["steer"].append(ex["recovery"])
        elif ex["method"] == "attn_amplify":
            rec[m]["spot"].append(ex["recovery"])

    fig, ax = plt.subplots(figsize=(9, 4.8))
    width, xs = 0.26, range(len(MODELS))
    for off, key, color, lab in ((-1, "none", "0.75", "no intervention"),
                                 (0, "spot", "tab:gray", "Spotlight (attention, best)"),
                                 (1, "steer", "tab:blue", "Value steering @strength 2")):
        vals = [max(rec[m][key]) if key == "spot" and rec[m][key]
                else (st.mean(rec[m][key]) if rec[m][key] else float("nan")) for m in MODELS]
        ax.bar([i + off * width for i in xs], vals, width, color=color, label=lab)
    ax.axhline(1.0, color="tab:red", linestyle=":", linewidth=1.5)
    ax.text(3.35, 1.03, "clean-context ceiling", color="tab:red", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABEL[m] for m in MODELS], fontsize=9)
    ax.set_ylabel("Compliance recovery")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("step6  Pushing the content works. Boosting attention does not.")
    save(fig, Path("docs/step6/figures/explain_headline.png"))

    gen = load("step6_steer-generate")
    if not gen:
        return
    by = defaultdict(lambda: defaultdict(list))
    for r in gen:
        ex = r["metrics"]["extra"]
        by[r["condition"]["model"]["family"]][ex["strength"] or 0.0].append(ex)
    score = defaultdict(dict)
    for r in rows:
        ex = r["metrics"]["extra"]
        if ex["method"] == "value_add" and ex["layer"] == PEAK3[r["condition"]["model"]["family"]] \
                and not ex["undecidable"] and ex["recovery"] is not None:
            score[r["condition"]["model"]["family"]].setdefault(ex["strength"], []).append(ex["recovery"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, m in zip(axes, ("qwen", "deepseek", "llama")):
        xs2 = sorted(by[m])
        ax.plot(xs2, [st.mean([e["compliant"] for e in by[m][x]]) for x in xs2],
                marker="o", ms=7, color="tab:blue", linewidth=2.5, label="ACTUAL compliance")
        ax.plot(xs2, [st.mean([e["name_ok"] for e in by[m][x]]) for x in xs2],
                marker="s", ms=6, color="tab:green", linestyle="--", label="name still valid")
        ax2 = ax.twinx()
        sx = sorted(score[m])
        ax2.plot(sx, [st.mean(score[m][s]) for s in sx], marker="^", ms=6,
                 color="tab:orange", alpha=0.85, label="preference-score recovery")
        ax2.set_ylabel("score recovery", color="tab:orange", fontsize=10)
        ax2.tick_params(axis="y", colors="tab:orange")
        ax.set_ylim(-0.08, 1.15)
        ax.set_xlabel("Steering strength")
        ax.set_title(LABEL[m], fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("rate")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("step6  The score keeps rising while real compliance collapses "
                 "— why generation had to be checked", fontsize=13)
    save(fig, Path("docs/step6/figures/explain_score_vs_real.png"))


if __name__ == "__main__":
    for fn in (explain_cliff, explain_bubble, explain_control, explain_step6):
        fn()
