"""스텝별 결과 그림 생성 — 재집계 기준으로 전부 다시 그린다.

HCLT 템플릿(A4·2단·10pt)에 맞춘다.
  · 단 폭 그림   : 폭 3.3in  → LaTeX에서 width=.46\\textwidth
  · 전체폭 그림  : 폭 7.0in  → figure* 환경
  · 라벨은 **영문**(한글 글꼴 깨짐 방지, CLAUDE.md §6)
  · 제목은 넣지 않는다 — 캡션은 LaTeX에서 단다

만드는 그림 (각 스텝 폴더의 figures/ 아래)
  step1/figures/cliff              준수율 절벽 (RQ1)                단 폭
  step3/figures/code_causality     통제 뺀 표기 순효과 (RQ2)        전체폭
  step3/figures/key_vs_value       내용 vs 어텐션 요약 막대          단 폭
  step4/figures/layer_alignment    관측 층 vs 인과 층 (RQ3)         전체폭

쓰는 법:
    python scripts/make_paper_figures.py
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
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "figure.dpi": 300,
})

MODELS = ["qwen", "deepseek", "llama", "stability"]
LABEL = {"qwen": "Qwen2.5-Coder-3B", "deepseek": "DeepSeek-Coder-6.7B",
         "llama": "Llama-3.2-3B", "stability": "StableCode-3B"}
C_VALUE, C_KEY, C_BOTH = "tab:blue", "0.45", "tab:orange"


def load(step: str):
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(Path("results").rglob(f"{step}/*.json"))]


def ci95(xs):
    if len(xs) < 2:
        return (xs[0] if xs else float("nan")), 0.0
    return st.mean(xs), 1.96 * st.stdev(xs) / math.sqrt(len(xs))


# ── 그림 1. 준수율 절벽 ──────────────────────────────────────────────────
def fig1_cliff(out: Path):
    rows = load("step1_cliff")
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        c = r["condition"]
        if c["instruction"]["target_notation"] != "camel":
            continue                              # snake 방향은 천장 효과 — 본문에 서술만
        n_viol = c["preceding"]["n_functions"] - c["preceding"]["n_compliant"]
        by[c["model"]["family"]][n_viol].append(1.0 if r["metrics"]["extra"]["first_compliant"] else 0.0)

    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    for m, color, mk in (("qwen", "tab:blue", "o"), ("stability", "tab:red", "s")):
        if m not in by:
            continue
        xs = sorted(by[m])
        pts = [ci95(by[m][x]) for x in xs]
        ax.plot(xs, [a for a, _ in pts], marker=mk, ms=3, color=color, label=LABEL[m])
        ax.fill_between(xs, [a - b for a, b in pts], [a + b for a, b in pts],
                        color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel("Violating names in the context (out of 12)")
    ax.set_ylabel("Compliance rate")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linewidth=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "cliff.pdf"); fig.savefig(out / "cliff.png")
    plt.close(fig)


# ── 그림 2. 코드 인과 (통제 뺀 순효과) ──────────────────────────────────
def _step3_cube():
    cube = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for r in load("step3_code-cause"):
        m = r["condition"]["model"]["family"]
        b = r["condition"]["preceding"]["pool_block"]
        d = r["metrics"]["extra"]["donor"]
        for L, v in r["metrics"]["per_layer"].items():
            for k in ("key", "value", "key_value"):
                x = v.get(f"{k}__recovery")
                if x is not None:
                    cube[m][d][int(L)][k][b] = x
    return cube


def _net(cube, m, L, kind, a="unrelated_camel", b="unrelated_snake"):
    xa, xb = cube[m][a][L][kind], cube[m][b][L][kind]
    return [xa[k] - xb[k] for k in sorted(set(xa) & set(xb))]


def fig2_code_causality(out: Path):
    cube = _step3_cube()
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.85), sharey=False)
    for ax, m in zip(axes, MODELS):
        layers = sorted(cube[m]["unrelated_camel"])
        for kind, color, lab in (("value", C_VALUE, "Value only"),
                                 ("key", C_KEY, "Key only"),
                                 ("key_value", C_BOTH, "Key + Value")):
            pts = [ci95(_net(cube, m, L, kind)) for L in layers]
            mu = [a for a, _ in pts]
            ax.plot(layers, mu, color=color, label=lab)
            ax.fill_between(layers, [a - b for a, b in pts], [a + b for a, b in pts],
                            color=color, alpha=0.2, linewidth=0)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(LABEL[m])
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.25, linewidth=0.4)
    axes[0].set_ylabel("Notation-specific\nrecovery")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "code_causality.pdf"); fig.savefig(out / "code_causality.png")
    plt.close(fig)


# ── 그림 3. 지침: 어디를 보나(step4) vs 어디가 작동하나(step5) ──────────
def fig3_instruction(out: Path):
    obs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in load("step4_instr-observe"):
        m = r["condition"]["model"]["family"]
        cnt = r["metrics"]["extra"].get("span_token_counts", {})
        for L, v in r["metrics"]["per_layer"].items():
            for span in ("instr_rule_word", "code_camel"):
                x, n = v.get(f"{span}__attention_weight"), cnt.get(span)
                if x is not None and n:
                    obs[m][int(L)][span].append(x / n)          # 토큰당 평균
    cause = defaultdict(lambda: defaultdict(list))
    for r in load("step5_instr-cause"):
        m = r["condition"]["model"]["family"]
        for L, v in r["metrics"]["per_layer"].items():
            x = v.get("value__recovery")
            if x is not None:
                cause[m][int(L)].append(x)

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95))
    for ax, m in zip(axes, MODELS):
        layers = sorted(obs[m])
        ax.plot(layers, [st.mean(obs[m][L]["instr_rule_word"]) for L in layers],
                color="tab:blue", label="Instruction word")
        ax.plot(layers, [st.mean(obs[m][L]["code_camel"]) for L in layers],
                color="tab:orange", label="Code names")
        ax.set_xlabel("Layer")
        ax.set_title(LABEL[m])
        ax.grid(alpha=0.25, linewidth=0.4)
        peak = max(cause[m], key=lambda L: st.mean(cause[m][L]))
        ax.axvline(peak, color="tab:red", linewidth=0.9, linestyle="--")
        ax.annotate(f"causal L{peak}", xy=(peak, ax.get_ylim()[1]),
                    xytext=(2, -6), textcoords="offset points",
                    fontsize=6, color="tab:red", va="top")
    axes[0].set_ylabel("Attention per token")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "layer_alignment.pdf"); fig.savefig(out / "layer_alignment.png")
    plt.close(fig)


# ── 그림 4. 내용 vs 어텐션 요약 ─────────────────────────────────────────
def fig4_key_vs_value(out: Path):
    cube = _step3_cube()
    peaks, vals, keys, verr, kerr = [], [], [], [], []
    for m in MODELS:
        layers = sorted(cube[m]["unrelated_camel"])
        L = max(layers, key=lambda L: st.mean(_net(cube, m, L, "value")))
        peaks.append(L)
        a, b = ci95(_net(cube, m, L, "value")); vals.append(a); verr.append(b)
        a, b = ci95(_net(cube, m, L, "key")); keys.append(a); kerr.append(b)

    x = range(len(MODELS))
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    ax.bar([i - 0.19 for i in x], vals, 0.38, yerr=verr, capsize=2,
           color=C_VALUE, label="Value only", error_kw={"linewidth": 0.7})
    ax.bar([i + 0.19 for i in x], keys, 0.38, yerr=kerr, capsize=2,
           color=C_KEY, label="Key only", error_kw={"linewidth": 0.7})
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{LABEL[m].split('-')[0]}\nL{p}" for m, p in zip(MODELS, peaks)])
    ax.set_ylabel("Notation-specific recovery")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "key_vs_value.pdf"); fig.savefig(out / "key_vs_value.png")
    plt.close(fig)


def main() -> None:
    jobs = [(fig1_cliff, "docs/step1/figures"),
            (fig2_code_causality, "docs/step3/figures"),
            (fig4_key_vs_value, "docs/step3/figures"),
            (fig3_instruction, "docs/step4/figures")]
    for fn, d in jobs:
        out = Path(d); out.mkdir(parents=True, exist_ok=True)
        fn(out)
        print(f"  {d}/  ←  {fn.__name__}")


if __name__ == "__main__":
    main()
