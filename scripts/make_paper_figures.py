"""스텝별 결과 그림 생성 — 재집계 기준으로 전부 다시 그린다.

HCLT 템플릿(A4·2단·10pt)에 맞춘다.
  · 단 폭 그림   : 폭 3.3in  → LaTeX에서 width=.46\\textwidth
  · 전체폭 그림  : 폭 7.0in  → figure* 환경
  · 라벨은 **영문**(한글 글꼴 깨짐 방지, CLAUDE.md §6)
  · 제목은 넣지 않는다 — 캡션은 LaTeX에서 단다

만드는 그림 (각 스텝 폴더의 figures/ 아래)
  step1/figures/cliff                    준수율 절벽 (RQ1)
  step3/figures/code_causality_<모델>    통제 뺀 표기 순효과 (RQ2)   모델당 1장
  step3/figures/key_vs_value             내용 vs 어텐션 요약 막대
  step4/figures/layer_alignment_<모델>   관측 층 vs 인과 층 (RQ3)    모델당 1장
  step5/figures/control                  지침 개입: 처치 vs 자기 통제

**모두 단 폭(3.3in)이다.** 모델마다 세로 눈금 범위가 크게 달라 한 장에 몰아넣으면 읽을 수 없고,
2단 조판에서도 단 폭 그림이 배치가 자유롭다.

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
    # **두 지침 방향을 모두 그린다.** camel만 그리면 극적인 절벽만 남아,
    # "RQ1은 방향 조건부로 성립한다"는 이 스텝의 핵심 단서가 그림에서 사라진다.
    rows = load("step1_cliff")
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        c = r["condition"]
        n_viol = c["preceding"]["n_functions"] - c["preceding"]["n_compliant"]
        key = (c["model"]["family"], c["instruction"]["target_notation"])
        by[key][n_viol].append(1.0 if r["metrics"]["extra"]["first_compliant"] else 0.0)

    fig, ax = plt.subplots(figsize=(3.3, 2.45))
    for m, color, mk in (("qwen", "tab:blue", "o"), ("stability", "tab:red", "s")):
        for tgt, style, alpha in (("camel", "-", 0.18), ("snake", "--", 0.10)):
            k = (m, tgt)
            if k not in by:
                continue
            xs = sorted(by[k])
            pts = [ci95(by[k][x]) for x in xs]
            ax.plot(xs, [a for a, _ in pts], marker=mk, ms=3, color=color, linestyle=style,
                    label=f"{LABEL[m].split('-')[0]}, {tgt} instr.")
            ax.fill_between(xs, [a - b for a, b in pts], [a + b for a, b in pts],
                            color=color, alpha=alpha, linewidth=0)
    ax.set_xlabel("Violating names in the context (out of 12)")
    ax.set_ylabel("Compliance rate")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(frameon=False, ncol=2, handlelength=1.6, columnspacing=1.0,
              loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0.0)
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
    """모델마다 그림 하나씩 — 단 폭. 세로 눈금 범위가 모델마다 크게 달라 한 장에 겹치면 못 읽는다."""
    cube = _step3_cube()
    for m in MODELS:
        layers = sorted(cube[m]["unrelated_camel"])
        fig, ax = plt.subplots(figsize=(3.3, 2.3))
        for kind, color, lab in (("value", C_VALUE, "Value only"),
                                 ("key", C_KEY, "Key only"),
                                 ("key_value", C_BOTH, "Key + Value")):
            pts = [ci95(_net(cube, m, L, kind)) for L in layers]
            mu = [a for a, _ in pts]
            ax.plot(layers, mu, color=color, label=lab)
            ax.fill_between(layers, [a - b for a, b in pts], [a + b for a, b in pts],
                            color=color, alpha=0.2, linewidth=0)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Notation-specific recovery")
        ax.margins(y=0.22)                      # 범례가 곡선을 덮지 않도록 위쪽 여유
        ax.legend(frameon=False, loc="upper left", handlelength=1.4)
        ax.grid(alpha=0.25, linewidth=0.4)
        fig.tight_layout(pad=0.4)
        fig.savefig(out / f"code_causality_{m}.pdf")
        fig.savefig(out / f"code_causality_{m}.png")
        plt.close(fig)


# ── 그림 3. 지침: 어디를 보나(step4) vs 어디가 작동하나(step5) ──────────
def fig3_instruction(out: Path):
    """모델마다 그림 하나씩 — 참조량 곡선 + step5 인과 봉우리 표시."""
    obs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in load("step4_instr-observe"):
        m = r["condition"]["model"]["family"]
        cnt = r["metrics"]["extra"].get("span_token_counts", {})
        for L, v in r["metrics"]["per_layer"].items():
            # 지침 쪽은 규칙문 지시어 하나, 코드 쪽은 **camel + snake 12개 전부**.
            # camel만 쓰면 코드 쪽이 절반만 잡혀 "지침을 더 본다"가 부풀려진다
            # (본문에 싣는 6~20배 비율은 12개 전부로 계산한다 — 같은 자여야 한다).
            x, n = v.get("instr_rule_word__attention_weight"), cnt.get("instr_rule_word")
            if x is not None and n:
                obs[m][int(L)]["instr_rule_word"].append(x / n)
            xs = [(v.get(f"{sp}__attention_weight"), cnt.get(sp))
                  for sp in ("code_camel", "code_snake")]
            if all(a is not None and b for a, b in xs):
                obs[m][int(L)]["code"].append(sum(a for a, _ in xs) / sum(b for _, b in xs))
    cause = defaultdict(lambda: defaultdict(list))
    for r in load("step5_instr-cause"):
        m = r["condition"]["model"]["family"]
        for L, v in r["metrics"]["per_layer"].items():
            x = v.get("value__recovery")
            if x is not None:
                cause[m][int(L)].append(x)

    for m in MODELS:
        layers = sorted(obs[m])
        instr = [st.mean(obs[m][L]["instr_rule_word"]) for L in layers]
        code = [st.mean(obs[m][L]["code"]) for L in layers]
        fig, ax = plt.subplots(figsize=(3.3, 2.3))
        ax.plot(layers, instr, color="tab:blue", label="Instruction word")
        ax.plot(layers, code, color="tab:orange", label="Code names (all 12)")
        peak = max(cause[m], key=lambda L: st.mean(cause[m][L]))
        ax.axvline(peak, color="tab:red", linewidth=0.9, linestyle="--")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Attention per token")
        ax.margins(y=0.30)                       # 라벨·범례가 곡선에 닿지 않도록
        top = ax.get_ylim()[1]
        # 세로선 라벨은 축 안쪽 빈 곳에, 선에서 떨어뜨려 놓는다
        ax.text(peak, top * 0.97, f" causal L{peak}", color="tab:red",
                fontsize=6.5, ha="left" if peak < layers[-1] * 0.7 else "right", va="top")
        ax.legend(frameon=False, loc="upper left", handlelength=1.4)
        ax.grid(alpha=0.25, linewidth=0.4)
        fig.tight_layout(pad=0.4)
        fig.savefig(out / f"layer_alignment_{m}.pdf")
        fig.savefig(out / f"layer_alignment_{m}.png")
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
    ax.set_xticklabels([f"{LABEL[m].split('-')[0]}\nL{p}" for m, p in zip(MODELS, peaks)], fontsize=6.5)
    ax.margins(y=0.20)
    ax.set_ylabel("Notation-specific recovery")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "key_vs_value.pdf"); fig.savefig(out / "key_vs_value.png")
    plt.close(fig)


# ── 그림 5. step5 통제 — 처치 vs 자기 통제 ──────────────────────────────
PEAK = {"qwen": 27, "deepseek": 17, "llama": 24, "stability": 19}


def fig5_instruction_control(out: Path):
    """지침 개입: 처치(반대 지침)와 자기 통제를 방식별로 나란히.

    자기 통제 = **같은 지침의 같은 지시어**를 덮는다. 표기 정보가 새로 들어가지 않으므로
    여기서 나오는 값은 '덮어쓰는 행위 자체'의 몫(거품)이다.
    """
    ctrl = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in load("step5_instr-cause-control"):
        ex = r["metrics"]["extra"]
        if ex.get("mode") != "intervene_sweep" or ex.get("undecidable"):
            continue
        m = r["condition"]["model"]["family"]
        vals = r["metrics"]["per_layer"].get(str(PEAK[m])) or r["metrics"]["per_layer"].get(PEAK[m]) or {}
        for k in ("value", "key"):
            x = vals.get(f"{k}__recovery")
            if x is not None:
                ctrl[m][ex["donor"]][k].append(x)
    treat = defaultdict(lambda: defaultdict(list))
    for r in load("step5_instr-cause"):
        m = r["condition"]["model"]["family"]
        if r["metrics"]["extra"].get("undecidable"):
            continue
        vals = r["metrics"]["per_layer"].get(str(PEAK[m]), {})
        for k in ("value", "key"):
            x = vals.get(f"{k}__recovery")
            if x is not None:
                treat[m][k].append(x)

    models = [m for m in MODELS if ctrl[m]]
    x = range(len(models))
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    series = [("value", "treat", C_VALUE, "Value: opposite instr."),
              ("value", "ctrl", "#9ecae1", "Value: self control"),
              ("key", "treat", C_KEY, "Key: opposite instr."),
              ("key", "ctrl", "#d9d9d9", "Key: self control")]
    w = 0.2
    for i, (kind, which, color, lab) in enumerate(series):
        vals, errs = [], []
        for m in models:
            xs = treat[m][kind] if which == "treat" else ctrl[m]["control_self"][kind]
            a, b = ci95(xs) if xs else (float("nan"), 0.0)
            vals.append(a); errs.append(b)
        ax.bar([j + (i - 1.5) * w for j in x], vals, w, yerr=errs, capsize=1.5,
               color=color, label=lab, error_kw={"linewidth": 0.6})
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{LABEL[m].split('-')[0]}\nL{PEAK[m]}" for m in models], fontsize=6.5)
    ax.set_ylabel("Transfer rate")
    ax.margins(y=0.30)
    ax.legend(frameon=False, fontsize=5.8, ncol=2, handlelength=1.0, columnspacing=0.8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "control.pdf"); fig.savefig(out / "control.png")
    plt.close(fig)


def main() -> None:
    jobs = [(fig1_cliff, "docs/step1/figures"),
            (fig2_code_causality, "docs/step3/figures"),
            (fig4_key_vs_value, "docs/step3/figures"),
            (fig3_instruction, "docs/step4/figures"),
            (fig5_instruction_control, "docs/step5/figures")]
    for fn, d in jobs:
        out = Path(d); out.mkdir(parents=True, exist_ok=True)
        fn(out)
        print(f"  {d}/  ←  {fn.__name__}")


if __name__ == "__main__":
    main()
