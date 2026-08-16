"""표에만 있고 그림이 없던 것들 — step6 층 대조 · 진단 A 위상 · step2 토큰 상세.

용어는 `docs/개념_트랜스포머와_개입.md` §0. 여기서 쓰는 것만 다시 적으면:

    조향(steering)   잔차 스트림에 `세기 × 방향`을 더해 행동을 미는 것 (step6)
    방향(vector)     camel 이름 자리의 잔차 평균 − snake 이름 자리의 잔차 평균
    되돌림(recovery) 절벽 바닥에서 목표까지 얼마나 돌아왔나. 1 = 완전 회복
    줄어듦(shrink)   ‖조각들의 평균‖ ÷ 조각 노름들의 평균. 1이면 평균 내도 손해 없음

만드는 그림
-----------
    step6/figures/crosslayer_<모델>   방향을 **뽑은 층**과 **넣는 층**을 달리한 대조
                                      → "방향이 층 특이적인가"
    diag/figures/phase_evidence       위상 근거 두 가지를 한 장에
                                      (조각 코사인 · 덮을 자리와 공여 자리의 위치 차이)
    step2/figures/token_detail_<모델> 이름 **토큰 하나하나**의 어텐션
                                      → 밑줄 `_` 토큰이 값을 얼마나 먹나

쓰는 법
-------
    python scripts/extra_figures.py
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
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "figure.dpi": 300,
})

MODELS = ["qwen", "deepseek", "llama", "stability"]
LABEL = {"qwen": "Qwen2.5-Coder-3B", "deepseek": "DeepSeek-Coder-6.7B",
         "llama": "Llama-3.2-3B", "stability": "StableCode-3B"}


def ci95(xs):
    if not xs:
        return float("nan"), float("nan")
    if len(xs) < 2:
        return xs[0], 0.0
    return st.mean(xs), 1.96 * st.stdev(xs) / math.sqrt(len(xs))


def load(step):
    d = Path("results") / step
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))] \
        if d.is_dir() else []


def _legend_above(ax, ncol=1):
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, frameon=False, ncol=ncol, handlelength=1.5, columnspacing=1.0,
              loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0.0)


# ── ① step6 층 대조 — 방향이 층 특이적인가 ──────────────────────────────
def fig_crosslayer(out: Path):
    """세 조합을 세기별로 비교한다.

        맞는층→맞는층   봉우리 층에서 뽑아 봉우리 층에 넣는다
        엉뚱층→엉뚱층   초반 층에서 뽑아 초반 층에 넣는다
        맞는층→엉뚱층   봉우리 층에서 뽑아 **초반 층에** 넣는다  ← 교차 주입
    """
    same, early, cross = (defaultdict(lambda: defaultdict(list)) for _ in range(3))
    peak, low = {}, {}
    for r in load("step6_steer"):
        ex = r["metrics"]["extra"]
        if ex["method"] != "value_add" or ex.get("undecidable") or ex.get("recovery") is None:
            continue
        m = r["condition"]["model"]["family"]
        peak[m] = max(peak.get(m, -1), ex["layer"]); low[m] = min(low.get(m, 999), ex["layer"])
    for r in load("step6_steer"):
        ex = r["metrics"]["extra"]
        if ex["method"] != "value_add" or ex.get("recovery") is None:
            continue
        m = r["condition"]["model"]["family"]
        (same if ex["layer"] == peak[m] else early)[m][ex["strength"]].append(ex["recovery"])
    for r in load("step6_steer-crosslayer"):
        ex = r["metrics"]["extra"]
        if ex.get("recovery") is None:
            continue
        cross[r["condition"]["model"]["family"]][ex["strength"]].append(ex["recovery"])

    out.mkdir(parents=True, exist_ok=True)
    print("① step6 층 대조 — 방향이 층 특이적인가")
    print(f"  {'모델':<11}{'세기':>5}{'맞는층→맞는층':>15}{'엉뚱층→엉뚱층':>15}{'맞는층→엉뚱층':>15}")
    for m in MODELS:
        if not cross[m]:
            continue
        S = sorted(cross[m])
        fig, ax = plt.subplots(figsize=(3.3, 2.5))
        for src, color, lab in ((same, "tab:blue", f"peak L{peak[m]} → peak"),
                                (early, "tab:cyan", f"early L{low[m]} → early"),
                                (cross, "tab:orange", f"peak L{peak[m]} → early L{low[m]}")):
            pts = [ci95(src[m][s]) for s in S]
            ax.errorbar(S, [a for a, _ in pts], yerr=[b for _, b in pts],
                        marker="o", ms=3, capsize=2, color=color, label=lab,
                        elinewidth=0.7)
        ax.axhline(1.0, color="tab:red", linewidth=0.7, linestyle=":")
        ax.set_xscale("log", base=2); ax.set_xticks(S); ax.set_xticklabels([f"{s:g}" for s in S])
        ax.set_xlabel("Steering strength"); ax.set_ylabel("Compliance recovery")
        ax.margins(y=0.12); ax.grid(alpha=0.25, linewidth=0.4)
        _legend_above(ax, ncol=1)
        fig.savefig(out / f"crosslayer_{m}.pdf", bbox_inches="tight")
        fig.savefig(out / f"crosslayer_{m}.png", bbox_inches="tight")
        plt.close(fig)
        for s in S:
            print(f"  {m:<11}{s:>5g}{st.mean(same[m][s]):>15.3f}"
                  f"{st.mean(early[m][s]):>15.3f}{st.mean(cross[m][s]):>15.3f}")
    print()


# ── ② 진단 A — 위상 근거 두 가지 ────────────────────────────────────────
def fig_phase(out: Path):
    """노름 말고 위상 쪽 근거를 그린다.

        조각 코사인   평균 낼 조각들이 서로 같은 방향인가 (1이면 평균 내도 안 지워짐)
        위치 차이     덮을 자리와 공여 자리가 몇 칸 어긋났나 (0이면 위상 어긋남 없음)
    """
    cos = defaultdict(lambda: defaultdict(list))
    off = defaultdict(list)
    for r in load("diag_kv-phase"):
        ex = r["metrics"]["extra"]
        m = r["condition"]["model"]["family"]
        off[m].append(ex["position_offset_abs_mean"])
        for L, v in r["metrics"]["per_layer"].items():
            cos[m]["key"].append(v["key_piece_cosine"])
            cos[m]["value"].append(v["value_piece_cosine"])
    if not off:
        return
    out.mkdir(parents=True, exist_ok=True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    x = range(len(MODELS))
    a1.bar([i - 0.19 for i in x], [st.mean(cos[m]["key"]) for m in MODELS], 0.38,
           color="0.45", label="Key pieces")
    a1.bar([i + 0.19 for i in x], [st.mean(cos[m]["value"]) for m in MODELS], 0.38,
           color="tab:blue", label="Value pieces")
    a1.set_xticks(list(x)); a1.set_xticklabels([LABEL[m].split("-")[0] for m in MODELS],
                                               rotation=15, ha="right")
    a1.set_ylabel("Mean cosine between pieces"); a1.set_ylim(0, 1)
    a1.grid(alpha=0.25, linewidth=0.4, axis="y")
    a1.legend(frameon=False, loc="upper right")
    a1.set_title("(a) do the averaged pieces point the same way?", fontsize=7.5)

    a2.bar(list(x), [st.mean(off[m]) for m in MODELS], 0.5, color="tab:orange")
    a2.set_xticks(list(x)); a2.set_xticklabels([LABEL[m].split("-")[0] for m in MODELS],
                                               rotation=15, ha="right")
    a2.set_ylabel("Position offset (tokens)")
    a2.grid(alpha=0.25, linewidth=0.4, axis="y")
    a2.set_title("(b) donor vs target position mismatch", fontsize=7.5)

    fig.tight_layout(pad=0.5)
    fig.savefig(out / "phase_evidence.pdf"); fig.savefig(out / "phase_evidence.png")
    plt.close(fig)

    print("② 진단 A — 위상 근거")
    print(f"  {'모델':<11}{'Key 조각 코사인':>16}{'Value 조각 코사인':>18}{'위치 어긋남(칸)':>16}")
    for m in MODELS:
        print(f"  {m:<11}{st.mean(cos[m]['key']):>16.3f}"
              f"{st.mean(cos[m]['value']):>18.3f}{st.mean(off[m]):>16.2f}")
    print()


# ── ③ step2 토큰 상세 — 밑줄 토큰이 값을 먹나 ───────────────────────────
def fig_token_detail(out: Path):
    """이름 안의 토큰 하나하나가 받은 어텐션. 봉우리 층에서 본다."""
    peak = {"qwen": 25, "deepseek": 30, "llama": 14, "stability": 15}
    agg = defaultdict(lambda: defaultdict(list))     # 모델 → 토큰글자 → [값]
    for r in load("step2_code-observe"):
        m = r["condition"]["model"]["family"]
        td = r["metrics"]["extra"].get("token_detail", {})
        L = str(peak.get(m, 0))
        for span in ("code_camel", "code_snake"):
            toks = td.get(span, {}).get("tokens", [])
            vals = td.get(span, {}).get("per_layer", {}).get(L, {}).get("a", [])
            for t, a in zip(toks, vals):
                txt = t["text"].strip()
                key = "_" if txt == "_" else ("(첫 조각)" if t is toks[0] else "(그 외)")
                agg[m][key].append(a)
    if not any(agg.values()):
        return
    out.mkdir(parents=True, exist_ok=True)
    print("③ step2 토큰 상세 — 봉우리 층에서 토큰 종류별 어텐션")
    print(f"  {'모델':<11}{'밑줄 _':>12}{'첫 조각':>12}{'그 외':>12}{'밑줄/그외':>11}")
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    keys = ["_", "(첫 조각)", "(그 외)"]
    x = range(len(MODELS))
    for i, (k, color) in enumerate(zip(keys, ("tab:red", "tab:blue", "0.6"))):
        ax.bar([j + (i - 1) * 0.27 for j in x],
               [st.mean(agg[m][k]) if agg[m][k] else 0 for m in MODELS],
               0.26, color=color, label={"_": "underscore token",
                                         "(첫 조각)": "first piece",
                                         "(그 외)": "other pieces"}[k])
    ax.set_xticks(list(x)); ax.set_xticklabels([LABEL[m].split("-")[0] for m in MODELS],
                                               rotation=15, ha="right")
    ax.set_ylabel("Attention per token (peak layer)")
    ax.grid(alpha=0.25, linewidth=0.4, axis="y")
    _legend_above(ax, ncol=3)
    fig.savefig(out / "token_detail.pdf", bbox_inches="tight")
    fig.savefig(out / "token_detail.png", bbox_inches="tight")
    plt.close(fig)
    for m in MODELS:
        u = st.mean(agg[m]["_"]) if agg[m]["_"] else float("nan")
        o = st.mean(agg[m]["(그 외)"]) if agg[m]["(그 외)"] else float("nan")
        f = st.mean(agg[m]["(첫 조각)"]) if agg[m]["(첫 조각)"] else float("nan")
        print(f"  {m:<11}{u:>12.5f}{f:>12.5f}{o:>12.5f}{u / o:>11.2f}")
    print()


def main() -> None:
    print("=" * 84)
    print("표에만 있던 것들을 그림으로")
    print("=" * 84)
    fig_crosslayer(Path("docs/step6/figures"))
    fig_phase(Path("docs/diag/figures"))
    fig_token_detail(Path("docs/step2/figures"))
    print("완료 — step6/crosslayer_* · diag/phase_evidence · step2/token_detail")


if __name__ == "__main__":
    main()
