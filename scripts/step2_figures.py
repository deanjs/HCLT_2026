"""step2(코드 신호 관측) 그림 — **토큰당 평균 기준으로 다시 그린다.**

왜 새로 만드나
--------------
기존 `docs/step2/figures/*.png` 16장은 **작도 코드가 저장소에 없었고**(통합할 때 PNG만
가져왔다), 값이 **구간 합**이었다. 그런데 `results.md` 본문은 "토큰 수로 나눈 값을 쓴다"고
적혀 있어 **글과 그림이 어긋나 있었다.**

합이 왜 문제인가 — 구간 길이가 모델마다 다르다(42묶음 평균):

    모델         camel 토큰   snake 토큰   비율
    Qwen           12.60       12.74     1.01   ← 거의 같다
    Llama          12.60       12.74     1.01
    DeepSeek       16.21       22.45     1.38   ← snake가 38% 길다
    StableCode     14.21       20.40     1.44

밑줄 `_`이 독립 토큰으로 떨어져 snake 구간이 길어진다. 합으로 그리면 DeepSeek·StableCode에서
"snake를 더 본다"가 **길이 때문에** 부풀려진다.

무엇을 그리나 (모델마다 4장, 단 폭 3.3in)
------------------------------------------
    spans_<모델>       토큰당 어텐션 — **구간 5종 전부** (프롬프트의 어디를 보나)
    attention_<모델>   토큰당 어텐션 — 코드 camel / snake / 지침(참고)
    av_<모델>          토큰당 Σa‖v‖ (기여 상한)    camel / snake
    vnorm_<모델>       토큰당 ‖v‖ (내용 크기)      camel / snake
    gap_<모델>         snake − camel 격차          **토큰당(실선) vs 합(점선)** 나란히

`gap_*`이 이 스크립트의 핵심이다 — 정규화가 결론을 바꾸는지 한 장에서 보인다.

읽는 법 (원리)
--------------
어텐션 한 줄은 **전체가 1로 합해진다**(softmax). 그래서 구간 값은 곧 **비중**이다.
`code_snake = 0.16`이면 "이름을 쓰려는 순간 모델 주의의 16%가 snake 이름 토큰들에 갔다".
토큰당 평균은 그 비중을 구간 토큰 수로 나눈 것 — "토큰 하나가 받은 몫".

띠는 42묶음에 대한 95% 신뢰구간(정규근사)이다.

쓰는 법
-------
    python scripts/step2_figures.py [--out docs/step2/figures]
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
    "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "figure.dpi": 300,
})

STEP = "step2_code-observe"
MODELS = ["qwen", "deepseek", "llama", "stability"]
LABEL = {"qwen": "Qwen2.5-Coder-3B", "deepseek": "DeepSeek-Coder-6.7B",
         "llama": "Llama-3.2-3B", "stability": "StableCode-3B"}
C_CAMEL, C_SNAKE, C_INSTR = "tab:blue", "tab:red", "0.55"

# 프롬프트의 어디를 보는가 — step2가 저장한 구간 5종 전부
SPANS = (
    ("code_camel",        C_CAMEL,     "-",  "code: camel names"),
    ("code_snake",        C_SNAKE,     "-",  "code: snake names"),
    ("instruction",       "0.35",      ":",  "instruction (whole)"),
    ("instr_target_word", "tab:green", "--", "instr: required word"),
    ("instr_viol_word",   "tab:purple", "--", "instr: opposite word"),
)


def ci95(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) < 2:
        return xs[0], 0.0
    return st.mean(xs), 1.96 * st.stdev(xs) / math.sqrt(len(xs))


def load():
    """모델 → 층 → 구간 → 지표 → [묶음별 토큰당 값] · 합 기준도 함께."""
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    counts = defaultdict(lambda: defaultdict(list))
    for p in sorted(Path("results", STEP).glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        m = r["condition"]["model"]["family"]
        cnt = r["metrics"]["extra"].get("span_token_counts", {})
        for span, n in cnt.items():
            counts[m][span].append(n)
        for L, vals in r["metrics"]["per_layer"].items():
            for key, x in vals.items():
                if x is None or "__" not in key:
                    continue
                span, metric = key.split("__", 1)
                n = cnt.get(span)
                raw[m][int(L)][span][metric].append(x)
                if n:
                    # v_norm은 이미 토큰당 평균이다(합이 아니다) → 나누지 않는다
                    per[m][int(L)][span][metric].append(x if metric == "v_norm" else x / n)
    return per, raw, counts


def _band(ax, layers, series, color, label, style="-"):
    pts = [ci95(series[L]) for L in layers]
    mu = [a for a, _ in pts]
    ax.plot(layers, mu, color=color, label=label, linestyle=style)
    ax.fill_between(layers, [a - b for a, b in pts], [a + b for a, b in pts],
                    color=color, alpha=0.15 if style == "-" else 0.10, linewidth=0)
    return mu


def _align_zero(ax1, ax2):
    """두 축의 0선을 같은 높이에 맞춘다 — 안 맞추면 부호를 잘못 읽는다."""
    (lo1, hi1), (lo2, hi2) = ax1.get_ylim(), ax2.get_ylim()
    if hi1 <= 0 or hi2 <= 0:
        return
    f = max(-lo1 / (hi1 - lo1), -lo2 / (hi2 - lo2))
    if f <= 0 or f >= 1:
        return
    ax1.set_ylim(-f * hi1 / (1 - f), hi1)
    ax2.set_ylim(-f * hi2 / (1 - f), hi2)


def _legend_above(ax, ncol=3, extra=None):
    """범례를 축 **밖(위)** 에 가로로 둔다 — 그래프 안에 두면 곡선과 겹친다."""
    h, l = ax.get_legend_handles_labels()
    if extra is not None:
        h2, l2 = extra.get_legend_handles_labels()
        h, l = h + h2, l + l2
    ax.legend(h, l, frameon=False, ncol=ncol, handlelength=1.5,
              columnspacing=1.1, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              borderaxespad=0.0)


def _nonneg(ax):
    """어텐션·노름은 음수가 될 수 없다 — 축을 0에서 시작해 빈 공간을 없앤다."""
    ax.set_ylim(bottom=0)


def _save(fig, out: Path, name: str):
    fig.tight_layout(pad=0.4)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)


def main() -> None:
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv \
        else Path("docs/step2/figures")
    out.mkdir(parents=True, exist_ok=True)
    per, raw, counts = load()

    print("=" * 84)
    print("step2 그림 재생성 — 토큰당 평균 기준")
    print("=" * 84)
    print(f"{'모델':<12}{'camel 토큰':>11}{'snake 토큰':>11}{'비율':>8}"
          f"{'봉우리층(토큰당)':>18}{'봉우리층(합)':>14}")

    for m in MODELS:
        layers = sorted(per[m])
        if not layers:
            continue
        A = lambda span, metric, src=per: [src[m][L][span][metric] for L in layers]  # noqa: E731

        # ① 토큰당 어텐션 — camel / snake / 지침(참고)
        fig, ax = plt.subplots(figsize=(3.3, 2.45))
        _band(ax, layers, dict(zip(layers, A("code_camel", "attention_weight"))),
              C_CAMEL, "camel names")
        _band(ax, layers, dict(zip(layers, A("code_snake", "attention_weight"))),
              C_SNAKE, "snake names")
        if per[m][layers[0]].get("instruction"):
            _band(ax, layers, dict(zip(layers, A("instruction", "attention_weight"))),
                  C_INSTR, "instruction (ref.)", style=":")
        ax.set_xlabel("Layer"); ax.set_ylabel("Attention per token")
        ax.margins(y=0.10); ax.grid(alpha=0.25, linewidth=0.4)
        _nonneg(ax); _legend_above(ax, ncol=3)
        _save(fig, out, f"attention_{m}")

        # ①-b 구간 5종 전부 — "프롬프트의 어디를 보나"
        fig, ax = plt.subplots(figsize=(3.3, 2.8))
        for span, color, style, lab in SPANS:
            if not per[m][layers[0]].get(span):
                continue
            _band(ax, layers, dict(zip(layers, A(span, "attention_weight"))),
                  color, lab, style=style)
        ax.set_xlabel("Layer"); ax.set_ylabel("Attention per token")
        ax.margins(y=0.10); ax.grid(alpha=0.25, linewidth=0.4)
        _nonneg(ax); _legend_above(ax, ncol=2)
        _save(fig, out, f"spans_{m}")

        # ② 토큰당 기여 상한 Σa‖v‖
        fig, ax = plt.subplots(figsize=(3.3, 2.45))
        _band(ax, layers, dict(zip(layers, A("code_camel", "av_norm"))), C_CAMEL, "camel names")
        _band(ax, layers, dict(zip(layers, A("code_snake", "av_norm"))), C_SNAKE, "snake names")
        ax.set_xlabel("Layer"); ax.set_ylabel(r"$\sum a\|v\|$ per token")
        ax.margins(y=0.10); ax.grid(alpha=0.25, linewidth=0.4)
        _nonneg(ax); _legend_above(ax, ncol=2)
        _save(fig, out, f"av_{m}")

        # ③ ‖v‖ — 내용 자체의 크기 (이미 토큰당 평균)
        fig, ax = plt.subplots(figsize=(3.3, 2.45))
        _band(ax, layers, dict(zip(layers, A("code_camel", "v_norm"))), C_CAMEL, "camel names")
        _band(ax, layers, dict(zip(layers, A("code_snake", "v_norm"))), C_SNAKE, "snake names")
        ax.set_xlabel("Layer"); ax.set_ylabel(r"$\|v\|$ per token")
        ax.margins(y=0.10); ax.grid(alpha=0.25, linewidth=0.4)
        _nonneg(ax); _legend_above(ax, ncol=2)
        _save(fig, out, f"vnorm_{m}")

        # ④ ★ 격차 — 토큰당(실선) vs 합(점선). 정규화가 결론을 바꾸는지 한 장에.
        #    묶음마다 차이를 먼저 내고(짝지음) 42묶음에 대해 평균·신뢰구간.
        gap_per, gap_raw = {}, {}
        for L in layers:
            cp = per[m][L]["code_camel"]["attention_weight"]
            sp = per[m][L]["code_snake"]["attention_weight"]
            cr = raw[m][L]["code_camel"]["attention_weight"]
            sr = raw[m][L]["code_snake"]["attention_weight"]
            gap_per[L] = [b - a for a, b in zip(cp, sp)]
            gap_raw[L] = [b - a for a, b in zip(cr, sr)]
        fig, ax = plt.subplots(figsize=(3.3, 2.45))
        mu_p = _band(ax, layers, gap_per, C_SNAKE, "per token (used)")
        ax2 = ax.twinx()
        mu_r = [st.mean(gap_raw[L]) for L in layers]
        ax2.plot(layers, mu_r, color="0.45", linestyle="--", linewidth=1.0,
                 label="span sum")
        ax2.set_ylabel("span sum", color="0.45", fontsize=7)
        ax2.tick_params(axis="y", labelcolor="0.45", labelsize=6)
        ax.set_xlabel("Layer"); ax.set_ylabel("snake − camel, per token")
        ax.margins(y=0.12); ax2.margins(y=0.12)
        ax.grid(alpha=0.25, linewidth=0.4)
        _align_zero(ax, ax2)                      # 0선을 같은 높이로
        ax.axhline(0, color="black", linewidth=0.5)
        _legend_above(ax, ncol=2, extra=ax2)
        _save(fig, out, f"gap_{m}")

        # ── 보고 ────────────────────────────────────────────────────
        cA = st.mean(counts[m]["code_camel"]); sA = st.mean(counts[m]["code_snake"])
        Lp = max(layers, key=lambda L: st.mean(gap_per[L]))
        Lr = max(layers, key=lambda L: st.mean(gap_raw[L]))
        mark = "" if Lp == Lr else "  ← 봉우리 이동"
        print(f"{m:<12}{cA:>11.2f}{sA:>11.2f}{sA / cA:>8.2f}"
              f"{f'L{Lp} ({st.mean(gap_per[Lp]):+.4f})':>18}"
              f"{f'L{Lr} ({st.mean(gap_raw[Lr]):+.4f})':>14}{mark}")

    print(f"\n그림 → {out}/  (모델당 attention·av·vnorm·gap 4장, pdf+png)")
    print("띠 = 42묶음 95% 신뢰구간. gap은 묶음마다 차이를 먼저 낸 짝지음 값이다.")


if __name__ == "__main__":
    main()
