"""step4(지침 관측) 그림 — **역할별 구간을 전부 그린다.**

왜 필요한가
-----------
step4는 구간을 **8개** 저장했는데 그림은 `layer_alignment_*` 한 종류뿐이었고,
그 그림은 `instr_rule_word` vs `code_camel` **두 선**만 그린다. 이 스텝의 핵심 기여가
**"지침의 표기어를 역할별로 분리했다"** 인데 정작 그 분리가 그림에 없었다.

지침 문장의 구조 (긍정형·camel 지침 기준)

    You are helping extend an existing Python module.
    In this project we generally write function names in camelCase.       ← 규칙문 (실제 지시)
    Every function name uses one of two styles only: camelCase or snake_case.  ← 후보열거 (대칭 나열)

그래서 표기어가 이렇게 나뉜다:

    instr_rule_word    규칙문의 지시어           1회   ← "지침을 보는가"의 핵심 신호
    instr_cand_camel   후보열거의 camelCase      1회   ┐ 역할이 같은 대칭 통제
    instr_cand_snake   후보열거의 snake_case     1회   ┘
    ---- 초판 키(합산, 참고용) ----
    instr_target_word  = rule_word + 해당 후보열거     **2회**  ← 부풀려진다
    instr_viol_word    = 반대쪽 후보열거만              1회

만드는 그림 (모델마다, 단 폭 3.3in)
------------------------------------
    spans_<모델>          역할별 구간 5종 (규칙문·후보열거 2종·코드 2종), 토큰당
    initial_vs_split_<모델>  **초판 키 vs 역할 분리 키** — 왜 분리가 필요했는지 한 장에
    ratio_<모델>          **지시어 ÷ 코드 비율을 두 기준으로** — 정규화가 결론을 바꾸나
                          1.0 위 = 지시어를 더 본다. 로그 눈금

`layer_alignment_<모델>`(논문용, 관측 봉우리 + 인과 봉우리)은
`scripts/make_paper_figures.py`가 계속 담당한다.

읽는 법
-------
값은 **토큰당 평균**이다(구간 합 ÷ 구간 토큰 수). 어텐션 한 줄은 합이 1이므로
구간 합은 "그 구간이 받은 주의의 비중", 토큰당은 "토큰 하나가 받은 몫"이다.
띠는 묶음에 대한 95% 신뢰구간.

지침 방향(camel·snake) 두 조건을 **평균**한다. 규칙문 지시어는 방향에 따라 실제 단어가
바뀌지만 **역할이 같고**(그 조건이 요구/금지하는 표기어), 후보열거 두 구간은 방향과 무관하게
같은 단어다.

**방향을 평균하는 것이 곧 자리 통제다.** 후보열거 문장은 336조건 전부
`... only: camelCase or snake_case.` 순서로 고정이라, `camelCase`는 늘 1번 자리,
`snake_case`는 늘 2번 자리다. 방향이 바뀌면 **그 단어의 역할만**(목표↔위반) 뒤집힌다.
step2에서 시드를 뒤집어 자리를 상쇄한 것과 같은 장치다(→ `docs/step2/results.md` §3-6).

    방향=camel :  camelCase(1번)=목표   snake_case(2번)=위반
    방향=snake :  camelCase(1번)=위반   snake_case(2번)=목표

방향별 분해는 `results.md` §3-0에 표로 있다. 요지:

    ① 규칙문 지시어 ÷ 코드  두 방향 모두 5.7~21.2배  → 결론 유지. **자리 교란 아님**
    ② 후보열거 camel−snake  방향·층에 따라 부호가 갈린다 → **여기서는 아무 주장도 못 한다**

②는 원래부터 해석에 쓰지 않는다(후보열거 = 대칭 나열, 진짜 명령은 규칙문). 그래도
"후보 camel을 더 본다" 같은 문장을 쓰면 안 된다는 뜻이므로 명시해 둔다.

쓰는 법
-------
    python scripts/step4_figures.py [--out docs/step4/figures]
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

STEP = "step4_instr-observe"
MODELS = ["qwen", "deepseek", "llama", "stability"]

# 역할별 — 이 스텝의 기여가 드러나는 조합
ROLE_SPANS = (
    ("instr_rule_word",  "tab:green",  "-",  "instr: rule word"),
    ("instr_cand_camel", "tab:blue",   "--", "instr: list camelCase"),
    ("instr_cand_snake", "tab:red",    "--", "instr: list snake_case"),
    ("code_camel",       "tab:cyan",   "-",  "code: camel names"),
    ("code_snake",       "tab:orange", "-",  "code: snake names"),
)
# 초판 vs 분리 — 왜 나눴는지
COMPARE_SPANS = (
    ("instr_target_word", "0.45",      "--", "initial (rule + candidates): required word, appears twice"),
    ("instr_viol_word",   "0.65",      ":",  "initial: opposite word, appears once"),
    ("instr_rule_word",   "tab:green", "-",  "split out: rule-sentence word only, appears once"),
)


def ci95(xs):
    if not xs:
        return float("nan"), float("nan")
    if len(xs) < 2:
        return xs[0], 0.0
    return st.mean(xs), 1.96 * st.stdev(xs) / math.sqrt(len(xs))


def load():
    """모델 → 층 → 구간 → [묶음별 토큰당 어텐션] · 합 기준 · 구간별 평균 토큰 수."""
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    counts = defaultdict(lambda: defaultdict(list))
    for p in sorted(Path("results", STEP).glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        m = r["condition"]["model"]["family"]
        cnt = r["metrics"]["extra"].get("span_token_counts", {})
        for span, n in cnt.items():
            counts[m][span].append(n)
        for L, vals in r["metrics"]["per_layer"].items():
            for span, n in cnt.items():
                x = vals.get(f"{span}__attention_weight")
                if x is not None and n:
                    per[m][int(L)][span].append(x / n)
                    raw[m][int(L)][span].append(x)
    return per, raw, counts


def _band(ax, layers, series, color, style, label):
    pts = [ci95(series[L]) for L in layers]
    ax.plot(layers, [a for a, _ in pts], color=color, linestyle=style, label=label)
    ax.fill_between(layers, [a - b for a, b in pts], [a + b for a, b in pts],
                    color=color, alpha=0.15 if style == "-" else 0.10, linewidth=0)


def _finish(fig, ax, out: Path, name: str, ncol=2):
    ax.set_xlabel("Layer"); ax.set_ylabel("Attention per token")
    ax.set_ylim(bottom=0); ax.margins(y=0.10)
    ax.grid(alpha=0.25, linewidth=0.4)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, frameon=False, ncol=ncol, handlelength=1.5, columnspacing=1.1,
              loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0.0)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png")
    plt.close(fig)


def main() -> None:
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv \
        else Path("docs/step4/figures")
    out.mkdir(parents=True, exist_ok=True)
    per, raw, counts = load()

    print("=" * 92)
    print("step4 그림 — 역할별 구간 (토큰당 평균, 지침 방향 2종 평균)")
    print("=" * 92)
    print(f"{'모델':<11}{'규칙문 지시어':>13}{'후보 camel':>11}{'후보 snake':>11}"
          f"{'초판 요구어':>12}{'   ← 토큰 수'}")

    for m in MODELS:
        layers = sorted(per[m])
        if not layers:
            continue

        # ① 역할별 구간
        fig, ax = plt.subplots(figsize=(3.3, 2.8))
        for span, color, style, lab in ROLE_SPANS:
            if any(per[m][L].get(span) for L in layers):
                _band(ax, layers, {L: per[m][L][span] for L in layers}, color, style, lab)
        _finish(fig, ax, out, f"spans_{m}", ncol=2)

        # ② 초판 키 vs 역할 분리 키 — 왜 나눴는지
        fig, ax = plt.subplots(figsize=(3.3, 2.6))
        for span, color, style, lab in COMPARE_SPANS:
            if any(per[m][L].get(span) for L in layers):
                _band(ax, layers, {L: per[m][L][span] for L in layers}, color, style, lab)
        _finish(fig, ax, out, f"initial_vs_split_{m}", ncol=1)

        # ③ ★ 정규화가 결론을 바꾸나 — 지시어 ÷ 코드 비율을 두 기준으로
        fig, ax = plt.subplots(figsize=(3.3, 2.45))
        rp = [st.mean(per[m][L]["instr_rule_word"]) / st.mean(per[m][L]["code_camel"])
              for L in layers]
        rr = [st.mean(raw[m][L]["instr_rule_word"]) / st.mean(raw[m][L]["code_camel"])
              for L in layers]
        ax.plot(layers, rp, color="tab:green", label="per token (used)")
        ax.plot(layers, rr, color="0.45", linestyle="--", label="span sum")
        ax.axhline(1.0, color="black", linewidth=0.6)
        ax.set_xlabel("Layer"); ax.set_ylabel("instruction ÷ code names")
        ax.set_yscale("log"); ax.margins(y=0.10)
        ax.grid(alpha=0.25, linewidth=0.4)
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, l, frameon=False, ncol=2, handlelength=1.5, columnspacing=1.1,
                  loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0.0)
        fig.tight_layout(pad=0.4)
        fig.savefig(out / f"ratio_{m}.pdf"); fig.savefig(out / f"ratio_{m}.png")
        plt.close(fig)

        f = lambda s: st.mean(counts[m][s]) if counts[m][s] else float("nan")  # noqa: E731
        print(f"{m:<11}{f('instr_rule_word'):>13.1f}{f('instr_cand_camel'):>11.1f}"
              f"{f('instr_cand_snake'):>11.1f}{f('instr_target_word'):>12.1f}")

    print(f"\n그림 → {out}/  (모델당 spans·initial_vs_split·ratio 3장, pdf+png)")
    print("초판 요구어 = 규칙문 + 후보열거(2회) → 합으로 재면 부풀려진다. 해석은 규칙문 지시어로 한다.")
    print("논문용 layer_alignment_<모델>은 scripts/make_paper_figures.py가 만든다.")


if __name__ == "__main__":
    main()
