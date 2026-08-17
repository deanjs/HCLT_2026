"""step2 자리 통제 — 표기 격차가 **표기 때문인가 자리 때문인가.**

무엇이 문제였나
---------------
`scripts/step2_token_bars.py`가 드러낸 대로, 어텐션은 **프롬프트에 나온 순서를 따라 떨어지는데**
42묶음이 전부 같은 배치라 **위반이 항상 1번 자리**다(1번은 나머지 평균의 2.5배).

    시드 42 :  S C C S S S S C C C C S      ← 42묶음 전부 동일

표기 배치는 묶음이 아니라 **시드**가 정한다(`prompt.py:151`의 seeded shuffle).
그래서 "위반을 더 본다"와 "1번 자리를 더 본다"가 100% 겹쳐 가를 수 없다.

어떻게 가르나
-------------
시드 67이 시드 42의 **정확한 반대 배치**다.

    시드 67 :  C S S C C C C S S S S C      ← 모든 자리가 뒤집힌다

두 시드를 합치면 모든 자리에서 위반이 정확히 절반이 되어 **자리가 수학적으로 소거된다.**

    자리 효과만 있다면      두 시드 평균 위반 = 12자리÷2 = 준수      →  차이 0
    진짜 위반 효과 Δ가 있다면  두 시드 평균 차이 = Δ×6              →  Δ만 남는다

읽는 법
-------
`두 시드 평균` 칸이 답이다. 0을 물면 격차는 자리에서 온 것이고,
양수로 남으면 그게 자리를 지운 진짜 위반 효과다.

쓰는 법
-------
    python scripts/step2_position_control.py
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

STEP = "step2_code-observe"
MODELS = ["qwen", "deepseek", "llama", "stability"]
BASE_SEED, FLIP_SEED = 42, 67


def ci95(xs):
    if len(xs) < 2:
        return (xs[0] if xs else float("nan")), float("nan")
    return st.mean(xs), 1.96 * st.stdev(xs) / math.sqrt(len(xs))


def main() -> None:
    cube = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for p in sorted(Path("results", STEP).glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        c = r["condition"]
        sd, m, b = c.get("seed"), c["model"]["family"], c["preceding"]["pool_block"]
        cnt = r["metrics"]["extra"]["span_token_counts"]
        for L, v in r["metrics"]["per_layer"].items():
            d = {}
            for sp in ("code_snake", "code_camel"):
                x = v.get(sp + "__attention_weight")
                if x is not None:
                    d[sp] = x / cnt[sp]                     # 토큰당
            if len(d) == 2:
                cube[m][sd][int(L)][b] = d["code_snake"] - d["code_camel"]

    print("=" * 92)
    print("step2 자리 통제 — 위반−준수 격차(토큰당). 두 시드 평균이 자리를 지운 값이다")
    print("=" * 92)

    missing = [m for m in MODELS if FLIP_SEED not in cube[m]]
    if missing:
        print(f"⚠️ 시드 {FLIP_SEED} 결과가 없는 모델: {', '.join(missing)}")
        print("   notebooks/step2_position-control.ipynb 를 먼저 돌린다.\n")

    print(f"{'모델':<11}{'봉우리층':>9}{'시드42':>18}{'시드67':>18}{'두 시드 평균':>20}{'0을 무나':>10}")
    for m in MODELS:
        if BASE_SEED not in cube[m]:
            continue
        seeds = [s for s in (BASE_SEED, FLIP_SEED) if s in cube[m]]
        layers = sorted(cube[m][BASE_SEED])
        # 봉우리 층은 **시드 42 기준**으로 잡는다(기존 보고와 같은 층에서 비교하려고).
        Lp = max(layers, key=lambda L: st.mean(list(cube[m][BASE_SEED][L].values())))
        cells = []
        for s in (BASE_SEED, FLIP_SEED):
            if s in cube[m]:
                a, e = ci95(list(cube[m][s][Lp].values()))
                cells.append(f"{a:+.5f}±{e:.5f}")
            else:
                cells.append("—")
        if len(seeds) == 2:
            # 같은 묶음끼리 짝지어 평균 — 묶음 난이도가 상쇄된다
            ks = set(cube[m][BASE_SEED][Lp]) & set(cube[m][FLIP_SEED][Lp])
            paired = [(cube[m][BASE_SEED][Lp][k] + cube[m][FLIP_SEED][Lp][k]) / 2 for k in ks]
            a, e = ci95(paired)
            avg, crosses = f"{a:+.5f}±{e:.5f}", "예" if (a - e) <= 0 <= (a + e) else "아니오"
        else:
            avg, crosses = "—", "—"
        print(f"{m:<11}{f'L{Lp}':>9}{cells[0]:>18}{cells[1]:>18}{avg:>20}{crosses:>10}")

    print("\n'0을 무나 = 예'  →  격차는 **자리** 때문이다. step2 §4-(1)을 철회한다.")
    print("'0을 무나 = 아니오' →  자리를 지우고도 남는 **진짜 위반 효과**다. 그 값으로 다시 쓴다.")


if __name__ == "__main__":
    main()
