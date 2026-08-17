"""step4 자리 통제 — 지침 방향을 뒤집어 **자리 효과와 역할 효과를 가른다.**

무엇이 문제였나
---------------
지침 마지막 문장의 단어 순서는 **336조건 전부 고정**이다.

    Every function name uses one of two styles only: camelCase or snake_case.
                                                      ↑1번        ↑2번

그런데 어느 쪽이 **목표**인지는 지침 방향이 정한다.

    방향 = camel  →  목표(camelCase)가 1번 자리
    방향 = snake  →  목표(snake_case)가 2번 자리

즉 "목표를 더 본다"와 "1번 자리를 더 본다"가 겹칠 수 있다 — step2에서 표기 배치가
시드로 고정돼 있던 것과 **같은 함정**이다.

어떻게 가르나
-------------
step2는 시드 67을 새로 돌려야 했지만, **step4는 재실행이 필요 없다.**
지침 방향을 처음부터 양쪽 다 돌렸기 때문이다(camel 168 + snake 168).

    방향별로 갈라 부호가 뒤집히면   →  자리 효과
    두 방향에서 부호가 유지되면     →  역할(목표/위반) 효과

두 가지를 잰다
--------------
    ① 주 지표 : 규칙문 지시어 ÷ 코드 이름 (토큰당 배율)
                 규칙문 지시어는 **문장 끝 고정 자리**라 방향이 바뀌어도 자리가 안 바뀐다.
                 → 자리 교란을 안 받아야 한다. 확인용.
    ② 후보열거 : camelCase − snake_case (토큰당 어텐션, **단어 기준**)
                 자리가 1번/2번으로 고정이라 단어 기준 격차가 곧 자리 효과다.
                 → 방향을 바꿔도 자리는 그대로이므로, 부호가 갈리면 못 쓴다.

⚠️ 뒤집을 수 없는 것
--------------------
**지침은 늘 프롬프트 맨 앞**이다. ①의 "지침을 코드보다 몇 배 더 본다"에서 그 자리 이점은
설계상 뒤집을 수 없다 — 한계로 남는다(`results.md` §5).

쓰는 법
-------
    python scripts/step4_direction_control.py
"""

from __future__ import annotations

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

STEP = "step4_instr-observe"
MODELS = ["qwen", "deepseek", "llama", "stability"]
MS = {"qwen": "Qwen", "deepseek": "DeepSeek", "llama": "Llama", "stability": "StableCode"}
DIRS = ("camel", "snake")


def ci95(xs):
    if len(xs) < 2:
        return (xs[0] if xs else float("nan")), float("nan")
    return st.mean(xs), 1.96 * st.stdev(xs) / math.sqrt(len(xs))


def load():
    """모델 → 방향 → 층 → {묶음: (규칙문÷코드 배율, 목표−위반 격차)}."""
    box = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for p in sorted(Path("results", STEP).glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        c = r["condition"]
        m = c["model"]["family"]
        tgt = c["instruction"]["target_notation"]
        b = c["preceding"]["pool_block"]
        cnt = r["metrics"]["extra"]["span_token_counts"]
        if "instr_cand_camel" not in cnt:          # 초판 실행분에는 분리 스팬이 없다
            continue
        for L, v in r["metrics"]["per_layer"].items():
            try:
                rule = v["instr_rule_word__attention_weight"] / cnt["instr_rule_word"]
                code = ((v["code_camel__attention_weight"] + v["code_snake__attention_weight"])
                        / (cnt["code_camel"] + cnt["code_snake"]))
                ca = v["instr_cand_camel__attention_weight"] / cnt["instr_cand_camel"]
                sn = v["instr_cand_snake__attention_weight"] / cnt["instr_cand_snake"]
            except KeyError:
                continue
            # ②는 **단어 기준**(camelCase − snake_case)으로 잰다. 자리가 1번/2번으로
            # 고정이므로 단어 기준 격차가 곧 "자리 효과"다. 역할 기준(목표−위반)으로 재면
            # 방향에 따라 부호가 정의상 뒤집혀 자리 효과와 구분되지 않는다.
            box[m][tgt][int(L)][b] = (rule / code if code else float("nan"), ca - sn)
    return box


def main() -> None:
    box = load()

    print("=" * 96)
    print("① 주 지표 — 규칙문 지시어 ÷ 코드 이름 (토큰당 배율). 1보다 크면 지침을 더 본다")
    print("   규칙문 지시어는 문장 끝 고정 자리라 방향이 바뀌어도 자리가 안 바뀐다")
    print("=" * 96)
    print(f"{'모델':<11}{'봉우리층':>9}{'방향=camel':>13}{'방향=snake':>13}"
          f"{'두 방향 평균':>15}{'부호 유지':>11}")
    for m in MODELS:
        Ls = sorted(set(box[m]["camel"]) & set(box[m]["snake"]))
        if not Ls:
            continue

        def pair(L):
            ks = set(box[m]["camel"][L]) & set(box[m]["snake"][L])
            return [(box[m]["camel"][L][k][0] + box[m]["snake"][L][k][0]) / 2 for k in ks]

        Lp = max(Ls, key=lambda L: st.mean(pair(L)))
        a = st.mean([x[0] for x in box[m]["camel"][Lp].values()])
        b = st.mean([x[0] for x in box[m]["snake"][Lp].values()])
        mu, _ = ci95(pair(Lp))
        ok = "예" if (a > 1) == (b > 1) else "**아니오**"
        print(f"{MS[m]:<11}{'L' + str(Lp):>9}{a:>12.2f}배{b:>12.2f}배{mu:>14.2f}배{ok:>11}")

    print("\n" + "=" * 96)
    print("② 후보열거 — camelCase − snake_case (토큰당 어텐션, 단어 기준)")
    print("   자리가 1번/2번 고정이라 이 격차가 곧 자리 효과다 → 방향별로 갈리면 못 쓴다")
    print("=" * 96)
    print("   ⚠️ 봉우리 층을 어떻게 고르냐에 따라 답이 달라진다 — 그래서 **전 층을 센다**")
    print(f"{'모델':<11}{'층수':>6}{'부호 뒤집히는 층':>18}{'유지되는 층':>14}"
          f"{'두 방향 평균이 0을 무는 층':>26}")
    for m in MODELS:
        Ls = sorted(set(box[m]["camel"]) & set(box[m]["snake"]))
        if not Ls:
            continue
        flip = keep = zero = 0
        for L in Ls:
            ks = set(box[m]["camel"][L]) & set(box[m]["snake"][L])
            a = st.mean([box[m]["camel"][L][k][1] for k in ks])
            b = st.mean([box[m]["snake"][L][k][1] for k in ks])
            mu, e = ci95([(box[m]["camel"][L][k][1] + box[m]["snake"][L][k][1]) / 2 for k in ks])
            flip += (a > 0) != (b > 0)
            keep += (a > 0) == (b > 0)
            zero += mu - e <= 0 <= mu + e
        print(f"{MS[m]:<11}{len(Ls):>6}{flip:>18}{keep:>14}{zero:>26}")

    print("\n①에서 부호가 유지되면 주 지표는 자리 교란을 안 받는다 — step4의 결론은 무사하다.")
    print("②는 층마다 부호가 갈린다 — 어느 층을 고르냐로 답이 뒤바뀌므로 아무 주장도 못 한다.")
    print("   원래도 해석에 안 쓴다(대칭 나열). 이 표는 '왜 안 쓰는가'의 근거다.")
    print("\n⚠️ 지침은 늘 프롬프트 맨 앞이다. ①의 자리 이점은 설계상 뒤집을 수 없다(한계).")


if __name__ == "__main__":
    main()
