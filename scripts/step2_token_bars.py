"""step2 토큰별 어텐션 막대 — **프롬프트 안 어느 단어를 얼마나 봤나.**

왜 필요한가
-----------
지금까지 그림은 전부 **구간 합**(camel 12칸을 다 더한 값)이었다. 그래서
"모델이 위반 이름을 더 본다"는 알 수 있어도 **어느 단어를 봤는지**는 볼 수 없었다.
그런데 결과 파일의 `extra.token_detail`에는 토큰 하나하나의 어텐션이 층마다
이미 저장돼 있다(`per_layer.<층>.a`). 그걸 쓰지 않고 있었다.

만드는 그림
-----------
    tokens_<모델>   봉우리 층에서 이름 토큰 24개(camel 12 + snake 12)를 개별 막대로.
                    가로축 = 프롬프트에 나온 순서(칸 번호), 색 = 어느 무리인가.

읽는 법
-------
어텐션 한 줄은 합이 1이므로 막대 높이는 곧 **그 토큰 하나가 받은 주의의 비중**이다.
막대 높이를 24개 다 더하면 구간 합 두 개(0.155 + 0.126)가 된다.

★ 이 그림이 드러낸 것 — **자리 교란**
----------------------------------
막대를 그려 보면 어텐션이 **프롬프트에 나온 순서를 따라 떨어진다.** 그런데 42묶음이
**전부 같은 배치**다:

    자리   1    2    3    4    5    6    7    8    9   10   11   12
    표기   S    C    C    S    S    S    S    C    C    C    C    S      ← 42묶음 동일

**위반(S)이 항상 1번 자리**이고, 1번은 나머지 이름 평균의 2.5배를 받는다.
그래서 "위반을 더 본다"는 관측이 **"1번 자리를 더 본다"와 구별되지 않는다.**
아래 ②가 그 검산이다 — 1번 자리를 빼면 3모델에서 차이가 0으로 사라진다.

위치와 표기가 모든 묶음에서 100% 겹치므로 **이 데이터로는 두 설명을 가를 수 없다.**
step3(인과)는 값을 덮어쓰고 점수 변화를 재므로 이 교란과 무관하다.

쓰는 법
-------
    python scripts/step2_token_bars.py [--out docs/step2/figures]
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "figure.dpi": 300,
})

STEP = "step2_code-observe"
MODELS = ["qwen", "deepseek", "llama", "stability"]
LABEL = {"qwen": "Qwen2.5-Coder-3B", "deepseek": "DeepSeek-Coder-6.7B",
         "llama": "Llama-3.2-3B", "stability": "StableCode-3B"}
# 코드 봉우리 층 — results.md §3의 "코드 봉우리"와 같다
PEAK = {"qwen": 25, "deepseek": 30, "llama": 14, "stability": 0}


def _name_groups(d):
    """토큰 인덱스가 연속인 것끼리 묶어 **이름 단위**로 만든다.

    이름당 토큰 수가 2개가 아닌 모델이 있어(DeepSeek 3~4조각) 고정 폭으로 자르면 어긋난다.
    """
    ts = [t["t"] for t in d["tokens"]]
    g = [[0]]
    for i in range(1, len(ts)):
        if ts[i] == ts[i - 1] + 1:
            g[-1].append(i)
        else:
            g.append([i])
    return g


def main() -> None:
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv \
        else Path("docs/step2/figures")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("step2 토큰별 어텐션 — 봉우리 층에서 어느 단어를 얼마나 봤나")
    print("=" * 84)

    for m in MODELS:
        L = str(PEAK[m])
        # 묶음마다 이름이 다르므로 **블록 0 하나**를 대표로 그린다(어느 단어인지 보여야 하니까).
        f = next((p for p in sorted(Path("results", STEP).glob("*.json"))
                  if json.loads(p.read_text(encoding="utf-8"))["condition"]["model"]["family"] == m
                  and json.loads(p.read_text(encoding="utf-8"))["condition"]["preceding"]["pool_block"] == 0),
                 None)
        if f is None:
            continue
        ex = json.loads(f.read_text(encoding="utf-8"))["metrics"]["extra"]
        td = ex.get("token_detail") or {}
        if not td:
            continue

        rows = []
        for span, color, lab in (("code_snake", "tab:red", "violating (snake)"),
                                 ("code_camel", "tab:blue", "compliant (camel)")):
            d = td.get(span)
            if not d or L not in d["per_layer"]:
                continue
            for tok, a in zip(d["tokens"], d["per_layer"][L]["a"]):
                rows.append((tok["t"], tok["text"], a, color, lab))
        if not rows:
            continue
        rows.sort()                                     # 프롬프트에 나온 순서대로

        fig, ax = plt.subplots(figsize=(7.0, 2.4))
        xs = range(len(rows))
        seen = set()
        for i, (_, _, a, color, lab) in zip(xs, rows):
            ax.bar(i, a, 0.75, color=color, label=lab if lab not in seen else None)
            seen.add(lab)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([t.strip() for _, t, _, _, _ in rows], rotation=60,
                           ha="right", fontsize=5.5)
        ax.set_ylabel("Attention share")
        ax.set_xlabel(f"Name tokens, in prompt order  (layer {L})", fontsize=7.5)
        ax.margins(x=0.01)
        ax.grid(axis="y", alpha=0.25, linewidth=0.4)
        ax.legend(frameon=False, ncol=2, loc="lower center",
                  bbox_to_anchor=(0.5, 1.01), borderaxespad=0.0)
        fig.savefig(out / f"tokens_{m}.pdf", bbox_inches="tight")
        fig.savefig(out / f"tokens_{m}.png", bbox_inches="tight")
        plt.close(fig)

        sn = [a for _, _, a, c, _ in rows if c == "tab:red"]
        cm = [a for _, _, a, c, _ in rows if c == "tab:blue"]
        top = max(rows, key=lambda r: r[2])
        print(f"\n■ {LABEL[m]}  (L{L}, 블록 0)")
        print(f"   snake 12칸 합 {sum(sn):.4f}  ·  camel 12칸 합 {sum(cm):.4f}")
        print(f"   가장 많이 본 토큰: {top[1]!r} (칸 {top[0]}, {top[2]:.4f})")
        print("   상위 5개: " + " · ".join(
            f"{t.strip()} {a:.4f}" for _, t, a, _, _ in sorted(rows, key=lambda r: -r[2])[:5]))

    print(f"\n그림 → {out}/tokens_<모델>.png")
    print("막대 하나 = 토큰 하나가 받은 주의의 비중. 24개를 다 더하면 구간 합 두 개가 된다.")

    # ── ② 자리 교란 검산 — 1번 자리를 빼면 격차가 남는가 ────────────────
    print("\n" + "=" * 84)
    print("② 자리 교란 — 42묶음이 전부 같은 배치(S C C S S S S C C C C S)라 위반이 항상 1번이다")
    print("=" * 84)
    print(f"{'모델':<11}{'전체 위반':>10}{'전체 준수':>10}{'차이':>9}"
          f"{'│ 1번 뺀 위반':>14}{'준수':>9}{'차이':>9}")
    for m in MODELS:
        L = str(PEAK[m])
        pos = defaultdict(list)
        for p in sorted(Path("results", STEP).glob("*.json")):
            r = json.loads(p.read_text(encoding="utf-8"))
            if r["condition"]["model"]["family"] != m:
                continue
            td = r["metrics"]["extra"].get("token_detail") or {}
            rows = []
            for span in ("code_snake", "code_camel"):
                d = td.get(span)
                if not d or L not in d["per_layer"]:
                    continue
                a = d["per_layer"][L]["a"]
                for gi in _name_groups(d):
                    rows.append((d["tokens"][gi[0]]["t"], sum(a[i] for i in gi), span))
            if not rows:
                continue
            rows.sort()
            for k, (_, v, span) in enumerate(rows, 1):
                pos[k].append((v, span))
        if not pos:
            continue
        per = {k: (st.mean([a for a, _ in v]), v[0][1]) for k, v in pos.items()}
        S = [v for k, (v, sp) in per.items() if sp == "code_snake"]
        C = [v for k, (v, sp) in per.items() if sp == "code_camel"]
        S2 = [v for k, (v, sp) in per.items() if sp == "code_snake" and k != 1]
        print(f"{m:<11}{sum(S):>10.4f}{sum(C):>10.4f}{sum(S) - sum(C):>9.4f}"
              f"{st.mean(S2):>14.4f}{st.mean(C):>9.4f}{st.mean(S2) - st.mean(C):>9.4f}")
    print("\n오른쪽 세 칸은 **이름당 평균**이다(개수가 5 대 6이라 합으로 못 비교한다).")
    print("1번 자리를 빼면 차이가 0 근처로 사라지면, 격차는 표기가 아니라 자리에서 온 것이다.")


if __name__ == "__main__":
    main()
