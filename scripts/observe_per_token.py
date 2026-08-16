"""step2·step4 관측 재집계 — 구간 지표를 **토큰 수로 나눠** 공정하게 비교한다.

왜 필요한가
-----------
`attention_weight`와 `av_norm`은 구간 안 토큰들의 **합**이다. 그런데 비교하는 두 구간의
토큰 수가 다르다.
  · DeepSeek·StableCode는 밑줄을 따로 떼어(`' split','_','config'`) snake 구간 토큰이 훨씬 많다.
  · 지침의 요구 표기어는 규칙문과 후보열거에서 **2번** 나오고 반대 표기어는 **1번**만 나온다.
그래서 "snake를 더 본다" / "요구어를 더 본다"의 일부는 **토큰이 많아서**일 수 있다.

이 스크립트는 결과에 함께 저장된 `span_token_counts`로 **토큰당 평균**을 다시 계산해,
합으로 봤을 때와 봉우리 층·부호가 유지되는지 확인한다. 재실험은 필요 없다(§6 불변, 읽기만).

쓰는 법
-------
    python scripts/observe_per_token.py results/step2_code-observe
    python scripts/observe_per_token.py results/step4_instr-observe
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

METRICS = ("attention_weight", "av_norm")   # 둘 다 구간 '합' — 정규화 대상
# 스텝별로 "무엇과 무엇을 비교하는가"
PAIRS = {
    "step2": [("code_snake", "code_camel", "코드: snake − camel")],
    "step4": [("code_snake", "code_camel", "코드: snake − camel"),
              ("instr_rule_word", "code_camel", "지침 지시어 − 코드 camel"),
              ("instr_cand_camel", "instr_cand_snake", "후보열거: camel − snake"),
              ("instr_target_word", "instr_viol_word", "(초판 합산키) 요구어 − 반대어")],
}


def load(root: str | Path):
    rows = []
    for p in sorted(Path(root).rglob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        rows.append({
            "model": r["condition"]["model"]["family"],
            "counts": r["metrics"]["extra"].get("span_token_counts", {}),
            "per_layer": {int(L): v for L, v in r["metrics"]["per_layer"].items()},
        })
    return rows


def diff_curves(rows, model, a, b, metric):
    """층 → (합 기준 차이 목록, 토큰당 평균 기준 차이 목록)."""
    raw, per = defaultdict(list), defaultdict(list)
    for r in rows:
        if r["model"] != model:
            continue
        na, nb = r["counts"].get(a), r["counts"].get(b)
        for L, v in r["per_layer"].items():
            xa, xb = v.get(f"{a}__{metric}"), v.get(f"{b}__{metric}")
            if xa is None or xb is None:
                continue
            raw[L].append(xa - xb)
            if na and nb:
                per[L].append(xa / na - xb / nb)
    return raw, per


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    root = sys.argv[1]
    step = "step4" if "step4" in str(root) else "step2"
    rows = load(root)
    models = sorted({r["model"] for r in rows})

    print("=" * 100)
    print(f"{step} 관측 재집계 — 합 기준 vs 토큰당 평균 기준")
    print("=" * 100)

    # 구간별 평균 토큰 수 먼저
    print("\n[구간 평균 토큰 수]")
    keys = sorted({k for r in rows for k in r["counts"]})
    print(f"{'모델':<11}" + "".join(f"{k:>20}" for k in keys))
    for m in models:
        cs = [r["counts"] for r in rows if r["model"] == m]
        print(f"{m:<11}" + "".join(
            f"{st.mean([c[k] for c in cs if k in c]):>20.2f}" if any(k in c for c in cs)
            else f"{'-':>20}" for k in keys))

    for a, b, label in PAIRS[step]:
        print(f"\n[{label}]")
        print(f"{'모델':<11}{'지표':<18}{'합: 봉우리층':>14}{'합: 값':>12}"
              f"{'토큰당: 봉우리층':>18}{'토큰당: 값':>13}{'부호 유지':>10}")
        for m in models:
            for metric in METRICS:
                raw, per = diff_curves(rows, m, a, b, metric)
                if not raw or not per:
                    print(f"{m:<11}{metric:<18}{'구간 없음':>14}")
                    continue
                # 가장 격차가 큰 층(절댓값 기준)
                Lr = max(raw, key=lambda L: abs(st.mean(raw[L])))
                Lp = max(per, key=lambda L: abs(st.mean(per[L])))
                vr, vp = st.mean(raw[Lr]), st.mean(per[Lp])
                same = "예" if (vr > 0) == (vp > 0) else "**아니오**"
                print(f"{m:<11}{metric:<18}{f'L{Lr}':>14}{vr:>12.4f}"
                      f"{f'L{Lp}':>18}{vp:>13.4f}{same:>10}")

    print("\n부호가 유지되면 정성 결론은 그대로다. 봉우리 층이 옮겨가면 그 사실을 논문에 적어야 한다.")


if __name__ == "__main__":
    main()
