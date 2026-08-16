"""구조적 결함 감사 — 결과 JSON 재집계 스크립트.

`docs/결함검증_재실험판정.md`의 모든 수치를 이 스크립트 하나로 재생성한다.
결과는 읽기만 하고 아무것도 쓰지 않는다(§6 불변).

쓰는 법:
    python scripts/audit_recheck.py <step3 결과 루트> <step5 결과 루트>

    # 예: 브랜치별로 흩어져 있으므로 git archive로 풀어서 준다
    mkdir -p /tmp/a3 /tmp/a5
    git archive origin/step3/code-cause results | tar -x -C /tmp/a3
    git archive origin/step5/instr-cause results | tar -x -C /tmp/a5
    python scripts/audit_recheck.py /tmp/a3 /tmp/a5

결과 폴더 이름이 규약(results/<step>/)과 다르게 흩어져 있어도(구조적 결함 B2/D1)
하위 폴더를 전부 훑으므로 그대로 동작한다.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

KINDS = ("key", "value", "key_value")


def load_all(root: str | Path) -> list[dict[str, Any]]:
    """루트 아래 모든 결과 JSON을 읽는다(폴더 레이아웃 무관)."""
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(Path(root).rglob("*.json"))]


def by_layer(records: list[dict]) -> dict:
    """모델 → 공여종류 → 층 → 개입종류 → [회복률...] 로 접는다."""
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for r in records:
        model = r["condition"]["model"]["family"]
        donor = r["metrics"]["extra"].get("donor", "?")
        for layer, vals in r["metrics"]["per_layer"].items():
            for kind in KINDS:
                v = vals.get(f"{kind}__recovery")
                if v is not None:
                    out[model][donor][int(layer)][kind].append(v)
    return out


def peak_layer(cube: dict, model: str, donor: str = "compliant") -> int:
    """그 모델의 '내용만(Value)' 평균 곡선이 가장 높은 층."""
    curves = cube[model][donor]
    return max(curves, key=lambda L: st.mean(curves[L]["value"]))


def mean_at(cube: dict, model: str, donor: str, layer: int, kind: str) -> float:
    return st.mean(cube[model][donor][layer][kind])


def report_step3(root: str | Path) -> None:
    cube = by_layer(load_all(root))
    print("\n" + "=" * 78)
    print("step3(코드 인과) — 통제를 빼기 전과 뺀 뒤")
    print("=" * 78)
    print(f"{'모델':<11}{'개입':<11}{'같은camel':>10}{'다른camel':>10}{'다른snake':>10}"
          f"{'순효과(짝맞춤)':>15}")
    print("-" * 78)
    for model in sorted(cube):
        L = peak_layer(cube, model)
        for kind in ("value", "key", "key_value"):
            same = mean_at(cube, model, "compliant", L, kind)
            uc = mean_at(cube, model, "unrelated_camel", L, kind)
            us = mean_at(cube, model, "unrelated_snake", L, kind)
            tail = f"   (L{L})" if kind == "value" else ""
            print(f"{model:<11}{kind:<11}{same:>10.3f}{uc:>10.3f}{us:>10.3f}{uc - us:>15.3f}{tail}")
        print()
    print("순효과 = 다른camel − 다른snake. 두 공여는 문맥 구성(6함수·단일표기)이 같아")
    print("차이가 오직 표기에서만 온다 — 문맥 길이 교란이 소거된 짝맞춤 비교.")

    print("\n" + "-" * 78)
    print("편향 점검 ① 봉우리 층을 Value로 골랐는데 Key가 손해를 보나")
    print("-" * 78)
    print(f"{'모델':<11}{'Value 봉우리':>13}{'Key 봉우리':>11}{'Key@V봉':>10}{'Key@K봉':>10}")
    for model in sorted(cube):
        curves = cube[model]["compliant"]
        lv = peak_layer(cube, model)
        lk = max(curves, key=lambda L: st.mean(curves[L]["key"]))
        print(f"{model:<11}{lv:>13}{lk:>11}"
              f"{st.mean(curves[lv]['key']):>10.3f}{st.mean(curves[lk]['key']):>10.3f}")

    print("\n" + "-" * 78)
    print("편향 점검 ② 층 편집→복구 루프가 누적 오염을 남기나 (양 끝이 0이어야 정상)")
    print("-" * 78)
    print(f"{'모델':<11}{'층수':>6}{'L0 value':>11}{'마지막층 value':>16}{'마지막층 K+V':>15}")
    for model in sorted(cube):
        curves = cube[model]["compliant"]
        last = max(curves)
        print(f"{model:<11}{len(curves):>6}{st.mean(curves[0]['value']):>11.4f}"
              f"{st.mean(curves[last]['value']):>16.4f}{st.mean(curves[last]['key_value']):>15.4f}")


def report_step5(root: str | Path) -> None:
    records = load_all(root)
    cube = by_layer(records)
    gaps: dict[str, list[float]] = defaultdict(list)
    dirs: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for r in records:
        model = r["condition"]["model"]["family"]
        tgt = r["condition"]["instruction"]["target_notation"]
        ex = r["metrics"]["extra"]
        gaps[model].append(abs(ex["S_clean"] - ex["S_base"]))
        for layer, vals in r["metrics"]["per_layer"].items():
            v = vals.get("value__recovery")
            if v is not None:
                dirs[model][tgt][int(layer)].append(v)

    print("\n" + "=" * 78)
    print("step5(지침 인과) — 방향 비대칭과 판정불가 임계")
    print("=" * 78)
    print(f"{'모델':<11}{'봉우리':>7}{'camel방향':>10}{'snake방향':>10}{'전이>1':>8}"
          f"{'|gap|중앙':>11}{'gap<1 개수':>12}{'Key/Value':>11}")
    for model in sorted(dirs):
        merged: dict[int, list[float]] = defaultdict(list)
        for tgt in dirs[model]:
            for layer, vs in dirs[model][tgt].items():
                merged[layer] += vs
        L = max(merged, key=lambda x: st.mean(merged[x]))
        c = st.mean(dirs[model]["camel"][L])
        s = st.mean(dirs[model]["snake"][L])
        over = sum(1 for x in merged[L] if x > 1)
        donor = next(iter(cube[model]))          # step5는 공여가 한 종류뿐
        kv = mean_at(cube, model, donor, L, "key") / mean_at(cube, model, donor, L, "value")
        print(f"{model:<11}{L:>7}{c:>10.2f}{s:>10.2f}{over:>8}"
              f"{st.median(gaps[model]):>11.2f}{sum(1 for g in gaps[model] if g < 1.0):>12}{kv:>11.2f}")
    print("\nstep5에는 통제 공여가 한 종류(opposite)뿐이라 step3처럼 순효과를 뺄 수 없다.")


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    report_step3(sys.argv[1])
    report_step5(sys.argv[2])


if __name__ == "__main__":
    main()
