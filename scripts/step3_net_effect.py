"""step3(코드 인과) 재집계 — 통제를 뺀 **표기 순효과**로 다시 계산한다.

왜 필요한가
-----------
step3는 공여(덮어넣는 값)를 3종으로 두고 통제를 세웠는데, 지금까지의 보고는 통제를 빼지 않은
**원값**이었다. 원값에는 "무엇을 덮든 생기는 교란"이 섞여 있다. 이 스크립트는 그 교란을 빼고
**표기(camel vs snake) 때문에 생긴 몫만** 남긴다.

무엇을 빼나 (두 가지 기준을 모두 낸다)
-----------------------------------
  짝맞춤(권장)  = 다른camel − 다른snake
      두 공여는 문맥 구성이 **완전히 같다**(둘 다 6함수·단일표기 별도 모듈).
      차이가 오직 표기에서만 오므로 문맥 길이 교란이 소거된다.
  보조          = 같은camel − 다른snake
      처치는 본래 조건(12함수 준수판)이지만 통제와 문맥 구성이 다르다.

통계
----
같은 이름 묶음(pool_block)에서 세 공여를 다 돌렸으므로 **묶음 단위로 짝을 지어** 차이를 먼저
계산한 뒤 42묶음에 대해 평균·신뢰구간을 낸다(짝지음 설계). 신뢰구간은 정규근사
(평균 ± 1.96·표준오차)다.

쓰는 법
-------
    mkdir -p /tmp/a3 && git archive origin/step3/code-cause results | tar -x -C /tmp/a3
    python scripts/step3_net_effect.py /tmp/a3 [--csv 출력폴더]

결과는 읽기만 한다(§6 불변).
"""

from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

KINDS = ("value", "key", "key_value")
KIND_LABEL = {"value": "내용만(Value)", "key": "어텐션만(Key)", "key_value": "둘다(K+V)"}
TREAT, FORM, NEG = "compliant", "unrelated_camel", "unrelated_snake"
GAP_MIN = 1.0          # |S_clean − S_base| 가 이보다 작으면 판정 불가


def load(root: str | Path) -> dict:
    """모델 → 공여 → 층 → 개입종류 → {묶음: 회복률} 로 접는다. gap도 함께 모은다."""
    cube: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    gaps: dict = defaultdict(dict)
    for p in sorted(Path(root).rglob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        model = r["condition"]["model"]["family"]
        block = r["condition"]["preceding"]["pool_block"]
        donor = r["metrics"]["extra"]["donor"]
        ex = r["metrics"]["extra"]
        gaps[model][(block, donor)] = abs(ex["S_clean"] - ex["S_base"])
        for layer, vals in r["metrics"]["per_layer"].items():
            for kind in KINDS:
                v = vals.get(f"{kind}__recovery")
                if v is not None:
                    cube[model][donor][int(layer)][kind][block] = v
    return cube, gaps


def paired(cube: dict, model: str, layer: int, kind: str, a: str, b: str) -> list[float]:
    """같은 묶음끼리 짝지어 (a − b) 차이 목록을 낸다."""
    xa = cube[model][a][layer][kind]
    xb = cube[model][b][layer][kind]
    return [xa[k] - xb[k] for k in sorted(set(xa) & set(xb))]


def ci95(xs: list[float]) -> tuple[float, float]:
    """평균과 95% 신뢰구간 반폭(정규근사)."""
    if len(xs) < 2:
        return (xs[0] if xs else float("nan")), float("nan")
    return st.mean(xs), 1.96 * st.stdev(xs) / math.sqrt(len(xs))


def net_curve(cube: dict, model: str, kind: str, a: str, b: str) -> dict[int, tuple[float, float]]:
    return {L: ci95(paired(cube, model, L, kind, a, b)) for L in sorted(cube[model][a])}


def peak_of(curve: dict[int, tuple[float, float]]) -> int:
    return max(curve, key=lambda L: curve[L][0])


def make_figure(cube: dict, model: str, out: Path) -> None:
    """층별 곡선 그림 — 왼쪽 원값, 오른쪽 순효과. 라벨은 영문(§6 글자깨짐 방지)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted(cube[model][TREAT])
    style = {"value": ("tab:blue", "Value only"),
             "key": ("tab:gray", "Key only"),
             "key_value": ("tab:orange", "Key + Value")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, mode in zip(axes, ("raw", "net")):
        for kind in KINDS:
            color, label = style[kind]
            if mode == "raw":
                pts = [ci95(list(cube[model][TREAT][L][kind].values())) for L in layers]
            else:
                pts = [ci95(paired(cube, model, L, kind, FORM, NEG)) for L in layers]
            mu = [m for m, _ in pts]
            lo = [m - c for m, c in pts]
            hi = [m + c for m, c in pts]
            ax.plot(layers, mu, color=color, label=label, linewidth=1.6)
            ax.fill_between(layers, lo, hi, color=color, alpha=0.2, linewidth=0)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xlabel("Intervened layer")
        ax.set_title("Raw recovery (no control)" if mode == "raw"
                     else "Notation-specific effect (control subtracted)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Recovery")
    axes[1].set_ylabel("Recovery difference")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle(f"step3 code causality — {model}"
                 "  (control = other-camel minus other-snake donor)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    root = sys.argv[1]
    csv_dir = fig_dir = None
    if "--csv" in sys.argv:
        csv_dir = Path(sys.argv[sys.argv.index("--csv") + 1])
        csv_dir.mkdir(parents=True, exist_ok=True)
    if "--figs" in sys.argv:
        fig_dir = Path(sys.argv[sys.argv.index("--figs") + 1])
        fig_dir.mkdir(parents=True, exist_ok=True)
    cube, gaps = load(root)

    print("=" * 92)
    print("step3 재집계 — 통제를 뺀 표기 순효과  (짝맞춤: 다른camel − 다른snake)")
    print("=" * 92)

    for model in sorted(cube):
        # 봉우리 층은 **순효과 곡선**에서 새로 고른다(원값 곡선이 아니라).
        vcurve = net_curve(cube, model, "value", FORM, NEG)
        Lnet = peak_of(vcurve)
        rawv = {L: ci95(list(cube[model][TREAT][L]["value"].values())) for L in sorted(cube[model][TREAT])}
        Lraw = peak_of(rawv)

        n_bad = sum(1 for (b, d), g in gaps[model].items() if g < GAP_MIN)
        print(f"\n■ {model}   순효과 봉우리 L{Lnet}"
              f"{'' if Lnet == Lraw else f'  (원값 봉우리는 L{Lraw} — 다름!)'}"
              f"   판정불가(|gap|<{GAP_MIN}) {n_bad}/{len(gaps[model])}개")
        print(f"  {'개입':<14}{'원값(같은camel)':>18}{'순효과(짝맞춤)':>20}{'순효과(보조)':>18}")
        for kind in KINDS:
            raw_m, raw_c = ci95(list(cube[model][TREAT][Lnet][kind].values()))
            net_m, net_c = ci95(paired(cube, model, Lnet, kind, FORM, NEG))
            alt_m, alt_c = ci95(paired(cube, model, Lnet, kind, TREAT, NEG))
            print(f"  {KIND_LABEL[kind]:<14}{raw_m:>11.3f}±{raw_c:<6.3f}"
                  f"{net_m:>13.3f}±{net_c:<6.3f}{alt_m:>11.3f}±{alt_c:<6.3f}")

        # 분할검증: 앞 21묶음으로 층 고르고 뒤 21묶음에서 값 확인
        blocks = sorted(cube[model][FORM][Lnet]["value"])
        half = len(blocks) // 2
        first, second = set(blocks[:half]), set(blocks[half:])
        def half_curve(sel):
            return {L: st.mean([cube[model][FORM][L]["value"][b] - cube[model][NEG][L]["value"][b]
                                for b in sel]) for L in sorted(cube[model][FORM])}
        L1 = max(half_curve(first), key=half_curve(first).get)
        print(f"  분할검증: 앞{half}묶음이 고른 층 L{L1} → 뒤{len(second)}묶음에서 값 "
              f"{half_curve(second)[L1]:.3f}   (전체 L{Lnet}에서 {vcurve[Lnet][0]:.3f})")

        if csv_dir:
            with (csv_dir / f"step3_net_{model}.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["layer", "kind", "net_mean", "net_ci95", "raw_mean", "raw_ci95"])
                for L in sorted(cube[model][FORM]):
                    for kind in KINDS:
                        nm, nc = ci95(paired(cube, model, L, kind, FORM, NEG))
                        rm, rc = ci95(list(cube[model][TREAT][L][kind].values()))
                        w.writerow([L, kind, f"{nm:.6f}", f"{nc:.6f}", f"{rm:.6f}", f"{rc:.6f}"])

    print("\n" + "=" * 92)
    print("요약: Key 순효과가 0을 물면 '어텐션 경로는 표기 정보를 나르지 않는다'")
    print("=" * 92)
    print(f"{'모델':<11}{'Value 순효과':>20}{'Key 순효과':>20}{'K+V 순효과':>20}{'0을 무나(Key)':>14}")
    for model in sorted(cube):
        L = peak_of(net_curve(cube, model, "value", FORM, NEG))
        row = []
        for kind in KINDS:
            m, c = ci95(paired(cube, model, L, kind, FORM, NEG))
            row.append((m, c))
        km, kc = row[1]
        crosses = "예" if (km - kc) <= 0 <= (km + kc) else "아니오"
        print(f"{model:<11}" + "".join(f"{m:>13.3f}±{c:<6.3f}" for m, c in row) + f"{crosses:>14}")
    if csv_dir:
        print(f"\n층별 곡선 CSV → {csv_dir}/step3_net_<모델>.csv")
    if fig_dir:
        for model in sorted(cube):
            make_figure(cube, model, fig_dir / f"net_effect_{model}.png")
        print(f"그림 → {fig_dir}/net_effect_<모델>.png")


if __name__ == "__main__":
    main()
