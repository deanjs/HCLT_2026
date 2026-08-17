"""step5(지침 인과) 순효과 — **짝지음 신뢰구간**까지 낸다.

왜 필요한가
-----------
지금까지 step5 보고는 처치와 통제의 평균을 **따로** 내고 그 차이만 적었다.

    Key 처치 0.106±0.005 · 자기통제 0.107±0.004 → 순효과 −0.001   ← 차이에 신뢰구간이 없다

그래서 "순효과가 0이다"를 **통계적으로 주장할 수 없었다.** 처치와 통제는 같은 42묶음·
같은 지침 방향에서 돌았으므로 **묶음 단위로 짝을 지어** 차이를 먼저 계산할 수 있다.
그러면 묶음마다 다른 난이도가 상쇄되어 신뢰구간이 훨씬 좁아지고, 0 포함 여부를 말할 수 있다.

step3(`step3_net_effect.py`)는 이미 이 방식이다. 이 스크립트가 step5를 같은 규격으로 맞춘다.

무엇을 빼나
-----------
    순효과 = 처치(반대 지침 공여) − 통제

  통제 두 종류를 **모두** 낸다.
    self           : 같은 지시어를 평균 내 덮음 → **평균 풀링 + 덮어쓰기 교란**을 잡는다
                     (항등 개입이 아니다. token_unit="mean"이라 (k₁,k₂)→(평균,평균))
    unrelated_word : 지침 안의 무관한 단어("project")를 덮음 → 음성 통제
                     (표기어와 의미·토큰 수·위치가 달라 완전한 음성 통제는 아니다)

짝짓기 키
---------
    (모델, 이름 묶음 pool_block, 지침 방향 target_notation)
  처치 336개 = 4모델 × 방향 2 × 묶음 42. 통제 스윕도 같은 336개라 **1:1로 맞물린다.**

통계
----
묶음마다 차이를 먼저 구하고, 그 차이들의 평균 ± 1.96·표준오차(정규근사).
**어느 한쪽이라도 판정 불가(|S_반대 − S_기준| < 1.0)면 그 짝은 뺀다.**

쓰는 법
-------
    python scripts/step5_net_effect.py [처치폴더] [통제폴더] [--gap-min 1.0]

  통제 폴더를 생략하면 **전 층 스윕 폴더가 있으면 그것을**, 없으면 기존 단일 층 폴더를 쓴다.
  어느 쪽을 썼는지는 출력 첫머리에 찍힌다. **두 폴더를 섞어 읽으면 예외를 던진다** —
  짝짓기 키에 층이 없어 같은 조건이 조용히 덮이기 때문이다.
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

KINDS = ("value", "key", "key_value")
KIND_LABEL = {"value": "내용만(Value)", "key": "어텐션만(Key)", "key_value": "둘다(K+V)"}
MODELS = ["qwen", "deepseek", "llama", "stability"]
CONTROLS = ("control_self", "control_unrelated_word")
CTRL_LABEL = {"control_self": "자기 통제(self)", "control_unrelated_word": "음성 통제(무관어)"}
DEFAULT_GAP_MIN = 1.0

TREAT_DIR = "results/step5_instr-cause"
# 통제는 폴더가 둘이다. 새 폴더(전 층 스윕)가 있으면 그쪽을 쓴다 — 어느 쪽을 썼는지 반드시 찍는다.
CTRL_DIR_1LAYER = "results/step5_instr-cause-control"        # 봉우리 층 한 곳 (구)
CTRL_DIR_SWEEP = "results/step5_instr-cause-control-sweep"   # 전 층 (신)


def default_ctrl_dirs() -> list[Path]:
    """통제 폴더를 고른다. 전 층 스윕이 있으면 그쪽, 없으면 기존 단일 층.

    전 층 스윕은 한 폴더(`…-control-sweep/`)일 수도 있고, 모델별로 갈려
    저장됐을 수도 있다(`step5_control_sweep_<모델>/`). 둘 다 받는다.
    **단일 층 폴더와는 절대 섞지 않는다** — 짝짓기 키에 층이 없어 조용히 덮인다.
    """
    one = Path(CTRL_DIR_SWEEP)
    if one.is_dir() and any(one.glob("*.json")):
        return [one]
    split = sorted(d for d in Path("results").glob("step5_control_sweep_*")
                   if d.is_dir() and any(d.glob("*.json")))
    return split or [Path(CTRL_DIR_1LAYER)]


def load_sweeps(root: Path) -> dict:
    """스윕 결과를 (모델, 묶음, 방향, 공여) → {층: {방식: 전이율}, 'gap': …} 로 접는다.

    통제 폴더에는 단일 층 실행분도 섞여 있다. 처치와 층 구조를 맞추기 위해
    **스윕만** 쓴다(mode == 'intervene_sweep').
    """
    out: dict = {}
    for p in sorted(Path(root).rglob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        ex = r["metrics"]["extra"]
        if ex.get("mode") != "intervene_sweep":
            continue
        cond = r["condition"]
        key = (cond["model"]["family"], cond["preceding"]["pool_block"],
               cond["instruction"]["target_notation"], ex.get("donor"))
        per_layer = {int(L): v for L, v in r["metrics"]["per_layer"].items()}
        # 처치분에는 gap 필드가 없다(그 뒤에 추가됨) → 원점수로 직접 계산한다.
        gap = ex.get("gap")
        if gap is None:
            gap = abs(ex["S_clean"] - ex["S_base"])
        if key in out:
            # 조용한 덮어쓰기 금지(§7). 키에 층이 없으므로, 같은 조건을 다른 층 구성으로
            # 돌린 실행분이 한 폴더에 섞이면 뒤에 읽힌 쪽이 앞을 지운다 — 결과를 보고도
            # 어느 쪽 숫자인지 알 수 없게 된다.
            prev = sorted(out[key]["per_layer"])
            now = sorted(per_layer)
            raise SystemExit(
                f"같은 조건이 두 번 있다: {key}\n"
                f"  이미 읽은 층 {len(prev)}개 {prev[:3]}… · 새로 읽은 층 {len(now)}개 {now[:3]}…\n"
                f"  파일: {p}\n"
                f"  통제 폴더를 섞어 읽고 있다. 한 폴더만 지정한다"
                f"({CTRL_DIR_1LAYER} 또는 {CTRL_DIR_SWEEP})."
            )
        out[key] = {"per_layer": per_layer, "gap": gap}
    return out


def ci95(xs: list[float]) -> tuple[float, float, int]:
    """평균, 95% 신뢰반폭, n."""
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return xs[0], float("nan"), 1
    return st.mean(xs), 1.96 * st.stdev(xs) / math.sqrt(n), n


def paired(treat: dict, ctrl: dict, model: str, donor: str, layer: int, kind: str,
           gap_min: float, direction: str | None = None) -> list[float]:
    """묶음마다 (처치 − 통제)를 먼저 계산해 목록으로 돌려준다."""
    diffs: list[float] = []
    for (m, block, tgt, d), t in treat.items():
        if m != model or d != "opposite_instruction":
            continue
        if direction is not None and tgt != direction:
            continue
        c = ctrl.get((m, block, tgt, donor))
        if c is None:
            continue
        if t["gap"] < gap_min or c["gap"] < gap_min:      # 어느 한쪽이라도 판정 불가면 제외
            continue
        tv = t["per_layer"].get(layer, {}).get(f"{kind}__recovery")
        cv = c["per_layer"].get(layer, {}).get(f"{kind}__recovery")
        if tv is None or cv is None:
            continue
        diffs.append(tv - cv)
    return diffs


def peak_layer(treat: dict, ctrl: dict, model: str, gap_min: float) -> int | None:
    """Value 순효과(자기 통제 기준)가 가장 큰 층."""
    layers = sorted({L for (m, _b, _t, d), v in treat.items() if m == model
                     for L in v["per_layer"]})
    best, best_v = None, float("-inf")
    for L in layers:
        xs = paired(treat, ctrl, model, "control_self", L, "value", gap_min)
        if not xs:
            continue
        v = st.mean(xs)
        if v > best_v:
            best, best_v = L, v
    return best


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    gap_min = DEFAULT_GAP_MIN
    if "--gap-min" in sys.argv:
        gap_min = float(sys.argv[sys.argv.index("--gap-min") + 1])
    treat_dir = Path(args[0]) if args else Path(TREAT_DIR)
    ctrl_dirs = [Path(a) for a in args[1:]] or default_ctrl_dirs()

    treat = load_sweeps(treat_dir)
    ctrl = {}
    for d in ctrl_dirs:
        part = load_sweeps(d)
        dup = set(part) & set(ctrl)
        if dup:
            raise SystemExit(f"통제 폴더끼리 조건이 겹친다: {sorted(dup)[:3]}…")
        ctrl.update(part)
    if not treat or not ctrl:
        raise SystemExit(f"결과를 못 찾았다: 처치 {len(treat)}개 · 통제 {len(ctrl)}개")

    print("=" * 100)
    print("step5 순효과 — 묶음끼리 짝지어 뺀 뒤 신뢰구간을 낸다 (판정 불가 기준 "
          f"|S_반대 − S_기준| < {gap_min:g} 제외)")
    print("=" * 100)
    n_ctrl_layers = {len(v["per_layer"]) for v in ctrl.values()}
    print(f"처치 {treat_dir}  {len(treat)}개")
    print(f"통제 {' + '.join(str(d) for d in ctrl_dirs)}  {len(ctrl)}개 "
          f"· 조건당 층 {sorted(n_ctrl_layers)}")
    if max(n_ctrl_layers, default=0) <= 1:
        print("  ⚠️ 통제가 층 하나뿐이다 — 층별 순효과 곡선은 만들 수 없고,"
              " 아래 봉우리 층은 '선택'이 아니라 '유일한 후보'다.")
    print()

    summary: dict = {}
    for model in MODELS:
        L = peak_layer(treat, ctrl, model, gap_min)
        if L is None:
            print(f"■ {model}: 짝지을 수 있는 조건이 없다 (전부 판정 불가)\n")
            continue
        print(f"■ {model}   봉우리 층 L{L} (Value 순효과 기준, 자기 통제)")
        print(f"  {'방식':<14}{'통제':<18}{'처치 평균':>11}{'통제 평균':>11}"
              f"{'순효과(짝지음)':>18}{'n':>5}{'0을 무나':>9}")
        for kind in KINDS:
            for donor in CONTROLS:
                diffs = paired(treat, ctrl, model, donor, L, kind, gap_min)
                if not diffs:
                    continue
                # 참고용 원값 평균(짝지음 아님)
                tvals = [t["per_layer"].get(L, {}).get(f"{kind}__recovery")
                         for (m, _b, _t, d), t in treat.items()
                         if m == model and d == "opposite_instruction" and t["gap"] >= gap_min]
                cvals = [c["per_layer"].get(L, {}).get(f"{kind}__recovery")
                         for (m, _b, _t, d), c in ctrl.items()
                         if m == model and d == donor and c["gap"] >= gap_min]
                tvals = [x for x in tvals if x is not None]
                cvals = [x for x in cvals if x is not None]
                mean, half, n = ci95(diffs)
                zero = "예" if (mean - half) <= 0 <= (mean + half) else "**아니오**"
                print(f"  {KIND_LABEL[kind]:<14}{CTRL_LABEL[donor]:<18}"
                      f"{st.mean(tvals):>11.3f}{st.mean(cvals):>11.3f}"
                      f"{mean:>10.3f}±{half:<7.3f}{n:>5}{zero:>9}")
                summary[(model, kind, donor)] = (mean, half, n)
        # 방향별 (Value·Key, 자기 통제)
        print(f"  {'— 방향별 (자기 통제)':<32}{'camel 지침':>16}{'snake 지침':>16}")
        for kind in ("value", "key"):
            cells = []
            for d in ("camel", "snake"):
                xs = paired(treat, ctrl, model, "control_self", L, kind, gap_min, direction=d)
                m_, h_, n_ = ci95(xs)
                cells.append(f"{m_:.3f}±{h_:.3f}" if n_ > 1 else "—")
            print(f"  {KIND_LABEL[kind]:<32}{cells[0]:>16}{cells[1]:>16}")
        print()

    print("=" * 100)
    print("한 줄 요약 — 순효과의 신뢰구간이 0을 물면 '그 경로로는 아무것도 안 옮겨진다'를 말할 수 있다.")
    print("=" * 100)
    print(f"{'모델':<12}{'Value 순효과':>20}{'Key 순효과':>20}{'Key가 0을 무나':>16}")
    for model in MODELS:
        v = summary.get((model, "value", "control_self"))
        k = summary.get((model, "key", "control_self"))
        if not v or not k:
            continue
        zero = "예" if (k[0] - k[1]) <= 0 <= (k[0] + k[1]) else "**아니오**"
        print(f"{model:<12}{f'{v[0]:.3f}±{v[1]:.3f}':>20}"
              f"{f'{k[0]:.3f}±{k[1]:.3f}':>20}{zero:>16}")
    print()
    print("주의 — 자기 통제는 **항등 개입이 아니다**. token_unit='mean'이라 같은 단어의 조각들을")
    print("       평균 낸 뒤 덮으므로, 잡아내는 것은 '평균 풀링 + 덮어쓰기 교란'이다.")
    print("       따라서 Key 순효과 ≈ 0이 뜻하는 것은 '반대 표기 공여가 그 교란을 넘는 추가 효과를")
    print("       만들지 못했다'이지, 'native Key 경로에 표기 정보가 없다'가 아니다.")


if __name__ == "__main__":
    main()
