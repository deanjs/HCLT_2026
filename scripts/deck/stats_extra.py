# -*- coding: utf-8 -*-
"""stats.json에 **나중에 생긴 실험**의 집계를 덧붙인다.

`stats.py`는 초판 덱을 만들 때 쓴 것이라 아래 셋이 빠져 있다.

    step2_pos   자리 통제(시드 42 ↔ 67) — 시드별·두 시드 평균, 지표 3종
    step5       지침 인과 — 순효과(처치 − 자기통제), 전 층 스윕 실행분
    step6_gen2  처방 생성 검증 — **on_task 판정을 거친** 진짜 준수율, 조건별

수치의 출처는 `results/` 하나뿐이다(§6). 손으로 적은 값은 없다.

    python stats.py && python stats_extra.py
"""
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

R = Path("../../results")
MODELS = ["qwen", "deepseek", "llama", "stability"]
BASE_SEED, FLIP_SEED = 42, 67
METRICS = {"attention_weight": True, "av_norm": True, "v_norm": False}
TASK_WORDS = ("remove", "duplicat", "dedup", "uniq", "distinct")
GAP_MIN = 1.0


def ci(xs):
    if not xs:
        return float("nan"), float("nan")
    if len(xs) < 2:
        return xs[0], 0.0
    return st.mean(xs), 1.96 * st.stdev(xs) / math.sqrt(len(xs))


def rd(x, n=5):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, n)


# ══════════════════════════════════════════════ step2 자리 통제 ══════════
def step2_pos():
    box = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for p in (R / "step2_code-observe").glob("*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        c = r["condition"]
        sd, m, b = c.get("seed"), c["model"]["family"], c["preceding"]["pool_block"]
        cnt = r["metrics"]["extra"]["span_token_counts"]
        for L, v in r["metrics"]["per_layer"].items():
            for met, per_tok in METRICS.items():
                d = {}
                for sp in ("code_snake", "code_camel"):
                    x = v.get(f"{sp}__{met}")
                    if x is not None:
                        d[sp] = x / cnt[sp] if per_tok else x
                if len(d) == 2:
                    box[met][m][sd][int(L)][b] = d["code_snake"] - d["code_camel"]

    out = {}
    for met in METRICS:
        out[met] = {}
        for m in MODELS:
            C = box[met][m]
            if BASE_SEED not in C or FLIP_SEED not in C:
                continue
            # 봉우리 층은 **시드 42 기준** — 기존 보고와 같은 층에서 견주려는 것이다.
            Lp = max(C[BASE_SEED], key=lambda L: st.mean(list(C[BASE_SEED][L].values())))
            ks = sorted(set(C[BASE_SEED][Lp]) & set(C[FLIP_SEED][Lp]))
            pair = [(C[BASE_SEED][Lp][k] + C[FLIP_SEED][Lp][k]) / 2 for k in ks]
            mu, e = ci(pair)
            a42 = st.mean(list(C[BASE_SEED][Lp].values()))
            a67 = st.mean(list(C[FLIP_SEED][Lp].values()))
            # 전 층 점검 — 시드42로 층을 고른 편향을 걷어낸다
            Ls = sorted(set(C[BASE_SEED]) & set(C[FLIP_SEED]))
            pos = neg = 0
            for L in Ls:
                kk = set(C[BASE_SEED][L]) & set(C[FLIP_SEED][L])
                a, b_ = ci([(C[BASE_SEED][L][k] + C[FLIP_SEED][L][k]) / 2 for k in kk])
                pos += a - b_ > 0
                neg += a + b_ < 0
            out[met][m] = {
                "peak": Lp, "s42": rd(a42), "s67": rd(a67),
                "mean": rd(mu), "ci": rd(e), "n": len(ks),
                "crosses0": bool(mu - e <= 0 <= mu + e),
                "kept_pct": rd(mu / a42 * 100, 1) if a42 else None,
                "n_layers": len(Ls), "sig_pos": pos, "sig_neg": neg,
                "n_zero": len(Ls) - pos - neg,
            }
    return out


# ═══════════════════════════════════════════════════ step5 지침 인과 ═════
def step5():
    """처치(opposite) − 자기통제(self), 같은 (모델·묶음·방향)끼리 짝지어 뺀다."""
    cube = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    folders = ["step5_instr-cause"] + [d.name for d in R.glob("step5_control_sweep_*")]
    for folder in folders:
        for p in (R / folder).glob("*.json"):
            r = json.loads(p.read_text(encoding="utf-8"))
            ex = r["metrics"]["extra"]
            if ex.get("mode") != "intervene_sweep":
                continue
            gap = ex.get("gap")
            if gap is None:
                gap = abs(ex["S_clean"] - ex["S_base"])
            if gap < GAP_MIN:                     # 판정 불가는 뺀다
                continue
            c = r["condition"]
            key = (c["preceding"]["pool_block"], c["instruction"]["target_notation"])
            m, d = c["model"]["family"], ex["donor"]
            for L, vals in r["metrics"]["per_layer"].items():
                for kind in ("value", "key"):
                    x = vals.get(f"{kind}__recovery")
                    if x is not None:
                        cube[m][d][kind][int(L)][key] = x

    out = {}
    T, S = "opposite_instruction", "control_self"
    for m in MODELS:
        if not cube[m][T]["value"] or not cube[m][S]["value"]:
            continue
        common = sorted(set(cube[m][T]["value"]) & set(cube[m][S]["value"]))

        def net(L, kind):
            tk, ck = cube[m][T][kind][L], cube[m][S][kind][L]
            kk = set(tk) & set(ck)
            return [tk[k] - ck[k] for k in kk]

        Lp = max(common, key=lambda L: st.mean(net(L, "value")) if net(L, "value") else -9)
        nv, nk = net(Lp, "value"), net(Lp, "key")
        mv, ev = ci(nv)
        mk, ek = ci(nk)
        out[m] = {
            "peak": Lp, "n_layers": len(common), "n": len(nv),
            "value": [rd(mv, 4), rd(ev, 4)], "key": [rd(mk, 4), rd(ek, 4)],
            "raw": rd(st.mean(list(cube[m][T]["value"][Lp].values())), 4),
            "ctrl": rd(st.mean(list(cube[m][S]["value"][Lp].values())), 4),
            "depth": rd(Lp / (max(common) + 1) * 100, 1),
        }
    return out


# ═══════════════════════════════════════ step6 생성 — on_task 판정 ═══════
def on_task(name):
    return bool(name) and any(w in (name or "").lower() for w in TASK_WORDS)


def step6_gen2():
    """조건별 **진짜 준수율**(표기 맞음 × 과제 맞음)과 과제 적중률."""
    box = defaultdict(lambda: defaultdict(list))
    for p in (R / "step6_steer-generate").glob("*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        ex = r["metrics"]["extra"]
        m = r["condition"]["model"]["family"]
        meth = ex.get("method", "none")
        if meth == "value_add":
            key = f"value{int(ex['strength'])}"
        elif meth == "attn_amplify":
            key = f"SL_{ex.get('span','?')[:4]}_{ex.get('psi_target')}"
        else:
            key = "none"
        for g in ex.get("generations", [ex]):
            nm = g.get("name")
            box[m][key].append((bool(g.get("compliant")) and on_task(nm), on_task(nm), nm))

    out = {}
    for m, conds in box.items():
        out[m] = {}
        for k, rows in conds.items():
            n = len(rows)
            out[m][k] = {"real": rd(sum(a for a, _, _ in rows) / n, 3),
                         "ontask": rd(sum(b for _, b, _ in rows) / n, 3), "n": n}
        # 값 조향 최적 세기 · Spotlight 최고
        va = {k: v for k, v in out[m].items() if k.startswith("value")}
        sl = {k: v for k, v in out[m].items() if k.startswith("SL")}
        if va:
            out[m]["_best_value"] = max(va, key=lambda k: va[k]["real"])
        if sl:
            out[m]["_best_SL"] = max(sl, key=lambda k: sl[k]["real"])
    return out


if __name__ == "__main__":
    S = json.load(open("stats.json", encoding="utf-8"))
    S["step2_pos"] = step2_pos()
    S["step5"] = step5()
    S["step6_gen2"] = step6_gen2()
    json.dump(S, open("stats.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("stats.json 갱신 — 키:", list(S))
    print("\n[step2_pos · 어텐션]")
    for m, v in S["step2_pos"]["attention_weight"].items():
        print(f"  {m:<10} L{v['peak']:<3} s42 {v['s42']:+.5f}  s67 {v['s67']:+.5f}  "
              f"평균 {v['mean']:+.5f}±{v['ci']:.5f}  0을무나={v['crosses0']}  남은 {v['kept_pct']}%")
    print("\n[step5 순효과]")
    for m, v in S["step5"].items():
        print(f"  {m:<10} L{v['peak']:<3}({v['depth']}%)  Value {v['value'][0]:+.4f}±{v['value'][1]:.4f}"
              f"  Key {v['key'][0]:+.4f}±{v['key'][1]:.4f}  n={v['n']}")
    print("\n[step6 생성 — 진짜 준수율]")
    for m, v in S["step6_gen2"].items():
        bv, bs = v.get("_best_value"), v.get("_best_SL")
        print(f"  {m:<10} 무개입 {v['none']['real']:.3f}  "
              f"값조향 최적 {bv}={v[bv]['real']:.3f}  Spotlight 최고 {bs}={v[bs]['real']:.3f}")
