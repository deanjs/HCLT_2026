"""슬라이드에 박을 수치를 결과에서 직접 뽑는다. 손으로 적지 않는다(§6)."""
import json, math, statistics as st, collections
from pathlib import Path
R = Path("/home/user/HCLT_2026/results")
M = ["qwen", "deepseek", "llama", "stability"]
out = {}

def ci(xs):
    if len(xs) < 2: return (xs[0] if xs else float("nan")), float("nan")
    return st.mean(xs), 1.96*st.stdev(xs)/math.sqrt(len(xs))

# ---- step1 절벽
c1 = collections.defaultdict(list)
for p in (R/"step1_cliff").glob("*.json"):
    r = json.loads(p.read_text()); c = r["condition"]
    v = 12 - c["preceding"]["n_compliant"]
    c1[(c["model"]["family"], c["instruction"]["target_notation"], v)].append(r["metrics"]["compliance_rate"])
out["step1"] = {f"{m}|{d}|{v}": round(st.mean(x), 4) for (m, d, v), x in c1.items()}

# ---- step2 봉우리층·간극 (토큰당)
s2 = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
gqa = {}
for p in (R/"step2_code-observe").glob("*.json"):
    r = json.loads(p.read_text()); m = r["condition"]["model"]["family"]
    ex = r["metrics"]["extra"]; cnt = ex["span_token_counts"]; gqa[m] = ex["gqa"]
    for L, vals in r["metrics"]["per_layer"].items():
        for sp in ("code_camel", "code_snake"):
            x = vals.get(f"{sp}__attention_weight")
            if x is not None and cnt.get(sp): s2[m][int(L)][sp].append(x/cnt[sp])
out["gqa"] = gqa
out["step2"] = {}
for m in M:
    Ls = sorted(s2[m])
    gap = {L: st.mean(s2[m][L]["code_snake"]) - st.mean(s2[m][L]["code_camel"]) for L in Ls}
    Lp = max(gap, key=gap.get)
    out["step2"][m] = {"n_layers": len(Ls), "peak": Lp, "gap": round(gap[Lp], 5),
                       "snake": round(st.mean(s2[m][Lp]["code_snake"]), 5),
                       "camel": round(st.mean(s2[m][Lp]["code_camel"]), 5)}

# ---- step3 순효과
c3 = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
for p in (R/"step3_code-cause").glob("*.json"):
    r = json.loads(p.read_text()); c = r["condition"]; ex = r["metrics"]["extra"]
    for L, v in r["metrics"]["per_layer"].items():
        for k in ("value", "key", "key_value"):
            x = v.get(f"{k}__recovery")
            if x is not None: c3[c["model"]["family"]][ex["donor"]][k].setdefault(int(L), {})[c["preceding"]["pool_block"]] = x
out["step3"] = {}
for m in M:
    Ls = sorted(c3[m]["unrelated_camel"]["value"])
    net = {}
    for L in Ls:
        A, B = c3[m]["unrelated_camel"]["value"][L], c3[m]["unrelated_snake"]["value"][L]
        net[L] = st.mean([A[b]-B[b] for b in set(A) & set(B)])
    Lp = max(net, key=net.get); d = {"peak": Lp, "n_layers": len(Ls)}
    for k in ("value", "key", "key_value"):
        A, B = c3[m]["unrelated_camel"][k][Lp], c3[m]["unrelated_snake"][k][Lp]
        mu, h = ci([A[b]-B[b] for b in set(A) & set(B)])
        d[k] = [round(mu, 4), round(h, 4)]
    raw = st.mean(list(c3[m]["compliant"]["value"][Lp].values()))
    bub = st.mean(list(c3[m]["unrelated_snake"]["value"][Lp].values()))
    d["raw"], d["bubble"], d["bubble_pct"] = round(raw, 4), round(bub, 4), round(bub/raw, 3)
    out["step3"][m] = d

# ---- step4 지시어÷코드 비율
s4p = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
s4r = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
for p in (R/"step4_instr-observe").glob("*.json"):
    r = json.loads(p.read_text()); m = r["condition"]["model"]["family"]
    cnt = r["metrics"]["extra"]["span_token_counts"]
    for L, vals in r["metrics"]["per_layer"].items():
        for sp, n in cnt.items():
            x = vals.get(f"{sp}__attention_weight")
            if x is not None and n:
                s4p[m][int(L)][sp].append(x/n); s4r[m][int(L)][sp].append(x)
out["step4"] = {}
for m in M:
    Ls = sorted(s4p[m])
    above_p = sum(1 for L in Ls if st.mean(s4p[m][L]["instr_rule_word"]) > st.mean(s4p[m][L]["code_camel"]))
    above_r = sum(1 for L in Ls if st.mean(s4r[m][L]["instr_rule_word"]) > st.mean(s4r[m][L]["code_camel"]))
    Lp = max(Ls, key=lambda L: st.mean(s4p[m][L]["instr_rule_word"]))
    out["step4"][m] = {"n_layers": len(Ls), "peak": Lp,
                       "above_per_token": above_p, "above_sum": above_r,
                       "n_spans": len(s4p[m][Ls[0]])}

# ---- diag 노름 수축
dg = collections.defaultdict(lambda: collections.defaultdict(list))
poff = collections.defaultdict(list)
for p in (R/"diag_kv-phase").glob("*.json"):
    r = json.loads(p.read_text()); m = r["condition"]["model"]["family"]
    poff[m].append(r["metrics"]["extra"]["position_offset_abs_mean"])
    for L, v in r["metrics"]["per_layer"].items():
        for k in ("key_shrink", "value_shrink", "key_minus_value"):
            if v.get(k) is not None: dg[m][k].append(v[k])
out["diag"] = {m: {"key_shrink": round(st.mean(dg[m]["key_shrink"]), 3),
                   "value_shrink": round(st.mean(dg[m]["value_shrink"]), 3),
                   "key_minus_value": round(st.mean(dg[m]["key_minus_value"]), 3),
                   "pos_offset": round(st.mean(poff[m]), 2)} for m in M if dg[m]}

# ---- step6 조향 vs Spotlight
s6 = collections.defaultdict(lambda: collections.defaultdict(list))
for p in (R/"step6_steer").glob("*.json"):
    r = json.loads(p.read_text()); ex = r["metrics"]["extra"]
    if ex.get("undecidable"): continue
    m = r["condition"]["model"]["family"]
    s6[m][(ex["kind"], ex.get("strength"), ex.get("psi_target"))].append(ex["recovery"])
out["step6"] = {}
for m in M:
    va = {k[1]: st.mean(v) for k, v in s6[m].items() if k[0] == "value_add" and k[1] is not None}
    sp = {k[2]: st.mean(v) for k, v in s6[m].items() if k[0] == "spotlight" and k[2] is not None}
    out["step6"][m] = {"value_add_best": [max(va, key=va.get), round(max(va.values()), 3)] if va else None,
                       "spotlight_best": [max(sp, key=sp.get), round(max(sp.values()), 3)] if sp else None,
                       "n_strength": len(va), "n_psi": len(sp)}
# 이름 건강도
hm = collections.defaultdict(lambda: collections.defaultdict(list))
for p in (R/"step6_steer-generate").glob("*.json"):
    r = json.loads(p.read_text()); ex = r["metrics"]["extra"]; m = r["condition"]["model"]["family"]
    hm[m][ex["method"]].append((bool(ex.get("name_ok")), bool(ex.get("compliant"))))
out["step6_gen"] = {m: {meth: [round(sum(a for a, _ in v)/len(v), 3), round(sum(b for _, b in v)/len(v), 3), len(v)]
                        for meth, v in d.items()} for m, d in hm.items()}

# ---- 파일 수 인구조사
out["census"] = {d.name: len(list(d.glob("*.json"))) for d in sorted(R.iterdir()) if d.is_dir()}
Path("stats.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({k: out[k] for k in ("gqa","step2","step3","step4","diag","step6","step6_gen","census")}, indent=1, ensure_ascii=False))
