"""step6 집계 — 처방 비교표 · 이름 건전성 · 층 특이성.

세 실행분을 함께 읽는다.
  results/step6_steer/              본실험(선호 점수 회복)
  results/step6_steer-generate/     조향 하에서 실제 생성한 이름
  results/step6_steer-crosslayer/   맞는 층 방향을 엉뚱한 층에 주입

쓰는 법:
    python scripts/step6_summary.py [--figs docs/step6/figures]
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

MODELS = ["qwen", "deepseek", "llama", "stability"]
LABEL = {"qwen": "Qwen2.5-Coder-3B", "deepseek": "DeepSeek-Coder-6.7B",
         "llama": "Llama-3.2-3B", "stability": "StableCode-3B"}


def load(step: str):
    d = Path("results") / step
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))] \
        if d.is_dir() else []


def ci95(xs):
    if not xs:
        return float("nan"), float("nan")
    if len(xs) < 2:
        return xs[0], 0.0
    return st.mean(xs), 1.96 * st.stdev(xs) / math.sqrt(len(xs))


# 생성 과제("removes duplicate items from a list, preserving order")의 내용어.
# 이름이 이 중 하나도 안 담고 있으면 **과제와 무관한 이름**이다.
TASK_WORDS = ("remove", "duplicat", "dedup", "uniq", "distinct")


def on_task(name) -> bool:
    """생성된 이름이 **요청한 과제의 함수**인가.

    표기(camel/snake) 판정만으로는 부족하다. 어텐션을 지시어에 몰아주면 모델이 그
    단어를 그대로 베껴 `def camelCase(...)`를 쓰는데, 정규식상 camel이라 준수로
    세어진다(name_health도 통과한다 — 형식·길이·반복이 다 정상이므로).
    실제로 Spotlight rule_word ψ0.3에서 Llama 17/21, DeepSeek 8/21이 `camelCase`였다.
    """
    return bool(name) and any(w in name.lower() for w in TASK_WORDS)


def compliant_real(ex) -> float:
    """진짜 준수 — **표기가 맞고 + 과제도 맞아야** 1."""
    return 1.0 if (ex["compliant"] and on_task(ex["name"])) else 0.0


def _gen_key(ex) -> tuple:
    """생성 결과를 묶는 조건 키 — **방법을 반드시 포함한다.**

    strength만으로 묶으면 strength가 없는 Spotlight(attn_amplify)가 `or 0.0` 때문에
    무개입과 같은 칸에 조용히 섞인다. 값이 뒤섞여도 에러가 안 나므로 키에 방법을 넣는다.
    """
    meth = ex["method"]
    if meth == "value_add":
        return (1, float(ex["strength"]), "")
    if meth == "attn_amplify":
        return (2, float(ex["psi_target"]), ex.get("span") or "")
    return (0, 0.0, "")


def _gen_key_order(k: tuple) -> tuple:
    return k


def _gen_key_label(k: tuple) -> str:
    kind, val, span = k
    if kind == 0:
        return "무개입"
    if kind == 1:
        return f"값조향 세기{val:g}"
    return f"SL {span[:6]} ψ{val:g}"


def pearson(xs, ys) -> float:
    """상관계수. 표본이 2개 이하거나 한쪽이 상수면 nan."""
    if len(xs) < 3:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return num / den if den else float("nan")


def peak_layers(rows):
    """모델별 '맞는 층' — 조건에 쓰인 값 중 큰 쪽(엉뚱한 층은 초반으로 잡았다)."""
    seen = defaultdict(set)
    for r in rows:
        ex = r["metrics"]["extra"]
        if ex["method"] == "value_add":
            seen[r["condition"]["model"]["family"]].add(ex["layer"])
    return {m: max(v) for m, v in seen.items()}, {m: min(v) for m, v in seen.items()}


def main() -> None:
    steer = load("step6_steer")
    gen = load("step6_steer-generate")
    cross = load("step6_steer-crosslayer")
    fig_dir = None
    if "--figs" in sys.argv:
        fig_dir = Path(sys.argv[sys.argv.index("--figs") + 1])
        fig_dir.mkdir(parents=True, exist_ok=True)
    PEAK, EARLY = peak_layers(steer)

    # ── ① 헤드라인 비교표 ────────────────────────────────────────────
    rec = defaultdict(lambda: defaultdict(list))
    psi = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    und = defaultdict(int)
    for r in steer:
        ex = r["metrics"]["extra"]
        m = r["condition"]["model"]["family"]
        if ex.get("undecidable") or ex.get("recovery") is None:
            und[m] += 1
            continue
        meth = ex["method"]
        if meth == "none":
            key = "무개입"
        elif meth == "value_add":
            where = "맞는층" if ex["layer"] == PEAK[m] else "엉뚱층"
            key = f"값조향 {where} 세기{ex['strength']:g}"
        else:
            key = f"Spotlight {ex['span']} ψ{ex['psi_target']:g}"
            psi[m][key]["before"].append(ex["attn_span_before"])
            psi[m][key]["after"].append(ex["attn_span_after"])
        rec[m][key].append(ex["recovery"])

    print("=" * 96)
    print("① 처방 비교 — 절벽 바닥에서 얼마나 되살아나나 (회복률 1 = 문맥이 깨끗한 상태)")
    print("=" * 96)
    order = (["무개입"]
             + [f"값조향 맞는층 세기{s:g}" for s in (1, 2, 4, 8)]
             + [f"값조향 엉뚱층 세기{s:g}" for s in (1, 2, 4, 8)]
             + [f"Spotlight {sp} ψ{p:g}" for sp in ("rule_word", "instruction") for p in (0.1, 0.3)])
    print(f"{'방법':<26}" + "".join(f"{LABEL[m].split('-')[0]:>16}" for m in MODELS))
    for key in order:
        line = f"{key:<26}"
        for m in MODELS:
            v = rec[m].get(key)
            line += f"{(f'{st.mean(v):.3f}' if v else '—'):>16}"
        print(line)
    print("\n판정 불가로 뺀 조건: " + " · ".join(f"{m} {und[m]}" for m in MODELS))

    # ── ② Spotlight가 실제로 걸렸는가 ────────────────────────────────
    print("\n" + "=" * 96)
    print("② Spotlight가 실제로 걸렸는가 — 개입 전 → 후 어텐션 비중")
    print("   비중이 올랐는데도 회복이 0이어야 「어텐션 접근으로는 안 된다」의 근거가 된다.")
    print("=" * 96)
    print(f"{'모델':<11}{'조건':<28}{'개입 전':>10}{'개입 후':>10}{'올랐나':>8}")
    for m in MODELS:
        for key in sorted(psi[m]):
            b, a = st.mean(psi[m][key]["before"]), st.mean(psi[m][key]["after"])
            print(f"{m:<11}{key:<28}{b:>10.3f}{a:>10.3f}{'예' if a > b + 1e-9 else '아니오':>8}")

    # ── ③ 이름이 멀쩡한가 ────────────────────────────────────────────
    if gen:
        print("\n" + "=" * 96)
        print("③ 조향을 세게 걸면 이름이 깨지는가 (실제 생성)")
        print("=" * 96)
        # 조건 키에 **방법을 함께** 넣는다. strength만으로 묶으면 strength가 없는
        # Spotlight(attn_amplify)가 무개입(0.0)과 같은 칸에 조용히 섞인다.
        by = defaultdict(lambda: defaultdict(list))
        for r in gen:
            ex = r["metrics"]["extra"]
            by[r["condition"]["model"]["family"]][_gen_key(ex)].append(ex)
        print(f"{'모델':<11}{'조건':>18}{'진짜 준수율':>12}{'과제 맞음':>10}"
              f"{'camel':>8}{'snake':>8}{'그 외':>8}{'n':>5}")
        for m in MODELS:
            for s in sorted(by[m], key=_gen_key_order):
                v = by[m][s]
                nota = [e["notation"] for e in v]
                print(f"{m:<11}{_gen_key_label(s):>18}{st.mean([compliant_real(e) for e in v]):>9.3f}"
                      f"{st.mean([on_task(e['name']) for e in v]):>12.3f}"
                      f"{nota.count('camel') / len(v):>8.2f}{nota.count('snake') / len(v):>8.2f}"
                      f"{nota.count('other') / len(v):>8.2f}{len(v):>5}")
            print()
        bad = [e for r in gen if not (e := r["metrics"]["extra"])["name_ok"]]
        print(f"깨진 이름 {len(bad)}개 / 전체 {len(gen)}개")
        for e in bad[:8]:
            print(f"  세기 {e['strength'] or 0:g}: {e['name']!r} — {e['name_reason']}")

    # ── ③-b 점수 회복률이 실제 준수율을 대변하는가 ────────────────────
    if gen:
        print("\n" + "=" * 96)
        print("③-b 점수 회복률과 실제 준수율의 관계 — 같은 조건끼리 짝지어 본다")
        print("    회복률은 step6_steer, 준수율은 step6_steer-generate. 조건이 겹치는 것만 쓴다.")
        print("=" * 96)
        sc_by = defaultdict(list)
        for r in steer:
            ex = r["metrics"]["extra"]
            if ex.get("undecidable") or ex.get("recovery") is None:
                continue
            sc_by[(r["condition"]["model"]["family"], ex["method"],
                   ex.get("layer"), _gen_key(ex))].append(ex["recovery"])
        gn_by = defaultdict(list)
        for r in gen:
            ex = r["metrics"]["extra"]
            gn_by[(r["condition"]["model"]["family"], ex["method"],
                   ex.get("layer"), _gen_key(ex))].append(compliant_real(ex))
        pairs = [(k[0], k[3], st.mean(sc_by[k]), st.mean(gn_by[k]))
                 for k in gn_by if k in sc_by]

        print(f"{'모델':<11}{'조건':>18}{'회복률':>10}{'실제 준수율':>13}")
        for m in MODELS:
            for _, key, s, g in sorted([p for p in pairs if p[0] == m], key=lambda p: p[1]):
                print(f"{m:<11}{_gen_key_label(key):>18}{s:>10.3f}{g:>13.3f}")
            print()

        rs = [(p[2], p[3]) for p in pairs]
        strong = [(p[2], p[3]) for p in pairs if p[1][0] == 1 and p[1][1] >= 2]
        weak = [(p[2], p[3]) for p in pairs if p[1][0] != 1 or p[1][1] <= 2]
        print(f"{'구간':<26}{'n':>5}{'상관계수 r':>12}")
        for tag, sel in (("전체", rs), ("약한 조향(세기 ≤2)", weak), ("센 조향(세기 ≥2)", strong)):
            print(f"{tag:<26}{len(sel):>5}"
                  f"{pearson([a for a, _ in sel], [b for _, b in sel]):>12.3f}")

        lo = [p for p in pairs if p[2] < 1.0 and p[1][0] != 0]
        hi = [p for p in pairs if p[2] >= 1.5]
        print("\n회복률 구간별 실제 준수율의 **범위** — 이게 핵심이다")
        print(f"{'회복률':<16}{'n':>5}{'준수율 최소':>12}{'준수율 최대':>12}")
        for tag, sel in (("< 1.0", lo), ("≥ 1.5", hi)):
            if sel:
                print(f"{tag:<16}{len(sel):>5}{min(p[3] for p in sel):>12.3f}"
                      f"{max(p[3] for p in sel):>12.3f}")
        print("\n같은 회복률에서 준수율이 넓게 흩어지면, 회복률만으로 처방의 성패를 말할 수 없다.")

    # ── ④ 방향이 층 특이적인가 ───────────────────────────────────────
    if cross:
        print("\n" + "=" * 96)
        print("④ 방향이 층 특이적인가 — 맞는 층 방향을 엉뚱한 층에 넣으면")
        print("=" * 96)
        cr = defaultdict(lambda: defaultdict(list))
        for r in cross:
            ex = r["metrics"]["extra"]
            if ex.get("undecidable") or ex.get("recovery") is None:
                continue
            cr[r["condition"]["model"]["family"]][ex["strength"]].append(ex["recovery"])
        print(f"{'모델':<11}{'세기':>6}{'맞는층→맞는층':>16}{'엉뚱층→엉뚱층':>16}{'맞는층→엉뚱층':>16}")
        for m in MODELS:
            for s in sorted(cr[m]):
                same = rec[m].get(f"값조향 맞는층 세기{s:g}")
                early = rec[m].get(f"값조향 엉뚱층 세기{s:g}")
                f = lambda v: f"{st.mean(v):.3f}" if v else "—"     # noqa: E731
                print(f"{m:<11}{s:>6g}{f(same):>16}{f(early):>16}{f(cr[m][s]):>16}")
            print()

    if fig_dir:
        _figures(rec, gen, PEAK, fig_dir, steer)
        print(f"그림 → {fig_dir}/")


def _figures(rec, gen, PEAK, out: Path, steer=()):
    global _STEER_CACHE
    _STEER_CACHE = steer
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    matplotlib.rcParams.update({"font.family": "serif", "font.size": 8, "figure.dpi": 300,
                                "axes.labelsize": 8, "xtick.labelsize": 7,
                                "ytick.labelsize": 7, "legend.fontsize": 7})
    S = [1, 2, 4, 8]
    # 세기 곡선 — 방법별
    for m in MODELS:
        if not rec[m]:
            continue
        fig, ax = plt.subplots(figsize=(3.3, 2.3))
        for where, color, lab in (("맞는층", "tab:blue", f"Value steering @L{PEAK[m]}"),
                                  ("엉뚱층", "tab:cyan", "Value steering @early layer")):
            ys = [st.mean(rec[m][f"값조향 {where} 세기{s:g}"])
                  if rec[m].get(f"값조향 {where} 세기{s:g}") else float("nan") for s in S]
            ax.plot(S, ys, marker="o", ms=3, color=color, label=lab)
        sp = [st.mean(v) for k, v in rec[m].items() if k.startswith("Spotlight")]
        if sp:
            ax.axhline(max(sp), color="0.45", linestyle="--", linewidth=1.0,
                       label="Spotlight (best)")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axhline(1, color="tab:red", linewidth=0.7, linestyle=":")
        ax.set_xscale("log", base=2); ax.set_xticks(S); ax.set_xticklabels([str(s) for s in S])
        ax.set_xlabel("Steering strength"); ax.set_ylabel("Compliance recovery")
        ax.margins(y=0.10); ax.grid(alpha=0.25, linewidth=0.4)
        # 범례는 **축 밖(위)** 에 둔다. 안에 두면 곡선이 글자를 뚫고 지나간다
        # (Llama·StableCode에서 실제로 겹쳤다). bbox_inches="tight"로 저장해야 안 잘린다.
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01),
                  borderaxespad=0.0, ncol=1, handlelength=1.4)
        fig.savefig(out / f"strength_{m}.pdf", bbox_inches="tight")
        fig.savefig(out / f"strength_{m}.png", bbox_inches="tight")
        plt.close(fig)

    if not gen:
        return
    by = defaultdict(lambda: defaultdict(list))
    for r in gen:
        ex = r["metrics"]["extra"]
        by[r["condition"]["model"]["family"]][ex["strength"] or 0.0].append(ex)
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for m, color in zip(MODELS, ("tab:blue", "tab:orange", "tab:green", "tab:red")):
        if not by[m]:
            continue
        xs = sorted(by[m])
        ax.plot(xs, [st.mean([e["compliant"] for e in by[m][x]]) for x in xs],
                marker="o", ms=3, color=color, label=LABEL[m].split("-")[0])
        ax.plot(xs, [st.mean([e["name_ok"] for e in by[m][x]]) for x in xs],
                marker="s", ms=3, color=color, linestyle="--", alpha=0.6)
    # 선 종류가 무엇을 뜻하는지 별도 범례 — 없으면 점선 4개가 설명 없이 떠 있다
    style_keys = [Line2D([], [], color="0.3", marker="o", ms=3, linestyle="-",
                         label="compliance (solid)"),
                  Line2D([], [], color="0.3", marker="s", ms=3, linestyle="--",
                         label="name intact (dashed)")]
    ax.set_xlabel("Steering strength"); ax.set_ylabel("Rate")
    ax.set_ylim(-0.05, 1.10); ax.grid(alpha=0.25, linewidth=0.4)
    # 범례 둘을 **축 밖(위)** 에 두 줄로 합친다. 선 종류 범례를 축 안(lower left)에 두면
    # 바닥에 깔리는 StableCode 곡선이 글자를 뚫고 지나간다 — 실제로 겹쳤다.
    model_keys, model_labels = ax.get_legend_handles_labels()
    ax.legend(handles=model_keys + style_keys,
              labels=model_labels + [h.get_label() for h in style_keys],
              frameon=False, fontsize=6.5, ncol=3, columnspacing=0.9, handlelength=1.3,
              loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0.0)
    fig.savefig(out / "name_health.pdf", bbox_inches="tight")
    fig.savefig(out / "name_health.png", bbox_inches="tight")
    plt.close(fig)

    # ── 처방 비교, **실제 준수율로** ────────────────────────────────────
    # explain_headline은 같은 비교를 '점수'로 한다. 그런데 점수는 준수율을 대변하지
    # 않으므로(§3-3), 논문에 쓰는 방법 비교는 이 그림이어야 한다.
    g = defaultdict(dict)
    for r in gen:
        ex = r["metrics"]["extra"]
        g[r["condition"]["model"]["family"]].setdefault(_gen_key(ex), []).append(compliant_real(ex))
    bars = [("no intervention", "0.75", lambda d: d.get((0, 0.0, ""))),
            ("Value steering (best α)", "tab:blue",
             lambda d: max((v for k, v in d.items() if k[0] == 1), key=st.mean, default=None)),
            ("Spotlight (best span·ψ)", "tab:gray",
             lambda d: max((v for k, v in d.items() if k[0] == 2), key=st.mean, default=None))]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    width, xs = 0.26, range(len(MODELS))
    for off, (lab, color, pick) in zip((-1, 0, 1), bars):
        ys, es = [], []
        for m in MODELS:
            v = pick(g[m])
            a, c = ci95(v) if v else (float("nan"), float("nan"))
            ys.append(a); es.append(c)
        ax.bar([i + off * width for i in xs], ys, width, yerr=es, capsize=2,
               color=color, label=lab, error_kw={"linewidth": 0.7})
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABEL[m].split("-")[0] for m in MODELS], fontsize=6.5)
    ax.set_ylabel("Actual compliance rate")
    ax.set_ylim(0, 1.32)
    ax.legend(frameon=False, fontsize=6.2, ncol=1, handlelength=1.3,
              loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0.0)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    fig.savefig(out / "compliance_by_method.pdf", bbox_inches="tight")
    fig.savefig(out / "compliance_by_method.png", bbox_inches="tight")
    plt.close(fig)

    # ── 용량-반응: 세기를 올리면 이름이 어디로 가나 ──────────────────────
    # step6의 핵심을 한 장에. 개입만 하면 snake(위반)가 사라지고(=인과), 세게 밀면
    # camel을 지나 Pascal로 넘어간다(=과잉). 회복률은 그동안 계속 오른다(=용량계로 못 쓴다).
    import re as _re
    PASCAL = _re.compile(r"^[A-Z][a-z0-9]*(?:[A-Z][a-z0-9]*)+$")

    def _bucket(ex):
        n, nota = ex["name"], ex["notation"]
        if nota == "snake":
            return 0                                   # 위반
        if nota == "camel" and on_task(n):
            return 1                                   # 준수
        if n and PASCAL.match(n):
            return 2                                   # camel 과잉(PascalCase)
        return 3                                       # 그 외 · 이름 없음

    S = [0, 1, 2, 4, 8]
    BUCK = [("violating (snake)", "tab:red"), ("compliant (camel)", "tab:blue"),
            ("overshoot (Pascal)", "tab:orange"), ("broken / none", "0.7")]
    dose = defaultdict(lambda: defaultdict(list))
    for r in gen:
        ex = r["metrics"]["extra"]
        if ex["method"] not in ("none", "value_add"):
            continue
        dose[r["condition"]["model"]["family"]][ex["strength"] or 0.0].append(_bucket(ex))

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0), sharey=True)
    for ax, m in zip(axes, MODELS):
        bottom = [0.0] * len(S)
        for b, (lab, col) in enumerate(BUCK):
            ys = [(dose[m][s].count(b) / len(dose[m][s])) if dose[m].get(s) else 0.0 for s in S]
            ax.bar(range(len(S)), ys, 0.72, bottom=bottom, color=col,
                   label=lab if m == MODELS[0] else None)
            bottom = [a + b2 for a, b2 in zip(bottom, ys)]
        ax2 = ax.twinx()
        ys = [st.mean(rec[m][f"값조향 맞는층 세기{s:g}"])
              if s and rec[m].get(f"값조향 맞는층 세기{s:g}") else 0.0 for s in S]
        ax2.plot(range(len(S)), ys, color="black", marker="o", ms=2.5, linewidth=1.0)
        ax2.set_ylim(0, 4.6)
        ax2.tick_params(axis="y", labelsize=5.5)
        if m != MODELS[-1]:
            ax2.set_yticklabels([])
        else:
            ax2.set_ylabel("score recovery", fontsize=6)
        ax.set_xticks(range(len(S))); ax.set_xticklabels([str(s) for s in S], fontsize=6.5)
        ax.set_title(LABEL[m].split("-")[0], fontsize=7)
        ax.set_xlabel("Steering strength", fontsize=7)
    axes[0].set_ylabel("Share of generated names")
    axes[0].set_ylim(0, 1.0)
    fig.legend(frameon=False, fontsize=6.2, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=1.2)
    fig.savefig(out / "dose_response.pdf", bbox_inches="tight")
    fig.savefig(out / "dose_response.png", bbox_inches="tight")
    plt.close(fig)

    # ── 회복률 vs 실제 준수율 산점도 ────────────────────────────────────
    # "점수가 준수율을 대변하지 않는다"를 한 장으로. 두 축이 짝지어진 조건만 찍는다.
    sc_by, gn_by = defaultdict(list), defaultdict(list)
    for r in _STEER_CACHE:
        ex = r["metrics"]["extra"]
        if ex.get("undecidable") or ex.get("recovery") is None:
            continue
        sc_by[(r["condition"]["model"]["family"], ex.get("layer"), _gen_key(ex))].append(ex["recovery"])
    for r in gen:
        ex = r["metrics"]["extra"]
        gn_by[(r["condition"]["model"]["family"], ex.get("layer"), _gen_key(ex))].append(compliant_real(ex))
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    MK = {0: ("o", "no intervention"), 1: ("^", "value steering"), 2: ("s", "Spotlight")}
    C = dict(zip(MODELS, ("tab:blue", "tab:orange", "tab:green", "tab:red")))
    for k in gn_by:
        if k not in sc_by:
            continue
        kind = k[2][0]
        ax.scatter(st.mean(sc_by[k]), st.mean(gn_by[k]), marker=MK[kind][0], s=22,
                   facecolors="none" if kind == 2 else C[k[0]], edgecolors=C[k[0]],
                   linewidths=1.0)
    ax.axvline(1.0, color="tab:red", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Preference-score recovery")
    ax.set_ylabel("Actual compliance rate")
    ax.set_ylim(-0.08, 1.15)
    ax.grid(alpha=0.25, linewidth=0.4)
    # 범례는 **모양=처방, 색=모델**을 따로 세운다. scatter에 label을 걸면 그때그때
    # 먼저 그려진 모델 색이 범례에 박혀, 처방을 뜻하는 마커가 특정 모델 색으로 보인다.
    keys = [Line2D([], [], marker=MK[i][0], color="none", markeredgecolor="0.3",
                   markerfacecolor="none" if i == 2 else "0.3", ms=4, label=MK[i][1])
            for i in (0, 1, 2)]
    keys += [Line2D([], [], marker="s", color="none", markerfacecolor=C[m],
                    markeredgecolor=C[m], ms=4, label=LABEL[m].split("-")[0]) for m in MODELS]
    ax.legend(handles=keys, frameon=False, fontsize=6.0, ncol=4, handlelength=1.0,
              columnspacing=0.7, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              borderaxespad=0.0)
    fig.savefig(out / "score_vs_compliance.pdf", bbox_inches="tight")
    fig.savefig(out / "score_vs_compliance.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
