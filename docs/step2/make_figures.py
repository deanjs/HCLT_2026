"""step2 결과 그래프 재생성 스크립트.

results/step2_code-observe_results_{model}/ 의 JSON 168개(모델4 × 묶음42)를 읽어
docs/step2/fig1, fig2 PNG를 다시 만든다. 결과물(results/)은 읽기만 한다.

    python docs/step2/make_figures.py     # 저장소 루트에서 실행
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "step2")

MODELS = {  # key -> (표시 이름, 결과 폴더)
    "qwen":     ("Qwen2.5-Coder-3B",    "results/step2_code-observe_results_qwen"),
    "deepseek": ("DeepSeek-Coder-6.7B", "results/step2_code-observe_results_deepseek"),
    "llama":    ("Llama-3.2-3B",        "results/step2_code-observe_results_llama"),
    "stable":   ("StableCode-3B",       "results/step2_code-observe_results_stable"),
}
ORDER = ["qwen", "deepseek", "llama", "stable"]
AXES = [
    ("attention_weight", "Attention weight (axis A: how much the model looks)"),
    ("av_norm", "||a·v|| (axis A x B: actual contribution to residual = impact)"),
    ("v_norm", "||v|| (axis B: size of value content)"),
]
CAMEL, SNAKE = "#2563eb", "#dc2626"


def load_means():
    """모델별 per-layer 평균(묶음 42개 평균)을 계산."""
    agg = {}
    for key in ORDER:
        disp, d = MODELS[key]
        files = sorted(glob.glob(os.path.join(ROOT, d, "*.json")))
        L = len(json.load(open(files[0]))["metrics"]["per_layer"])
        stack = {}
        for f in files:
            pl = json.load(open(f))["metrics"]["per_layer"]
            for sp in ("code_camel", "code_snake"):
                for ax, _ in AXES:
                    stack.setdefault((sp, ax), []).append(
                        [pl[str(l)][f"{sp}__{ax}"] for l in range(L)]
                    )
        means = {k: np.mean(np.array(v), axis=0) for k, v in stack.items()}
        agg[key] = {"disp": disp, "L": L, "means": means, "n": len(files)}
        print(f"{disp}: L={L}, n_batches={len(files)}")
    return agg


def fig1(agg):
    fig, g = plt.subplots(4, 3, figsize=(15, 16))
    for r, key in enumerate(ORDER):
        a = agg[key]; x = np.arange(a["L"])
        for c, (ax, title) in enumerate(AXES):
            p = g[r, c]
            p.plot(x, a["means"][("code_camel", ax)], color=CAMEL, lw=2, label="camelCase context")
            p.plot(x, a["means"][("code_snake", ax)], color=SNAKE, lw=2, label="snake_case context")
            p.set_xlabel("Layer index"); p.grid(alpha=0.25)
            if c == 0:
                p.set_ylabel(f"{a['disp']}\n(L={a['L']})", fontsize=11, fontweight="bold")
            if r == 0:
                p.set_title(title, fontsize=10)
            if r == 0 and c == 0:
                p.legend(fontsize=9)
    fig.suptitle("Step2 (RQ2, observational): camelCase vs snake_case code spans, "
                 "per-layer means over 42 contexts (504 names)", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(os.path.join(OUT, "fig1_camel_vs_snake_grid.png"), dpi=130)


def fig2(agg):
    colors = {"qwen": "#2563eb", "deepseek": "#059669", "llama": "#d97706", "stable": "#7c3aed"}
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5.5))
    for key in ORDER:
        a = agg[key]; depth = np.arange(a["L"]) / (a["L"] - 1); m = a["means"]
        axA.plot(depth, m[("code_camel", "attention_weight")] - m[("code_snake", "attention_weight")],
                 color=colors[key], lw=2, label=a["disp"])
        axB.plot(depth, m[("code_camel", "av_norm")] - m[("code_snake", "av_norm")],
                 color=colors[key], lw=2, label=a["disp"])
    for axp, ttl in [(axA, "Gap on ATTENTION axis (camel - snake)"),
                     (axB, "Gap on IMPACT axis ||a·v|| (camel - snake)")]:
        axp.axhline(0, color="k", lw=0.7); axp.grid(alpha=0.25)
        axp.set_xlabel("Relative depth (0=first layer, 1=last layer)"); axp.set_title(ttl)
        axp.legend(fontsize=9)
    axA.set_ylabel("camel - snake")
    fig.suptitle("Where do camel and snake separate? "
                 "Attention gap stays small; impact gap grows in LATE layers", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, "fig2_gap_attention_vs_impact.png"), dpi=130)


if __name__ == "__main__":
    agg = load_means()
    fig1(agg)
    fig2(agg)
    print("saved fig1, fig2 ->", OUT)
