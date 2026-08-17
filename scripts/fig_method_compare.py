# -*- coding: utf-8 -*-
"""논문 개념 그림 — 기존 처방과 우리 처방을 한 장에.

어텐션 출력 o = Σ a·v 를 위에 두고, **어느 인자를 건드리는가**로 두 줄을 가른다.
같은 프롬프트·같은 모델이고 손잡이만 다르다는 것이 이 그림의 요점이다.

수치는 `docs/step6/results.md` §3-2·§3-3에서 온다(어텐션 비중 0.002→0.231,
실제 준수율 0.000 대 1.000, 둘 다 4모델 중 3모델 기준).

    python scripts/fig_method_compare.py [--out docs/figures]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

for _f in fm.findSystemFonts(fontpaths=[str(Path.home() / ".fonts")]):
    fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "NanumGothic", "font.size": 9})

INK, MUT, HAIR = "#111111", "#777777", "#CCCCCC"
BAD, GOOD = "#B34A28", "#1F6FB4"


def main() -> None:
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv \
        else Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

    def box(x, y, w, h, t, sub=None, ec=HAIR, fc="white", tc=INK, fs=9.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    ec=ec, fc=fc, lw=1.1))
        ax.text(x + w / 2, y + h / 2 + (1.7 if sub else 0), t, ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold" if bold else "normal")
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.8, sub, ha="center", va="center",
                    fontsize=7.6, color=MUT)

    def arrow(x0, y0, x1, y1, c=MUT, lw=1.4):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=11, color=c, lw=lw, shrinkA=2, shrinkB=2))

    # 머리 — 분해식. 두 인자가 각각 무엇인지 아래에 단다
    ax.text(50, 57.5, r"$o_i \;=\; \sum_j\, a_{ij}\, v_j$", ha="center", fontsize=15, color=INK)
    ax.text(39.0, 51.4, "얼마나 보나", ha="center", fontsize=8, color=BAD)
    ax.text(56.5, 51.4, "무엇이 실려 오나", ha="center", fontsize=8, color=GOOD)
    ax.plot([41.5, 41.5], [54.4, 53.0], color=BAD, lw=1)
    ax.plot([53.0, 53.0], [54.4, 53.0], color=GOOD, lw=1)
    ax.plot([3, 97], [48, 48], color=HAIR, lw=1)

    # 공통 입력 — 상자 제목은 위로 빼서 안쪽 글자와 겹치지 않게 한다
    box(2, 12, 16, 30, "", None, fc="#FAFAFA")
    ax.text(10, 38.4, "프롬프트", ha="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.text(10, 31.5, "지침", ha="center", fontsize=8.5, color=INK)
    ax.text(10, 28.6, "“camelCase로 써라”", ha="center", fontsize=7.2, color=MUT)
    ax.text(10, 21.0, "선행 코드", ha="center", fontsize=8.5, color=INK)
    ax.text(10, 18.1, "위반 12 / 12", ha="center", fontsize=7.2, color=MUT)

    # 윗줄 — 기존
    ax.text(26, 41.5, "기존 — Spotlight", fontsize=9.5, color=BAD, fontweight="bold")
    arrow(18, 34, 25, 34, BAD)
    box(25, 28, 20, 12, "어텐션 $a$를 키운다", "지침 스팬 비중 ψ 올림", ec=BAD, tc=BAD)
    arrow(45, 34, 52, 34, BAD)
    box(52, 28, 20, 12, "ψ  0.002 → 0.231", "100배 — 실제로 걸렸다", ec=HAIR)
    arrow(72, 34, 79, 34, BAD)
    box(79, 28, 19, 12, "준수율  0.000", "4모델 중 3", ec=BAD, fc="#FBF1EE", tc=BAD, bold=True)

    # 아랫줄 — 우리
    ax.text(26, 23.5, "우리 — 값 조향", fontsize=9.5, color=GOOD, fontweight="bold")
    arrow(18, 16, 25, 16, GOOD)
    box(25, 10, 20, 12, "값 $v$를 민다", r"$h \leftarrow h + \alpha\, d$", ec=GOOD, tc=GOOD)
    arrow(45, 16, 52, 16, GOOD)
    box(52, 10, 20, 12, "후반 특정 한 층", "깊이 56~71%", ec=HAIR)
    arrow(72, 16, 79, 16, GOOD)
    box(79, 10, 19, 12, "준수율  1.000", "4모델 중 3", ec=GOOD, fc="#EEF4FB", tc=GOOD, bold=True)

    ax.text(50, 3.4, "같은 프롬프트 · 같은 모델 — 어느 손잡이를 돌리느냐만 다르다",
            ha="center", fontsize=8.4, color=MUT)

    for ext in ("png", "pdf"):
        fig.savefig(out / f"method_compare.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"→ {out}/method_compare.(png|pdf)")


if __name__ == "__main__":
    main()
