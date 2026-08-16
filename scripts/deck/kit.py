"""도해·수식 이미지 공통 도구. 슬라이드에 놓일 크기 그대로 그린다(축소 없음 → 글자 크기 정확)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from pathlib import Path

CANVAS   = "#0B0B0B"; SOFT = "#1C1C1C"; WHITE = "#FFFFFF"
ASH      = "#B9B9B9"; MUTE = "#797979"; HAIR = "#353535"; BRAND = "#D97757"
SANS     = ["Inter", "DejaVu Sans"]; MONO = ["IBM Plex Mono", "DejaVu Sans Mono"]
OUT = Path("assets"); OUT.mkdir(exist_ok=True)
PX = 1 / 144.0   # 1920x1080 기준틀의 1px = 1/144 inch

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": SANS,
    "mathtext.fontset": "cm", "figure.dpi": 300, "savefig.dpi": 300,
    "text.color": WHITE, "axes.edgecolor": HAIR,
})


class Dia:
    """px 좌표(도해 자체 기준틀)로 그리는 얇은 래퍼."""
    def __init__(self, w_px, h_px):
        self.W, self.H = w_px, h_px
        self.fig = plt.figure(figsize=(w_px * PX, h_px * PX))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w_px); self.ax.set_ylim(h_px, 0); self.ax.axis("off")

    def box(self, x, y, w, h, label=None, sub=None, fc=SOFT, ec=HAIR, lw=1.0,
            r=8, tc=WHITE, fs=9.5, subfs=8, mono=False, subcolor=MUTE, align="center"):
        self.ax.add_patch(FancyBboxPatch(
            (x, y + h), w, -h, boxstyle=f"round,pad=0,rounding_size={r}",
            fc=fc, ec=ec, lw=lw, mutation_aspect=1))
        fam = MONO if mono else SANS
        if label is not None:
            cy = y + h / 2 - (7 if sub else 0)
            tx, ha = (x + w / 2, "center") if align == "center" else (x + 12, "left")
            self.ax.text(tx, cy, label, ha=ha, va="center", color=tc,
                         fontsize=fs, family=fam)
        if sub is not None:
            tx, ha = (x + w / 2, "center") if align == "center" else (x + 12, "left")
            self.ax.text(tx, y + h / 2 + 10, sub, ha=ha, va="center",
                         color=subcolor, fontsize=subfs, family=MONO)
        return (x, y, w, h)

    def arrow(self, p0, p1, color=MUTE, lw=1.2, style="-|>", rad=0.0, ls="-"):
        self.ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=9, color=color, lw=lw,
            linestyle=ls, connectionstyle=f"arc3,rad={rad}",
            shrinkA=2, shrinkB=2))

    def text(self, x, y, s, color=ASH, fs=9, ha="left", va="center",
             mono=False, weight="normal", style="normal"):
        self.ax.text(x, y, s, color=color, fontsize=fs, ha=ha, va=va,
                     family=MONO if mono else SANS, fontweight=weight, style=style)

    def eyebrow(self, x, y, s, color=MUTE, fs=7.5):
        self.ax.text(x, y, s.upper(), color=color, fontsize=fs, ha="left",
                     va="center", family=SANS, fontweight=600)

    def line(self, x0, y0, x1, y1, color=HAIR, lw=1.0, ls="-"):
        self.ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, solid_capstyle="butt")

    def cell(self, x, y, w, h, fc, ec=None):
        self.ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec or fc, lw=0.5))

    def save(self, name):
        p = OUT / f"{name}.png"
        self.fig.savefig(p, transparent=True, dpi=300, pad_inches=0)
        plt.close(self.fig)
        return p


def eq(name, tex, fs=22, color=WHITE):
    """수식 하나를 투명 PNG로. 배치 때 높이로 맞춘다."""
    fig = plt.figure(figsize=(0.1, 0.1))
    t = fig.text(0, 0, f"${tex}$", fontsize=fs, color=color)
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer())
    fig.set_size_inches(bb.width / fig.dpi + 0.10, bb.height / fig.dpi + 0.10)
    t.set_position((0.05 / fig.get_figwidth(), 0.05 / fig.get_figheight()))
    t.set_va("bottom"); t.set_ha("left")
    p = OUT / f"eq_{name}.png"
    fig.savefig(p, transparent=True, dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return p
