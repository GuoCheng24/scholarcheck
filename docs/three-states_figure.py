"""Reproduces the figure in this directory.

Requires sciglyph:  pip install sciglyph
Run:  python three-states_figure.py
scholarcheck figure: the three verdicts differ by which sources answered.
That is a pattern, so draw it as one."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from sciglyph import set_canvas, RC, report
from sciglyph.arch import aspect

plt.rcParams.update(RC)
fig = plt.figure(figsize=(11.2, 4.3), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
set_canvas(fig)

INK, MUTE = "#1a1a1a", "#6b6b6b"
GREEN, RED, AMBER, GREY = "#2e7d4f", "#c0392b", "#b8860b", "#b4b4b4"

def disc(x, y, r, fc, ec=None, lw=1.4, z=8):
    ar = aspect()
    ax.add_patch(plt.matplotlib.patches.Ellipse(
        (x, y), 2*r/ar, 2*r, fc=fc, ec=ec or fc, lw=lw, zorder=z))

def cross(x, y, s=.011, c=AMBER, lw=2.0, z=9):
    ar = aspect()
    for dx, dy in ((1, 1), (1, -1)):
        ax.plot([x - s/ar*dx, x + s/ar*dx], [y - s*dy, y + s*dy],
                color=c, lw=lw, zorder=z, solid_capstyle="round")

SRC = ["OpenAlex", "Semantic\nScholar", "Crossref", "arXiv"]
XS = [.330, .420, .510, .600]
ax.text(.5, .945, "the three verdicts are three different evidence patterns",
        fontsize=10.8, ha="center", weight="bold", color=INK, zorder=20)
for x, s in zip(XS, SRC):
    ax.text(x, .845, s, fontsize=7.2, ha="center", va="center", color=MUTE, zorder=20)
ax.plot([.300, .640], [.800, .800], color="#dcdcdc", lw=1.0, zorder=2)

rows = [
    (.665, GREEN, "MATCH", "the paper exists",
     [("hit", 1), ("hit", 1), ("hit", 1), ("miss", 0)],
     '"Deep Residual Learning\nfor Image Recognition"'),
    (.470, RED, "NOT FOUND", "very likely hallucinated",
     [("miss", 0), ("miss", 0), ("miss", 0), ("miss", 0)],
     '"Quantum Topological Radiomics\nfor Zebra Diagnosis"'),
    (.275, AMBER, "INCONCLUSIVE", "no claim is made",
     [("err", 0), ("err", 0), ("err", 0), ("err", 0)],
     "the same real paper,\nwith the network down"),
]
for y, col, verdict, sub, marks, query in rows:
    ax.text(.288, y + .020, query, fontsize=6.9, ha="right", va="center",
            color=INK, zorder=20)
    for x, (kind, _) in zip(XS, marks):
        if kind == "hit":
            disc(x, y + .020, .017, GREEN)
        elif kind == "miss":
            disc(x, y + .020, .017, "white", GREY, 1.6)
        else:
            cross(x, y + .020)
    ax.add_patch(FancyBboxPatch((.690, y - .052), .285, .145,
                                boxstyle="round,pad=0,rounding_size=.014",
                                fc="white", ec=col, lw=1.5, zorder=4))
    ax.text(.708, y + .046, verdict, fontsize=9.6, ha="left", weight="bold",
            color=col, zorder=20)
    ax.text(.708, y - .008, sub, fontsize=7.2, ha="left", color=INK, zorder=20)
    ax.annotate("", xy=(.686, y + .020), xytext=(.628, y + .020),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3, mutation_scale=11))

# legend, drawn not typed
LY = .118
disc(.332, LY, .015, GREEN); ax.text(.352, LY, "record found", fontsize=7,
                                     va="center", color=INK, zorder=20)
disc(.452, LY, .015, "white", GREY, 1.6); ax.text(.472, LY, "queried, no record",
                                                  fontsize=7, va="center", color=INK, zorder=20)
cross(.604, LY, s=.010); ax.text(.622, LY, "source unreachable", fontsize=7,
                                 va="center", color=INK, zorder=20)

ax.text(.5, .042, "an outage produces the same empty result as a fake reference — "
                  "unless you track which sources answered",
        fontsize=7.4, ha="center", color=AMBER, style="italic", zorder=20)

report(fig, ax)
fig.savefig(Path(__file__).with_name("three-states.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
print("saved")
