"""Generate the GitHub social-preview card (1200x630). Reproducible: python3 make_social_preview.py

The card shows what the tool does to a three-entry bibliography: two real papers verified, one
invented DOI refused. The parse runs here - parse_refs is pure local text handling - so the card
cannot claim to read a file it can no longer read. The verdicts are the ones the tool returns for
these three, recorded next to them rather than looked up, because a card must draw without a
network and this one is generated in CI.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from cardkit import SANS, card  # noqa: E402
from scholarcheck.cli import parse_refs  # noqa: E402

BIB = """@inproceedings{he2016resnet, title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming}, doi={10.1109/CVPR.2016.90}}
@article{wang2025kakeya, title={Volume estimates for unions of convex sets},
  author={Wang, Hong}, doi={10.48550/arXiv.2502.17655}}
@article{fake2024zebra, title={Zebra stripes as a quantum error-correcting code},
  author={Nobody, A.}, doi={10.9999/nonexistent.2024.00001}}
"""
# two columns, not three: the bibtex key tells a reader nothing the verdict does not
VERDICT = {"he2016resnet": ("ok", "Deep Residual Learning, CVPR 2016"),
           "wang2025kakeya": ("ok", "Volume estimates, arXiv 2502.17655"),
           "fake2024zebra": ("SUSPECT", "10.9999/nonexistent - no such DOI")}

REFS = parse_refs(BIB)
if len(REFS) != 3:
    raise SystemExit(f"parse_refs reads {len(REFS)} of 3 entries; the card would misrepresent it")


def chart(ax, accent):
    import matplotlib.pyplot as plt
    y = 3.28
    for r in REFS:
        state, note = VERDICT[r["key"]]
        bad = state != "ok"
        colour = "#cf222e" if bad else "#1a7f37"
        ax.add_patch(plt.Rectangle((0.80, y - 0.19), 0.38, 0.38, color=colour, zorder=3))
        ax.text(1.42, y, state, fontsize=34, fontweight="bold" if bad else "normal",
                color=colour, family=SANS, va="center")
        ax.text(3.95, y, note, fontsize=34,
                color="#17181a" if bad else "#55585c", family=SANS, va="center")
        y -= 0.72
    ax.text(0.80, y - 0.08, "exit 1, so it stops a build", fontsize=36, fontweight="bold",
            color="#17181a", family=SANS, va="center")


out = card(
    out=str(pathlib.Path(__file__).parent / "social-preview.png"),
    accent="#1a7f37", badge="S",
    kicker="PYTHON PACKAGE  ·  pip install scholarcheck",
    headline="Stop hallucinated citations",
    evidence="checked against the registry that issued it",
    chart=chart,
    footer="github.com/GuoCheng24/scholarcheck",
    headline_size=46,
)
print(f"written {pathlib.Path(out).name}  {len(REFS)} references parsed")
