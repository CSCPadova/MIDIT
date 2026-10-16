"""
scripts/figures/preprocessing_pipeline_v2.py

Corrected, paper-style MIDIT preprocessing/data pipeline figure. Fixes the
factual errors of the previous version (augmentation before tokenization; .pt
stores full-song sequences; split + 8-bar chunking happen at load time, not in
preprocessing; split is song-level, not per-file).

Phase captions are placed as rotated labels on the left so they never cross the
inter-phase arrow; symbols are rendered with mathtext for a LaTeX-like look.

Run:  python scripts/figures/preprocessing_pipeline_v2.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── palette (muted, publication style) ────────────────────────────────────────
A1, A2   = "#2C6E9B", "#3E7C5A"       # phase accents
BAND1, BAND2 = "#EEF3F8", "#EDF4EF"
BAND_ED  = "#DDE4EC"
BOX_FILL, BOX_ED = "#FFFFFF", "#C9D1D9"
TEXT, SUB = "#1B2733", "#4C5866"
ARROW, ANNOT = "#6B7785", "#6B7785"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",
})

FIG_W, FIG_H = 7.6, 12.4
CX = 3.65
BOX_W, BOX_H = 4.4, 1.16
GAP = 0.50
ACCENT_W = 0.14
BAND_X0 = CX - BOX_W/2 - 0.72

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H); ax.axis("off")

ax.text(CX, FIG_H-0.42, "MIDIT — Data Preprocessing & Loading Pipeline",
        ha="center", va="center", fontsize=13.5, fontweight="bold", color="#152029")

# ── stages: (num, title, sub[, may contain \n], accent, right-annotation) ─────
stages = [
    ("1", "Raw MIDI input",
     r"Lakh MIDI (LMD) subset   $\cdot$   $\approx$50,000 files", A1, ""),
    ("2", "Quality filtering",
     "F1: $\\geq$50 notes      F2: $\\geq$15 s duration\n"
     "F3: velocity not near-constant ($\\leq$90% in one bin)", A1,
     "80.3% pass\n$\\rightarrow$ 40,138 songs"),
    ("3", "Pitch-shift  +  REMI+ tokenization",
     "for each shift $k \\in \\{-5,\\dots,+5\\}$: transpose (drums unchanged)\n"
     "clip to [21,108], then MidiTok REMI+ tokenize (vocab 550)", A1,
     "$\\times$11\naugmentation"),
    ("4", "Store  .pt  (full-song sequences)",
     r"token ids + bar positions per (song, shift)   $\cdot$   441,518 files", A1, ""),
    ("5", "Song-level split   (at load time)",
     "stem-grouped (all shifts together), seed 42, 80/10/10\n"
     "$\\rightarrow$ 353,210 training files", A2,
     "prevents\naugmentation leakage"),
    ("6", "On-the-fly 8-bar chunking  $\\rightarrow$  masking",
     "on load: extract an 8-bar chunk from a .pt,\n"
     "then apply token-level / bar-fill masking", A2, ""),
]

# y-top of each box (extra gap before the phase-2 group at index 4)
tops, y = [], FIG_H - 1.55
for i in range(len(stages)):
    if i == 4:
        y -= 0.55
    tops.append(y)
    y -= (BOX_H + GAP)

# ── phase bands + rotated left captions ──────────────────────────────────────
def band(first_top, last_top, fill, accent, label):
    top = first_top + 0.30
    bot = (last_top - BOX_H) - 0.26
    ax.add_patch(FancyBboxPatch((BAND_X0, bot), BOX_W + 2.05, top - bot,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 linewidth=1.0, edgecolor=BAND_ED, facecolor=fill, zorder=0))
    ax.text(BAND_X0 + 0.30, (top + bot)/2, label, rotation=90,
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color=accent, zorder=1)

band(tops[0], tops[3], BAND1, A1, "PHASE 1  —  Offline preprocessing")
band(tops[4], tops[5], BAND2, A2, "PHASE 2  —  Training-time loading")

# ── boxes ────────────────────────────────────────────────────────────────────
def draw_box(num, title, sub, accent, y_top):
    bx, by = CX - BOX_W/2, y_top - BOX_H
    ax.add_patch(FancyBboxPatch((bx, by), BOX_W, BOX_H,
                 boxstyle="round,pad=0.015,rounding_size=0.06",
                 linewidth=1.1, edgecolor=BOX_ED, facecolor=BOX_FILL, zorder=3))
    ax.add_patch(plt.Rectangle((bx, by + 0.06), ACCENT_W, BOX_H - 0.12,
                 facecolor=accent, edgecolor="none", zorder=4))
    ax.add_patch(plt.Circle((bx + 0.44, by + BOX_H/2), 0.17,
                 facecolor=accent, edgecolor="none", zorder=5))
    ax.text(bx + 0.44, by + BOX_H/2, num, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=6)
    two = "\n" in sub
    ax.text(bx + 0.76, by + BOX_H*(0.70 if two else 0.63), title, ha="left",
            va="center", fontsize=10.4, fontweight="bold", color=TEXT, zorder=5)
    ax.text(bx + 0.76, by + BOX_H*(0.28 if two else 0.30), sub, ha="left",
            va="center", fontsize=7.5, color=SUB, zorder=5, linespacing=1.55)

for (num, title, sub, accent, _annot), yt in zip(stages, tops):
    draw_box(num, title, sub, accent, yt)

# ── arrows + right-side annotations ──────────────────────────────────────────
for i in range(len(stages) - 1):
    a_top, a_bot = tops[i] - BOX_H, tops[i+1]
    dashed = (i == 3)
    ax.annotate("", xy=(CX, a_bot), xytext=(CX, a_top),
                arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.5,
                                mutation_scale=13,
                                linestyle="--" if dashed else "-"), zorder=2)
    annot = stages[i][4]
    if annot:
        ax.text(CX + BOX_W/2 + 0.18, (a_top + a_bot)/2, annot,
                ha="left", va="center", fontsize=7.1, style="italic",
                color=ANNOT, linespacing=1.4)
    if dashed:
        ax.text(CX + BOX_W/2 + 0.18, (a_top + a_bot)/2, "reads .pt store",
                ha="left", va="center", fontsize=7.1, style="italic", color=ANNOT)

# ── footnote ──────────────────────────────────────────────────────────────────
ax.text(CX, 0.40,
        "Augmentation precedes tokenization (each shifted MIDI is tokenized). "
        "The .pt files store full-song sequences;\nthe 8-bar chunking is performed "
        "on the fly at load time, not stored. The split is song-level, not per-file.",
        ha="center", va="center", fontsize=7.2, style="italic", color="#5A6570",
        linespacing=1.5)

out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..",
                                    "docs", "img", "preprocessing_pipeline_v2.png"))
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print("Saved:", out)
plt.close(fig)
