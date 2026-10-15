import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# ── colours ──────────────────────────────────────────────────────────────────
C_INPUT  = "#4A90D9"   # blue
C_PROC   = "#2A9D8F"   # teal
C_SPLIT  = "#E76F51"   # orange
C_OUT    = "#57CC99"   # green
C_ARROW  = "#555555"
C_BG     = "white"
C_TEXT   = "white"
C_ANNOT  = "#333333"

# ── layout constants ─────────────────────────────────────────────────────────
FIG_W, FIG_H = 6.5, 12.5
BOX_W, BOX_H = 4.2, 1.1
CX = FIG_W / 2          # horizontal centre
MARGIN_TOP = 0.6
STAGE_GAP  = 0.62        # gap between bottom of one box and top of next
ARROW_H    = STAGE_GAP   # arrow fills the gap

stages = [
    # (label, sublabel, colour)
    ("LMD-full",
     "168,385 MIDI files  ·  lmd_full subset",
     C_INPUT),

    ("Quality Filters",
     "F1: ≥ 50 notes   F2: ≥ 15 s duration\nF3: velocity not constant (>90 % in one bin)",
     C_PROC),

    ("REMI+ V2 Tokenization",
     "MidiTok  ·  vocab = 550 tokens\nbeat_res = {(0,4):8, (4,12):4}  ·  programs + time-sigs + tempos",
     C_PROC),

    ("Pitch Shifting Augmentation",
     "×11 shifts,  k ∈ {−5, …, +5} semitones\ndrums excluded  ·  pitches clipped to [21, 108]",
     C_PROC),

    ("Dataset Split",
     "80 % Train   ·   10 % Val   ·   10 % Test\nsong-level split  ·  seed = 42",
     C_SPLIT),

    ("Chunking & Storage",
     "8 bars / chunk  ·  441 k  .pt files\n≈ 14.5 M training chunks",
     C_OUT),
]

arrow_labels = [
    "",
    "80.3 % survival → ~50 k files",
    "×11 augmentation",
    "441 k sequences",
    "stratified by file",
    "",
]

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C_BG)
ax  = fig.add_axes([0, 0, 1, 1], facecolor=C_BG)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# title
ax.text(CX, FIG_H - 0.28, "MIDIT  —  Preprocessing Pipeline",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#222222")

def box_top(i):
    """Y coordinate of the top edge of stage i."""
    return FIG_H - MARGIN_TOP - 0.35 - i * (BOX_H + STAGE_GAP)

for i, (title, sub, colour) in enumerate(stages):
    top   = box_top(i)
    bx    = CX - BOX_W / 2
    by    = top - BOX_H

    # rounded box
    fancy = FancyBboxPatch(
        (bx, by), BOX_W, BOX_H,
        boxstyle="round,pad=0.06",
        linewidth=0, facecolor=colour, zorder=3
    )
    ax.add_patch(fancy)

    # stage number badge
    badge_r = 0.18
    badge_cx = bx + 0.28
    badge_cy = by + BOX_H / 2
    circle = plt.Circle((badge_cx, badge_cy), badge_r,
                         color="white", alpha=0.35, zorder=4)
    ax.add_patch(circle)
    ax.text(badge_cx, badge_cy, str(i + 1),
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="white", zorder=5)

    # title text
    ax.text(CX + 0.1, by + BOX_H * 0.65, title,
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=C_TEXT, zorder=4)

    # sub text
    ax.text(CX + 0.1, by + BOX_H * 0.27, sub,
            ha="center", va="center", fontsize=7.8,
            color="white", alpha=0.92, zorder=4,
            linespacing=1.45)

    # arrow + annotation below this box (not after last)
    if i < len(stages) - 1:
        arr_top = by              # bottom of current box
        arr_bot = box_top(i + 1) # top of next box

        ax.annotate(
            "", xy=(CX, arr_bot), xytext=(CX, arr_top),
            arrowprops=dict(
                arrowstyle="-|>",
                color=C_ARROW,
                lw=1.6,
                mutation_scale=14,
            ),
            zorder=2
        )

        lbl = arrow_labels[i]
        if lbl:
            ax.text(CX + BOX_W / 2 + 0.08,
                    (arr_top + arr_bot) / 2,
                    lbl,
                    ha="left", va="center", fontsize=7.2,
                    color=C_ANNOT, style="italic",
                    linespacing=1.35)

# legend
legend_items = [
    mpatches.Patch(facecolor=C_INPUT,  label="Input"),
    mpatches.Patch(facecolor=C_PROC,   label="Processing"),
    mpatches.Patch(facecolor=C_SPLIT,  label="Split"),
    mpatches.Patch(facecolor=C_OUT,    label="Output"),
]
ax.legend(handles=legend_items, loc="lower center",
          bbox_to_anchor=(0.5, 0.005),
          ncol=4, frameon=False,
          fontsize=8, handlelength=1.2, handletextpad=0.5,
          columnspacing=1.0)

# ── save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "figures", "preprocessing_pipeline.png"
)
out_path = os.path.normpath(out_path)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=C_BG)
print(f"Saved: {out_path}")
plt.close(fig)
