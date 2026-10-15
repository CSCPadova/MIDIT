"""
scripts/figures/generate_architecture.py

Publication-quality MIDIT architecture diagram for Master's thesis.
Verified against midit/model/masked_transformer.py and multi_head_output.py.

Architecture facts (code-verified):
  - token_embedding:    nn.Embedding(550, 512)   -- 550 = sum(per-type) + 4 special
  - position_embedding: nn.Embedding(8192, 512)  -- learned
  - LayerNorm after token+pos sum                -- masked_transformer.py:397
  - TransformerEncoderLayer norm_first=True      -- Pre-LN, GELU
  - 12 layers, d_model=512, nhead=8, ff=2048
  - MultiHeadOutput: LN -> Dropout -> 9x Linear(512, vocab_size_per_type)
  - Per-type vocab: Pitch=151, Vel=32, Dur=64, Prog=129, Pos=96,
                    Bar=1, Tempo=32, TimeSig=9, Rest=32

Usage:
    python scripts/figures/generate_architecture.py
    python scripts/figures/generate_architecture.py --output figures/custom.png
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── Palette ────────────────────────────────────────────────────────────────────
C_TOKEN_F  = '#D6EAF8';  C_TOKEN_E  = '#2980B9'
C_MASK_F   = '#FADBD8';  C_MASK_E   = '#E74C3C'
C_EMB_F    = '#E8DAEF';  C_EMB_E    = '#7D3C98'
C_LN_F     = '#FEF9E7';  C_LN_E     = '#F39C12'
C_ENC_F    = '#D5F5E3';  C_ENC_E    = '#1E8449'
C_MHA_F    = '#D6EAF8';  C_MHA_E    = '#2471A3'
C_FFN_F    = '#EBF5FB';  C_FFN_E    = '#1A5276'
C_RES_F    = '#FEF9E7';  C_RES_E    = '#F39C12'
C_PROJ_F   = '#F8F9FA';  C_PROJ_E   = '#566573'
C_DARK     = '#1C2833'
C_GREY     = '#717D7E'

HEAD_PALETTE = [
    ('#FADBD8', '#CB4335'),   # Pitch    - red
    ('#FDEBD0', '#CA6F1E'),   # Velocity - orange
    ('#FEF9E7', '#9A7D0A'),   # Duration - gold
    ('#D5F5E3', '#1E8449'),   # Program  - green
    ('#D6EAF8', '#1A5276'),   # Position - blue
    ('#EBF5FB', '#21618C'),   # Bar      - steel blue
    ('#E8DAEF', '#6C3483'),   # Tempo    - purple
    ('#FDEDEC', '#943126'),   # TimeSig  - dark red
    ('#F2F3F4', '#626567'),   # Rest     - grey
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def rbox(ax, x, y, w, h, fc, ec, lw=1.0, rounding=0.06):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0.0,rounding_size={rounding}',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
    )
    ax.add_patch(p)


def txt(ax, x, y, s, fs=8.5, fw='normal', fc='#1C2833',
        ha='center', va='center', style='normal'):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs,
            fontweight=fw, color=fc, fontstyle=style, zorder=5)


def vline(ax, x, y0, y1, lw=0.9, color='#444444'):
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=9), zorder=3)


# ── Main ────────────────────────────────────────────────────────────────────────

def make_figure(output_path: str = 'figures/midit_architecture.png') -> None:

    FIG_W = 6.5
    FIG_H = 10.2

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    CX = FIG_W / 2   # horizontal center

    # ── 1. INPUT TOKENS ──────────────────────────────────────────────────────
    N_TOK   = 13
    TOK_W   = 0.355
    TOK_H   = 0.42
    TOK_GAP = 0.035
    tok_seq_w = N_TOK * (TOK_W + TOK_GAP) - TOK_GAP
    tok_x0    = (FIG_W - tok_seq_w) / 2
    TOK_Y     = FIG_H - 0.88    # bottom of token row

    # Sequence: note group (normal), masked group, note group, partial masked
    token_defs = [
        ('Bar',  False), ('Pos',  False), ('Prog', False),
        ('Pit',  False), ('Vel',  False), ('Dur',  False),
        ('[M]',  True),  ('[M]',  True),  ('[M]',  True),
        ('Bar',  False), ('Pos',  False),
        ('[M]',  True),  ('[M]',  True),
    ]

    for i, (lbl, masked) in enumerate(token_defs):
        tx = tok_x0 + i * (TOK_W + TOK_GAP)
        fc = C_MASK_F if masked else C_TOKEN_F
        ec = C_MASK_E if masked else C_TOKEN_E
        tc = '#922B21' if masked else '#1B4F72'
        rbox(ax, tx, TOK_Y, TOK_W, TOK_H, fc, ec, lw=0.85)
        txt(ax, tx + TOK_W / 2, TOK_Y + TOK_H / 2, lbl,
            fs=7.0, fw='bold', fc=tc)

    # Section header
    txt(ax, CX, TOK_Y + TOK_H + 0.22,
        'Input: REMI+ Token Sequence',
        fs=10.5, fw='bold', fc=C_DARK)
    txt(ax, CX, TOK_Y + TOK_H + 0.07,
        'token IDs  ∈  { 0, …, 549 }  —  up to 8,192 positions',
        fs=7.8, fc='#555555', style='italic')

    # Legend
    leg_x = tok_x0
    leg_y = TOK_Y - 0.28
    for fc, ec, label_txt in [
        (C_TOKEN_F, C_TOKEN_E, 'normal token'),
        (C_MASK_F,  C_MASK_E,  '[MASK] token'),
    ]:
        rbox(ax, leg_x, leg_y, 0.18, 0.18, fc, ec, lw=0.75)
        txt(ax, leg_x + 0.28, leg_y + 0.09, label_txt,
            fs=7.0, fc=C_GREY, ha='left')
        leg_x += 1.40

    # ── 2. ARROW ─────────────────────────────────────────────────────────────
    A1_top = TOK_Y - 0.38
    A1_bot = A1_top - 0.30
    vline(ax, CX, A1_top, A1_bot)

    # ── 3. EMBEDDINGS ────────────────────────────────────────────────────────
    E_H   = 0.58
    E_W   = 2.05
    E_SEP = 0.22      # gap between the two boxes
    e1_x  = CX - E_W - E_SEP / 2
    e2_x  = CX + E_SEP / 2
    E_Y   = A1_bot - E_H

    rbox(ax, e1_x, E_Y, E_W, E_H, C_EMB_F, C_EMB_E, lw=1.1)
    txt(ax, e1_x + E_W / 2, E_Y + E_H * 0.65,
        'Token Embedding', fs=8.2, fw='bold', fc='#4A235A')
    txt(ax, e1_x + E_W / 2, E_Y + E_H * 0.28,
        '550 × 512', fs=7.5, fc='#6C3483')

    txt(ax, CX, E_Y + E_H / 2, '+', fs=16, fw='bold', fc='#7D3C98')

    rbox(ax, e2_x, E_Y, E_W, E_H, C_EMB_F, C_EMB_E, lw=1.1)
    txt(ax, e2_x + E_W / 2, E_Y + E_H * 0.65,
        'Positional Embedding', fs=8.2, fw='bold', fc='#4A235A')
    txt(ax, e2_x + E_W / 2, E_Y + E_H * 0.28,
        '8192 × 512  (learned)', fs=7.5, fc='#6C3483')

    # ── 4. ARROW + LAYERNORM ─────────────────────────────────────────────────
    A2_top = E_Y
    A2_bot = A2_top - 0.22
    vline(ax, CX, A2_top, A2_bot)

    LN0_H = 0.33
    LN0_Y = A2_bot - LN0_H
    rbox(ax, CX - 1.35, LN0_Y, 2.70, LN0_H, C_LN_F, C_LN_E, lw=0.95)
    txt(ax, CX, LN0_Y + LN0_H / 2,
        'Element-wise Sum  →  LayerNorm', fs=8.2, fw='bold', fc='#7D6608')

    # ── 5. ARROW ─────────────────────────────────────────────────────────────
    A3_top = LN0_Y
    A3_bot = A3_top - 0.28
    vline(ax, CX, A3_top, A3_bot)

    # ── 6. TRANSFORMER ENCODER ───────────────────────────────────────────────
    ENC_W  = 5.30
    ENC_H  = 3.52
    ENC_X  = (FIG_W - ENC_W) / 2
    ENC_Y  = A3_bot - ENC_H     # bottom of encoder box

    rbox(ax, ENC_X, ENC_Y, ENC_W, ENC_H, C_ENC_F, C_ENC_E,
         lw=1.5, rounding=0.10)

    # "×12" badge
    ax.text(
        ENC_X + ENC_W - 0.14, ENC_Y + ENC_H - 0.11, '× 12',
        ha='right', va='top', fontsize=13, fontweight='bold',
        color='#1E8449', zorder=6,
        bbox=dict(boxstyle='round,pad=0.14', facecolor='white',
                  edgecolor='#82B366', linewidth=1.0),
    )

    # Title inside encoder
    txt(ax, ENC_X + ENC_W / 2, ENC_Y + ENC_H - 0.21,
        'Transformer Encoder Layer', fs=9.8, fw='bold', fc='#1E8449')

    # ── Internal sub-blocks (pre-norm order) ─────────────────────────────────
    INN_W  = 3.45
    INN_X  = ENC_X + (ENC_W - INN_W) / 2
    INN_H  = 0.38
    INN_GAP = 0.115

    iy = ENC_Y + ENC_H - 0.50   # top of first inner block

    def inner(y_top, fc, ec, main, sub=None, bold_main=False):
        """Draw one sub-layer box; return y of bottom minus gap."""
        y_b = y_top - INN_H
        rbox(ax, INN_X, y_b, INN_W, INN_H, fc, ec, lw=0.85)
        if sub:
            txt(ax, INN_X + INN_W / 2, y_b + INN_H * 0.65,
                main, fs=7.8, fw='bold' if bold_main else 'normal', fc=C_DARK)
            txt(ax, INN_X + INN_W / 2, y_b + INN_H * 0.28,
                sub, fs=7.0, fc='#444444')
        else:
            txt(ax, INN_X + INN_W / 2, y_b + INN_H / 2,
                main, fs=7.8, fc=C_DARK)
        return y_b - INN_GAP

    def mini_arrow(y_top):
        ax.annotate(
            '', xy=(CX, y_top), xytext=(CX, y_top + INN_GAP),
            arrowprops=dict(arrowstyle='->', color='#666666',
                            lw=0.7, mutation_scale=7), zorder=4,
        )

    iy = inner(iy, C_LN_F, C_LN_E, 'LayerNorm')
    mini_arrow(iy + INN_GAP)
    iy = inner(iy, C_MHA_F, C_MHA_E,
               'Multi-Head Self-Attention',
               'h = 8,  d_model = 512,  d_k = d_v = 64',
               bold_main=True)
    mini_arrow(iy + INN_GAP)
    iy = inner(iy, C_RES_F, C_RES_E, 'Add  (residual connection)')
    mini_arrow(iy + INN_GAP)
    iy = inner(iy, C_LN_F,  C_LN_E,  'LayerNorm')
    mini_arrow(iy + INN_GAP)
    iy = inner(iy, C_FFN_F, C_FFN_E,
               'Feed-Forward Network',
               'dim_ff = 2048,  activation = GELU',
               bold_main=True)
    mini_arrow(iy + INN_GAP)
    inner(iy, C_RES_F, C_RES_E, 'Add  (residual connection)')

    # ── 7. ARROW from encoder ─────────────────────────────────────────────────
    A4_top = ENC_Y
    A4_bot = A4_top - 0.33
    vline(ax, CX, A4_top, A4_bot)
    txt(ax, CX + 0.10, (A4_top + A4_bot) / 2,
        'hidden states  (batch × seq × 512)',
        fs=7.2, fc='#555555', ha='left', style='italic')

    # ── 8. MULTI-HEAD OUTPUT BLOCK ────────────────────────────────────────────
    PROJ_H = 0.36
    PROJ_Y = A4_bot - PROJ_H
    rbox(ax, CX - 1.55, PROJ_Y, 3.10, PROJ_H, C_PROJ_F, C_PROJ_E, lw=1.0)
    txt(ax, CX, PROJ_Y + PROJ_H / 2,
        'Multi-Head Output  —  LayerNorm  +  Dropout (p = 0.1)',
        fs=8.2, fw='bold', fc=C_DARK)

    # ── 9. OUTPUT HEADS ───────────────────────────────────────────────────────
    N_HEADS   = 9
    H_GAP     = 0.038
    H_W       = (FIG_W - 0.38 - (N_HEADS - 1) * H_GAP) / N_HEADS
    H_H       = 0.75
    H_X0      = (FIG_W - (N_HEADS * H_W + (N_HEADS - 1) * H_GAP)) / 2
    H_Y_BOT   = PROJ_Y - 0.62 - H_H

    head_defs = [
        ('Pitch',    '151 classes', HEAD_PALETTE[0]),
        ('Velocity', '32 classes',  HEAD_PALETTE[1]),
        ('Duration', '64 classes',  HEAD_PALETTE[2]),
        ('Program',  '129 classes', HEAD_PALETTE[3]),
        ('Position', '96 classes',  HEAD_PALETTE[4]),
        ('Bar',      '1 class',     HEAD_PALETTE[5]),
        ('Tempo',    '32 classes',  HEAD_PALETTE[6]),
        ('TimeSig',  '9 classes',   HEAD_PALETTE[7]),
        ('Rest',     '32 classes',  HEAD_PALETTE[8]),
    ]

    for i, (name, sub, (fc, ec)) in enumerate(head_defs):
        hx  = H_X0 + i * (H_W + H_GAP)
        hcx = hx + H_W / 2

        # Fan-out line from bottom of projection block
        ax.plot([CX, hcx], [PROJ_Y, H_Y_BOT + H_H + 0.04],
                color='#AAAAAA', lw=0.7, zorder=1)

        rbox(ax, hx, H_Y_BOT, H_W, H_H, fc, ec, lw=0.85)
        txt(ax, hcx, H_Y_BOT + H_H * 0.65, name, fs=7.5, fw='bold', fc=C_DARK)
        txt(ax, hcx, H_Y_BOT + H_H * 0.30, sub, fs=6.5, fc='#444444')

    # "Linear(512, k)" label below heads
    txt(ax, CX, H_Y_BOT - 0.17,
        'Linear(512, k)  per head  —  loss computed at [MASK] positions only',
        fs=7.8, fc='#555555', style='italic')

    # ── Parameter count annotation ─────────────────────────────────────────────
    txt(ax, CX, H_Y_BOT - 0.38,
        '~42.5M parameters  —  trained from scratch on Lakh MIDI (V2, 550-token vocab)',
        fs=7.2, fc='#888888')

    # ── Save ───────────────────────────────────────────────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.10)
    print(f'Saved -> {out.resolve()}')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='figures/midit_architecture.png')
    args = parser.parse_args()
    make_figure(args.output)
