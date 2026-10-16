"""
scripts/figures/generate_piano_roll_figure.py

Composite piano roll figure: Original / Masked input / Fine-tuned reconstruction.
Seed 5, 3 masked bars (bars 5-7), fine-tuned MIDIT model.

Verified crop parameters (derived by pixel analysis):
  Y crop:  0 – 960   (bar ruler + full pitch range A3-A5; excludes velocity section)
  X crop:  0 – 1850  (piano keys + full 8-bar timeline)
  Masked:  x=1100 – 1580  (bars 5-7, confirmed from orig vs masked pixel diff)

Usage:
    python scripts/figures/generate_piano_roll_figure.py --input_dir <screenshots_dir>
    python scripts/figures/generate_piano_roll_figure.py --input_dir <screenshots_dir> --output docs/img/custom.png

    <screenshots_dir> must contain Original.png, Masked.png and Recostructed.png.
    Thesis figure (local path at the time):
        --input_dir C:/Users/simoz/OneDrive/Desktop/Image_FL/Seed_5_FineTunedModel_3_bars
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_OUT = Path('docs/img/piano_roll_seed5.png')

# ── Crop parameters (original image coordinates) ─────────────────────────────
Y0, Y1 = 0, 960          # piano roll including bar ruler; no velocity section
X0, X1 = 0, 1850         # full content

# ── Masked region (bars 5-7) in original image x-coordinates ─────────────────
MASK_X0 = 1100
MASK_X1 = 1580

# ── Layout ────────────────────────────────────────────────────────────────────
DPI           = 300
TOTAL_W_PX    = int(6.5 * DPI)   # 1950 px  (6.5-inch thesis width)
LABEL_W       = 260               # left margin for panel labels
PANEL_W       = TOTAL_W_PX - LABEL_W    # 1790 px
PANEL_CROP_W  = X1 - X0          # 1850 px

scale         = PANEL_W / PANEL_CROP_W  # ≈ 0.968
PANEL_H       = int((Y1 - Y0) * scale)  # ≈ 929 px

TOP_MARGIN    = 60    # space above first panel (bracket + label)
GAP           = 20    # gap between panels
BTM_MARGIN    = 24

TOTAL_H = TOP_MARGIN + 3 * PANEL_H + 2 * GAP + BTM_MARGIN

# Masked region in canvas coordinates
MX0 = LABEL_W + int((MASK_X0 - X0) * scale)
MX1 = LABEL_W + int((MASK_X1 - X0) * scale)

# ── Font loading ──────────────────────────────────────────────────────────────
def load_font(path, size, fallback_size=None):
    try:
        return ImageFont.truetype(str(path), size)
    except Exception:
        pass
    # Try DejaVu (bundled with many Python installs)
    for dv in [
        'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/calibri.ttf',
    ]:
        try:
            return ImageFont.truetype(dv, size)
        except Exception:
            pass
    return ImageFont.load_default()


def make_figure(input_dir: Path, output_path: Path = DEFAULT_OUT) -> None:

    # ── Canvas ────────────────────────────────────────────────────────────────
    canvas = Image.new('RGB', (TOTAL_W_PX, TOTAL_H), (255, 255, 255))

    # ── Fonts (PIL size = pixels; at 300 DPI: 10pt ≈ 42px) ───────────────────
    fnt_label = load_font('C:/Windows/Fonts/arialbd.ttf', 34)
    fnt_annot = load_font('C:/Windows/Fonts/arial.ttf',   34)

    # ── Load, crop, scale and paste each panel ────────────────────────────────
    filenames = ['Original', 'Masked', 'Recostructed']
    labels    = ['Original', 'Masked input', 'Fine-tuned\nreconstruction']

    panel_tops = []
    for i, (fname, lbl) in enumerate(zip(filenames, labels)):
        img  = Image.open(input_dir / f'{fname}.png').convert('RGB')
        crop = img.crop((X0, Y0, X1, Y1))
        panel = crop.resize((PANEL_W, PANEL_H), Image.LANCZOS)

        py = TOP_MARGIN + i * (PANEL_H + GAP)
        panel_tops.append(py)
        canvas.paste(panel, (LABEL_W, py))

        # Thin border around panel
        draw_tmp = ImageDraw.Draw(canvas)
        draw_tmp.rectangle(
            [LABEL_W, py, LABEL_W + PANEL_W - 1, py + PANEL_H - 1],
            outline=(160, 160, 160), width=1,
        )

    # ── Red semi-transparent overlay (masked region) ──────────────────────────
    overlay = Image.new('RGBA', (TOTAL_W_PX, TOTAL_H), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)

    RED_FILL = (210, 30, 30, 52)    # alpha=52/255 ≈ 20%
    for py in panel_tops:
        ov_draw.rectangle([MX0, py, MX1, py + PANEL_H], fill=RED_FILL)

    canvas = Image.alpha_composite(canvas.convert('RGBA'), overlay).convert('RGB')

    # ── Dashed vertical lines ─────────────────────────────────────────────────
    draw = ImageDraw.Draw(canvas)
    DASH, DGAP = 14, 9
    RED_DASH   = (195, 20, 20)

    for x in (MX0, MX1):
        y = TOP_MARGIN
        y_end = panel_tops[-1] + PANEL_H
        while y < y_end:
            draw.line([(x, y), (x, min(y + DASH, y_end))],
                      fill=RED_DASH, width=2)
            y += DASH + DGAP

    # ── Bracket above first panel ─────────────────────────────────────────────
    BKT_Y   = TOP_MARGIN - 12
    TICK_H  = 10
    LBL_Y   = BKT_Y - TICK_H - 36
    mid_x   = (MX0 + MX1) // 2
    RED_BKT = (190, 20, 20)

    draw.line([(MX0, BKT_Y), (MX1, BKT_Y)], fill=RED_BKT, width=2)
    draw.line([(MX0, BKT_Y - TICK_H), (MX0, BKT_Y)], fill=RED_BKT, width=2)
    draw.line([(MX1, BKT_Y - TICK_H), (MX1, BKT_Y)], fill=RED_BKT, width=2)

    annot_txt = 'masked bars (3)'
    bbox = draw.textbbox((0, 0), annot_txt, font=fnt_annot)
    tw = bbox[2] - bbox[0]
    draw.text((mid_x - tw // 2, LBL_Y), annot_txt,
              font=fnt_annot, fill=RED_BKT)

    # ── Panel labels (left margin) ─────────────────────────────────────────────
    for py, lbl in zip(panel_tops, labels):
        lines = lbl.split('\n')
        line_h = 46   # pixels per line
        block_h = len(lines) * line_h
        y_start = py + (PANEL_H - block_h) // 2
        for j, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=fnt_label)
            lw = bbox[2] - bbox[0]
            x_lbl = (LABEL_W - lw) // 2   # center in label margin
            draw.text((x_lbl, y_start + j * line_h), line,
                      font=fnt_label, fill=(20, 20, 20))

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(output_path), dpi=(DPI, DPI))
    w, h = canvas.size
    print(f'Saved -> {output_path.resolve()}')
    print(f'Size : {w} × {h} px  =  {w/DPI:.2f} × {h/DPI:.2f} in  at {DPI} DPI')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True,
                        help='Directory with Original.png, Masked.png and Recostructed.png.')
    parser.add_argument('--output', default=str(DEFAULT_OUT))
    args = parser.parse_args()
    make_figure(Path(args.input_dir), Path(args.output))
