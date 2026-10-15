"""
scripts/evaluation/merge_metrics.py

Merges per-seed CSVs produced by compute_muspy_metrics.py into a single
summary table and generates a LaTeX table for the thesis.

Expected input directory layout:
  metrics_dir/
    finetuned_1bars.csv
    finetuned_2bars.csv
    ...
    finetuned_5bars.csv
    zeroshot_1bars.csv
    ...
    zeroshot_5bars.csv

Each CSV has columns (from compute_muspy_metrics.py):
  experiment, cp, gs, pche, pitch_class_entropy, groove_consistency,
  scale_consistency, pitch_range, n_pitches_used

Output A — merged_metrics.csv:
  model, n_bars,
  cp_mean, cp_std, gs_mean, gs_std, pche_mean, pche_std,
  pce_mean, pce_std, sc_mean, sc_std,
  pr_mean, n_pitches_mean,
  gc_mean, gc_std   (groove_consistency; NaN if unavailable)

Output B — LaTeX table printed to stdout:
  Columns: CP, GS, PCHE, SC (4 most thesis-relevant)
  Rows: n_bars 1-5
  Side-by-side: Fine-tuned | Zero-shot
  Bold: best value per column per model

NOTE on groove_consistency:
  muspy.groove_consistency() should be called with measure_resolution=4.
  If the input CSVs were generated without this argument, gc values may be
  unreliable. To fix, update compute_muspy_metrics.py:
      muspy.groove_consistency(music)
      → muspy.groove_consistency(music, measure_resolution=4)
  then rerun that script before re-merging.

Usage:
  python scripts/evaluation/merge_metrics.py \\
      --metrics_dir /path/to/metrics/ \\
      --output_csv  /path/to/merged_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

MODELS  = ['finetuned', 'zeroshot']
N_BARS  = [1, 2, 3, 4, 5]

# Input CSV columns that carry numeric metrics
INPUT_METRIC_COLS = [
    'cp', 'gs', 'pche',
    'pitch_class_entropy', 'groove_consistency',
    'scale_consistency', 'pitch_range', 'n_pitches_used',
]

# Merged CSV column order
MERGED_COLS = [
    'model', 'n_bars',
    'cp_mean',   'cp_std',
    'gs_mean',   'gs_std',
    'pche_mean', 'pche_std',
    'pce_mean',  'pce_std',
    'sc_mean',   'sc_std',
    'pr_mean',   'n_pitches_mean',
    'gc_mean',   'gc_std',
]

# LaTeX table: (mean_col, std_col, header, direction)
LATEX_COLS: List[Tuple[str, str, str, str]] = [
    ('cp_mean',   'cp_std',   r'CP$\uparrow$',     'higher'),
    ('gs_mean',   'gs_std',   r'GS$\uparrow$',     'higher'),
    ('pche_mean', 'pche_std', r'PCHE$\downarrow$', 'lower'),
    ('sc_mean',   'sc_std',   r'SC$\uparrow$',     'higher'),
]


# ── Filename parsing ──────────────────────────────────────────────────────────

_FNAME_RE = re.compile(r'^(finetuned|zeroshot)_(\d+)bars\.csv$', re.IGNORECASE)


def parse_filename(path: Path) -> Optional[Tuple[str, int]]:
    """Return (model, n_bars) from filename, or None if it doesn't match."""
    m = _FNAME_RE.match(path.name)
    if m is None:
        return None
    return m.group(1).lower(), int(m.group(2))


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_csv(path: Path) -> List[Dict[str, float]]:
    """Load a per-seed metrics CSV into a list of dicts with float values."""
    rows: List[Dict[str, float]] = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for line in reader:
            row: Dict[str, float] = {}
            for col in INPUT_METRIC_COLS:
                raw = line.get(col, '').strip()
                try:
                    val = float(raw)
                    row[col] = float('nan') if math.isnan(val) else val
                except (ValueError, TypeError):
                    row[col] = float('nan')
            rows.append(row)
    return rows


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute mean and std (ddof=0, ignoring NaN) for each metric column."""
    result: Dict[str, float] = {}
    for col in INPUT_METRIC_COLS:
        values = np.array([r[col] for r in rows], dtype=float)
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            result[f'{col}_mean'] = float('nan')
            result[f'{col}_std']  = float('nan')
        else:
            result[f'{col}_mean'] = float(np.mean(valid))
            result[f'{col}_std']  = float(np.std(valid, ddof=0))
    return result


def _build_merged_row(model: str, n_bars: int, agg: Dict[str, float]) -> Dict:
    """Map aggregated stats to the merged CSV column names."""
    return {
        'model':         model,
        'n_bars':        n_bars,
        'cp_mean':       agg.get('cp_mean',                   float('nan')),
        'cp_std':        agg.get('cp_std',                    float('nan')),
        'gs_mean':       agg.get('gs_mean',                   float('nan')),
        'gs_std':        agg.get('gs_std',                    float('nan')),
        'pche_mean':     agg.get('pche_mean',                 float('nan')),
        'pche_std':      agg.get('pche_std',                  float('nan')),
        'pce_mean':      agg.get('pitch_class_entropy_mean',  float('nan')),
        'pce_std':       agg.get('pitch_class_entropy_std',   float('nan')),
        'sc_mean':       agg.get('scale_consistency_mean',    float('nan')),
        'sc_std':        agg.get('scale_consistency_std',     float('nan')),
        'pr_mean':       agg.get('pitch_range_mean',          float('nan')),
        'n_pitches_mean':agg.get('n_pitches_used_mean',       float('nan')),
        'gc_mean':       agg.get('groove_consistency_mean',   float('nan')),
        'gc_std':        agg.get('groove_consistency_std',    float('nan')),
    }


# ── groove_consistency diagnostics ───────────────────────────────────────────

def _gc_diagnostic(merged_rows: List[Dict]) -> str:
    """Return a diagnostic string about groove_consistency reliability."""
    gc_vals = [
        r['gc_mean'] for r in merged_rows
        if not (isinstance(r['gc_mean'], float) and math.isnan(r['gc_mean']))
    ]
    lines = [
        "",
        "=" * 70,
        "NOTE: groove_consistency (gc) diagnostic",
        "-" * 70,
    ]
    if not gc_vals:
        lines += [
            "  All gc values are NaN — groove_consistency was never computed",
            "  or always failed. The gc_mean/gc_std columns will be NaN.",
        ]
    elif len(set(round(v, 6) for v in gc_vals)) == 1:
        lines += [
            f"  WARNING: all non-NaN gc values are identical ({gc_vals[0]:.6f}).",
            "  This strongly suggests groove_consistency was computed with the",
            "  default measure_resolution, which may be incorrect for this data.",
        ]
    else:
        lines += [
            f"  gc values look varied (range: {min(gc_vals):.4f} – {max(gc_vals):.4f}).",
            "  Values appear plausible.",
        ]
    lines += [
        "",
        "  To apply the recommended fix, update compute_muspy_metrics.py:",
        "    muspy.groove_consistency(music)",
        "    → muspy.groove_consistency(music, measure_resolution=4)",
        "  then rerun compute_muspy_metrics.py and re-merge.",
        "=" * 70,
        "",
    ]
    return "\n".join(lines)


# ── LaTeX table ───────────────────────────────────────────────────────────────

def _fmt_cell(mean: float, std: float, bold: bool) -> str:
    """Format a mean ± std cell for LaTeX. Uses math mode."""
    nan = float('nan')
    if math.isnan(mean) or math.isnan(std):
        return '--'
    content = f'{mean:.4f} \\pm {std:.4f}'
    if bold:
        return f'$\\mathbf{{{content}}}$'
    return f'${content}$'


def _find_best(merged: Dict[Tuple[str, int], Dict],
               model: str,
               mean_col: str,
               direction: str) -> Optional[int]:
    """Return the n_bars with the best mean value for (model, mean_col)."""
    candidates = {
        nb: merged[(model, nb)][mean_col]
        for nb in N_BARS
        if (model, nb) in merged
        and not math.isnan(merged[(model, nb)][mean_col])
    }
    if not candidates:
        return None
    if direction == 'higher':
        return max(candidates, key=candidates.__getitem__)
    return min(candidates, key=candidates.__getitem__)


def build_latex_table(merged: Dict[Tuple[str, int], Dict]) -> str:
    """Return the LaTeX table string."""

    n_latex_cols = len(LATEX_COLS)

    # Column spec: c | c c c c | c c c c
    col_spec = 'c | ' + ' '.join(['c'] * n_latex_cols) + \
               ' | ' + ' '.join(['c'] * n_latex_cols)

    # Header row
    headers = ' & '.join(lc[2] for lc in LATEX_COLS)

    # Pre-compute best n_bars per (model, metric)
    best: Dict[Tuple[str, str], Optional[int]] = {}
    for model in MODELS:
        for mean_col, _, _, direction in LATEX_COLS:
            best[(model, mean_col)] = _find_best(merged, model, mean_col, direction)

    model_display = {'finetuned': 'Fine-tuned', 'zeroshot': 'Zero-shot'}

    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\caption{Infilling evaluation metrics: fine-tuned vs.\ zero-shot '
        r'(mean $\pm$ std across seeds). '
        r'CP: content preservation; GS: groove similarity; '
        r'PCHE: pitch entropy difference; SC: scale consistency. '
        r'Best per column per model in \textbf{bold}.}',
        r'\label{tab:infilling_metrics}',
        r'\begin{tabular}{' + col_spec + r'}',
        r'\toprule',
        r'& \multicolumn{' + str(n_latex_cols) + r'}{c|}{'
        + model_display['finetuned']
        + r'} & \multicolumn{' + str(n_latex_cols) + r'}{c}{'
        + model_display['zeroshot']
        + r'} \\',
        r'$n_\text{bars}$ & ' + headers + ' & ' + headers + r' \\',
        r'\midrule',
    ]

    for nb in N_BARS:
        cells = [str(nb)]
        for model in MODELS:
            row = merged.get((model, nb))
            for mean_col, std_col, _, _ in LATEX_COLS:
                if row is None:
                    cells.append('--')
                else:
                    is_best = (best[(model, mean_col)] == nb)
                    cells.append(_fmt_cell(row[mean_col], row[std_col], is_best))
        lines.append(' & '.join(cells) + r' \\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    return '\n'.join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-seed infilling metric CSVs into a summary table "
            "and generate a LaTeX table for the thesis."
        )
    )
    parser.add_argument(
        '--metrics_dir', required=True,
        help='Directory containing finetuned_Nbars.csv / zeroshot_Nbars.csv files.',
    )
    parser.add_argument(
        '--output_csv', default='merged_metrics.csv',
        help='Output path for merged_metrics.csv.',
    )
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_csv  = Path(args.output_csv)

    if not metrics_dir.exists():
        print(f"Error: --metrics_dir '{metrics_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # ── Discover and load input CSVs ──────────────────────────────────────────
    print(f"\n[merge_metrics] Scanning: {metrics_dir.resolve()}")
    csv_paths = sorted(metrics_dir.glob('*.csv'))

    loaded: Dict[Tuple[str, int], List[Dict[str, float]]] = {}
    skipped = []

    for path in csv_paths:
        parsed = parse_filename(path)
        if parsed is None:
            skipped.append(path.name)
            continue
        model, n_bars = parsed
        rows = load_csv(path)
        if not rows:
            print(f"  Warning: {path.name} is empty — skipping.")
            continue
        loaded[(model, n_bars)] = rows
        print(f"  Loaded: {path.name}  ({len(rows)} seed(s))")

    if skipped:
        print(f"  Skipped (no match): {', '.join(skipped)}")

    if not loaded:
        print("\nError: no valid CSVs found.", file=sys.stderr)
        sys.exit(1)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    merged: Dict[Tuple[str, int], Dict] = {}
    merged_rows: List[Dict] = []

    for (model, n_bars) in sorted(loaded.keys()):
        agg = aggregate(loaded[(model, n_bars)])
        row = _build_merged_row(model, n_bars, agg)
        merged[(model, n_bars)] = row
        merged_rows.append(row)

    # ── Write merged CSV ──────────────────────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=MERGED_COLS)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({
                col: '' if (isinstance(row.get(col), float) and math.isnan(row[col]))
                     else row.get(col, '')
                for col in MERGED_COLS
            })
    print(f"\nMerged CSV saved: {output_csv.resolve()}")

    # ── Print summary table ───────────────────────────────────────────────────
    _COL_W = 18
    hdr_metrics = ['cp', 'gs', 'pche', 'sc', 'pr', 'n_pitches']
    print(f"\n{'Model':<12} {'n_bars':>6}  "
          + '  '.join(f'{c:>{_COL_W}}' for c in hdr_metrics))
    print('-' * (12 + 8 + ((_COL_W + 2) * len(hdr_metrics))))

    def _s(mean_key, std_key, row):
        m = row.get(mean_key, float('nan'))
        s = row.get(std_key, float('nan'))
        if math.isnan(m):
            return f"{'N/A':>{_COL_W}}"
        if math.isnan(s):
            return f"{m:>{_COL_W}.4f}"
        return f"{m:.4f}±{s:.4f}".rjust(_COL_W)

    for (model, nb), row in sorted(merged.items()):
        pr_str  = f"{row['pr_mean']:.1f}".rjust(_COL_W) if not math.isnan(row['pr_mean']) else f"{'N/A':>{_COL_W}}"
        npu_str = f"{row['n_pitches_mean']:.1f}".rjust(_COL_W) if not math.isnan(row['n_pitches_mean']) else f"{'N/A':>{_COL_W}}"
        print(
            f"{model:<12} {nb:>6}  "
            f"{_s('cp_mean',   'cp_std',   row)}  "
            f"{_s('gs_mean',   'gs_std',   row)}  "
            f"{_s('pche_mean', 'pche_std', row)}  "
            f"{_s('sc_mean',   'sc_std',   row)}  "
            f"{pr_str}  {npu_str}"
        )

    # ── groove_consistency diagnostic ─────────────────────────────────────────
    print(_gc_diagnostic(merged_rows))

    # ── LaTeX table ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("LaTeX table (copy into thesis):")
    print("=" * 70)
    print()
    print(build_latex_table(merged))
    print()


if __name__ == '__main__':
    main()
