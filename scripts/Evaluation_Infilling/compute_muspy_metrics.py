"""
scripts/Evaluation_Infilling/compute_muspy_metrics.py

Batch evaluation of MIDIT infilling outputs.

For each experiment subdirectory (must contain original.mid, masked.mid,
reconstructed.mid) computes:

  P-I metrics  (pair-wise, reconstructed vs original):
    CP   — Content Preservation     (chroma cosine similarity, higher = better)
    GS   — Groove Similarity        (onset Jaccard,            higher = better)
    PCHE — Pitch entropy difference (|H_orig - H_recon|,       lower  = better)

  S-I metrics  (single-individual, reconstructed only):
    pitch_class_entropy   muspy.pitch_class_entropy()
    groove_consistency    muspy.groove_consistency()
    scale_consistency     muspy.scale_consistency()
    pitch_range           muspy.pitch_range()
    n_pitches_used        muspy.n_pitches_used()

Masked bars are auto-detected from masked.mid vs original.mid: any bar that
has notes in original.mid but zero notes in masked.mid is treated as masked.
If detection fails, metrics are computed over the full sequence.

Requirements:
    pip install muspy pretty_midi symusic numpy

Usage:
    python scripts/Evaluation_Infilling/compute_muspy_metrics.py \\
        --results_dir outputs/infilling/ \\
        --output_csv  outputs/muspy_metrics.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Import P-I metric helpers from sibling module ────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))
from compute_infilling_metrics import (
    _bar_boundaries,
    _load,
    _notes_in_range,
    compute_CP,
    compute_GS,
    compute_PCHE,
)

# ── muspy availability ────────────────────────────────────────────────────────

try:
    import muspy
    MUSPY_AVAILABLE = True
except ImportError:
    MUSPY_AVAILABLE = False
    print("Warning: muspy not found — S-I metrics will be NaN. "
          "Install with: pip install muspy", file=sys.stderr)

# ── Constants ─────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    'experiment',
    'cp', 'gs', 'pche',
    'pitch_class_entropy', 'groove_consistency',
    'scale_consistency', 'pitch_range', 'n_pitches_used',
]

SI_METRIC_NAMES = [
    'pitch_class_entropy', 'groove_consistency',
    'scale_consistency', 'pitch_range', 'n_pitches_used',
]

_NAN_ROW: Dict[str, float] = {k: float('nan') for k in CSV_COLUMNS[1:]}


# ── Masked-bar auto-detection ─────────────────────────────────────────────────

def _detect_masked_bars(
    orig_path: Path,
    masked_path: Path,
    max_bars: int = 64,
) -> Tuple[List[Tuple[int, int]], bool]:
    """Return (bar_range, used_full_fallback).

    Scans up to max_bars bars. A bar is considered masked when original.mid
    has at least one note there and masked.mid has zero notes there.

    Returns bar_range as a list of (t_start, t_end) tick tuples ready for
    compute_CP / compute_GS / compute_PCHE.  If no masked bars are found,
    falls back to the full-sequence range and sets used_full_fallback=True.
    """
    orig   = _load(orig_path)
    masked = _load(masked_path)

    boundaries = _bar_boundaries(orig, max_bars=max_bars)

    detected: List[Tuple[int, int]] = []
    for b in range(len(boundaries) - 1):
        t_start = boundaries[b]
        t_end   = boundaries[b + 1]
        notes_o = _notes_in_range(orig,   t_start, t_end)
        notes_m = _notes_in_range(masked, t_start, t_end)
        if len(notes_o) > 0 and len(notes_m) == 0:
            detected.append((t_start, t_end))

    if detected:
        return detected, False

    # Fallback: full sequence as a single range
    all_notes = _notes_in_range(orig, 0, boundaries[-1])
    if all_notes:
        t_end_full = max(n.time + n.duration for n in all_notes)
    else:
        t_end_full = boundaries[-1]
    return [(0, t_end_full)], True


# ── P-I metrics ───────────────────────────────────────────────────────────────

def _compute_pi_metrics(
    orig_path: Path,
    recon_path: Path,
    bar_range: List[Tuple[int, int]],
) -> Dict[str, float]:
    """Compute CP, GS, PCHE for the given bar_range."""
    orig  = _load(orig_path)
    recon = _load(recon_path)
    tpq   = orig.ticks_per_quarter

    try:
        cp, _ = compute_CP(orig, recon, bar_range)
    except Exception as e:
        print(f"    CP failed: {e}", file=sys.stderr)
        cp = float('nan')

    try:
        gs, _ = compute_GS(orig, recon, bar_range, tpq, grid_res=16)
    except Exception as e:
        print(f"    GS failed: {e}", file=sys.stderr)
        gs = float('nan')

    try:
        pche = compute_PCHE(orig, recon, bar_range)
    except Exception as e:
        print(f"    PCHE failed: {e}", file=sys.stderr)
        pche = float('nan')

    return {'cp': cp, 'gs': gs, 'pche': pche}


# ── S-I metrics ───────────────────────────────────────────────────────────────

def _si_fallback_pretty_midi(path: Path) -> Dict[str, float]:
    """Compute pitch_class_entropy, pitch_range, n_pitches_used via pretty_midi.

    groove_consistency and scale_consistency require muspy internals — set NaN.
    """
    nan = float('nan')
    result = {k: nan for k in SI_METRIC_NAMES}
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(str(path))
        pitches = [
            note.pitch
            for inst in pm.instruments
            for note in inst.notes
        ]
        if not pitches:
            return result

        hist = np.zeros(12, dtype=float)
        for p in pitches:
            hist[p % 12] += 1
        hist /= hist.sum()
        p_nz = hist[hist > 0]
        pce = float(-np.sum(p_nz * np.log2(p_nz)))

        result['pitch_class_entropy'] = pce
        result['pitch_range']         = float(max(pitches) - min(pitches))
        result['n_pitches_used']      = float(len(set(pitches)))
    except Exception as e:
        print(f"    pretty_midi fallback failed: {e}", file=sys.stderr)
    return result


def _crop_music_to_range(
    music: "muspy.Music",
    bar_range: List[Tuple[int, int]],
) -> "muspy.Music":
    """Return a new muspy.Music containing only notes in the masked bar_range.

    Notes are filtered to those whose onset (note.time) falls within any
    (t_start, t_end) pair in bar_range.  All surviving notes are re-timed
    by subtracting bar_range[0][0] so that the cropped region starts at
    tick 0 — required for groove_consistency's measure-grid to be meaningful.

    Duration clipping at t_end is not applied because the five S-I metrics
    used here (SC, GC, PCE, pitch_range, n_pitches_used) only read note.time
    and note.pitch, never note.end.
    """
    t_offset = bar_range[0][0]
    cropped_tracks = []
    for track in music.tracks:
        filtered = [
            copy.copy(n) for n in track.notes
            if any(t_start <= n.time < t_end for (t_start, t_end) in bar_range)
        ]
        for n in filtered:
            n.time -= t_offset
        if filtered:
            cropped_tracks.append(
                muspy.Track(
                    program=track.program,
                    is_drum=track.is_drum,
                    notes=filtered,
                )
            )
    return muspy.Music(resolution=music.resolution, tracks=cropped_tracks)


def _compute_si_metrics(
    path: Path,
    bar_range: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, float]:
    """Compute S-I metrics on the given MIDI file using muspy.

    If bar_range is provided, metrics are computed only over the masked bar
    region (notes cropped and re-timed to start at 0).  If bar_range is None
    the full file is used (original behaviour, kept as fallback).

    Falls back to pretty_midi if muspy.read_midi() raises.
    """
    nan = float('nan')
    result = {k: nan for k in SI_METRIC_NAMES}

    if not MUSPY_AVAILABLE:
        return _si_fallback_pretty_midi(path)

    try:
        music = muspy.read_midi(str(path))
    except Exception as e:
        print(f"    muspy.read_midi failed ({e}), trying pretty_midi fallback",
              file=sys.stderr)
        return _si_fallback_pretty_midi(path)

    if bar_range is not None:
        music = _crop_music_to_range(music, bar_range)

    metric_fns = {
        'pitch_class_entropy': muspy.pitch_class_entropy,
        'scale_consistency':   muspy.scale_consistency,
        'pitch_range':         muspy.pitch_range,
        'n_pitches_used':      muspy.n_pitches_used,
    }

    try:
        result['groove_consistency'] = float(muspy.groove_consistency(music, measure_resolution=4))
    except Exception as e:
        print(f"    groove_consistency failed: {e}", file=sys.stderr)
        result['groove_consistency'] = float('nan')

    for name, fn in metric_fns.items():
        try:
            result[name] = float(fn(music))
        except Exception as e:
            print(f"    {name} failed: {e}", file=sys.stderr)
            result[name] = nan

    return result


# ── Per-experiment processing ─────────────────────────────────────────────────

def process_experiment(subdir: Path) -> Optional[Dict]:
    """Process one experiment directory. Returns a row dict or None on failure."""
    orig_path   = subdir / 'original.mid'
    masked_path = subdir / 'masked.mid'
    recon_path  = subdir / 'reconstructed.mid'

    for p in (orig_path, masked_path, recon_path):
        if not p.exists():
            print(f"  Skipping {subdir.name}: missing {p.name}")
            return None

    print(f"  Processing: {subdir.name}")
    row: Dict[str, float] = {'experiment': subdir.name}
    row.update(_NAN_ROW)

    # ── Detect masked bars ────────────────────────────────────────────────────
    try:
        bar_range, used_fallback = _detect_masked_bars(orig_path, masked_path)
        n_detected = len(bar_range)
        if used_fallback:
            print(f"    No masked bars detected — using full sequence fallback")
        else:
            print(f"    Masked bars detected: {n_detected} bar(s)")
    except Exception as e:
        print(f"    Bar detection failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        bar_range = None

    # ── P-I metrics ───────────────────────────────────────────────────────────
    if bar_range is not None:
        try:
            pi = _compute_pi_metrics(orig_path, recon_path, bar_range)
            row.update(pi)
        except Exception as e:
            print(f"    P-I metrics failed: {e}", file=sys.stderr)
    else:
        print(f"    Skipping P-I metrics (bar detection failed)", file=sys.stderr)

    # ── S-I metrics (cropped to masked bar region when bar_range is available) ──
    try:
        si = _compute_si_metrics(recon_path, bar_range=bar_range)
        row.update(si)
    except Exception as e:
        print(f"    S-I metrics failed: {e}", file=sys.stderr)

    return row


# ── Summary table ─────────────────────────────────────────────────────────────

def _markdown_summary(rows: List[Dict], n_experiments: int) -> str:
    """Return a markdown table of mean ± std for each numeric metric."""
    numeric_cols = CSV_COLUMNS[1:]  # skip 'experiment'
    direction = {
        'cp':                   '^',
        'gs':                   '^',
        'pche':                 'v',
        'pitch_class_entropy':  ' ',
        'groove_consistency':   '^',
        'scale_consistency':    '^',
        'pitch_range':          ' ',
        'n_pitches_used':       ' ',
    }
    display_name = {
        'cp':                   'CP',
        'gs':                   'GS',
        'pche':                 'PCHE',
        'pitch_class_entropy':  'pitch_class_entropy',
        'groove_consistency':   'groove_consistency',
        'scale_consistency':    'scale_consistency',
        'pitch_range':          'pitch_range',
        'n_pitches_used':       'n_pitches_used',
    }

    lines = [
        f"## Infilling Metrics Summary — {n_experiments} experiment(s)\n",
        f"| Metric | Direction | Mean ± Std | Min | Max |",
        f"|--------|-----------|-----------|-----|-----|",
    ]

    for col in numeric_cols:
        values = [
            r[col] for r in rows
            if not (isinstance(r[col], float) and math.isnan(r[col]))
        ]
        if values:
            mean = np.mean(values)
            std  = np.std(values, ddof=0)
            vmin = np.min(values)
            vmax = np.max(values)

            # Format: integers for pitch_range and n_pitches_used
            if col in ('pitch_range', 'n_pitches_used'):
                cell = f"{mean:.1f} ± {std:.1f}"
                minmax = f"{vmin:.0f} | {vmax:.0f}"
            else:
                cell = f"{mean:.4f} ± {std:.4f}"
                minmax = f"{vmin:.4f} | {vmax:.4f}"

            dir_sym = direction[col]
            lines.append(
                f"| {display_name[col]:<22} | {dir_sym:<9} | {cell:<17} | {minmax} |"
            )
        else:
            lines.append(
                f"| {display_name[col]:<22} | {direction[col]:<9} | {'N/A':<17} | N/A | N/A |"
            )

    lines.append("")
    lines.append(
        "_Direction: ^ = higher is better, v = lower is better_"
    )
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch evaluation of MIDIT infilling outputs.\n\n"
            "Expects --results_dir to contain subdirectories, each with:\n"
            "  original.mid, masked.mid, reconstructed.mid\n\n"
            "Computes P-I metrics (CP, GS, PCHE) and S-I metrics "
            "(muspy) per experiment and saves to CSV."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--results_dir', required=True,
        help='Root directory containing experiment subdirectories.',
    )
    parser.add_argument(
        '--output_csv', default='outputs/muspy_metrics.csv',
        help='Path to save results CSV (default: outputs/muspy_metrics.csv).',
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_csv  = Path(args.output_csv)

    if not results_dir.exists():
        print(f"Error: --results_dir '{results_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Collect experiment subdirectories (any subdir, sorted)
    subdirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    if not subdirs:
        print(f"Error: no subdirectories found in '{results_dir}'.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[MIDIT Infilling — Batch Metrics]")
    print(f"  results_dir : {results_dir.resolve()}")
    print(f"  experiments : {len(subdirs)}")
    print(f"  output_csv  : {output_csv.resolve()}")
    print()

    rows: List[Dict] = []
    for subdir in subdirs:
        row = process_experiment(subdir)
        if row is not None:
            rows.append(row)

    if not rows:
        print("\nNo valid experiments found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, float('nan')) for col in CSV_COLUMNS})

    print(f"\nCSV saved to: {output_csv.resolve()}")
    print(f"Experiments written: {len(rows)}")

    # ── Print per-experiment table ────────────────────────────────────────────
    col_w = 22
    header = (
        f"\n{'Experiment':<20} {'CP':>7} {'GS':>7} {'PCHE':>7} "
        f"{'PCE':>7} {'GC':>7} {'SC':>7} {'PR':>7} {'NPU':>7}"
    )
    print(header)
    print("-" * len(header.rstrip()))
    for row in rows:
        def _fmt(v):
            if isinstance(v, float) and math.isnan(v):
                return "  N/A  "
            if isinstance(v, float):
                return f"{v:7.4f}"
            return f"{v:7.0f}"

        print(
            f"{row['experiment']:<20} "
            f"{_fmt(row['cp'])} "
            f"{_fmt(row['gs'])} "
            f"{_fmt(row['pche'])} "
            f"{_fmt(row['pitch_class_entropy'])} "
            f"{_fmt(row['groove_consistency'])} "
            f"{_fmt(row['scale_consistency'])} "
            f"{_fmt(row['pitch_range'])} "
            f"{_fmt(row['n_pitches_used'])}"
        )

    # ── Print markdown summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(_markdown_summary(rows, len(rows)))
    print("=" * 70)


if __name__ == '__main__':
    main()
