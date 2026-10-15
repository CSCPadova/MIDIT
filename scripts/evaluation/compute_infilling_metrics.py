"""
scripts/evaluation/compute_infilling_metrics.py

Three standard symbolic-music infilling evaluation metrics:

  CP   — Content Preservation
         Average cosine similarity between per-bar pitch chroma vectors
         (original vs infilled) over the masked bar range.
         Range: [0, 1].  Higher is better.
         Reference: Lu & Su (2018).

  GS   — Groove Similarity
         Average Jaccard similarity of onset-position grids at 16th-note
         resolution between corresponding masked bars.
         Range: [0, 1].  Higher is better.
         Reference: Wu & Yang (2020).

  PCHE — Pitch Class Histogram Entropy Difference
         |H(original masked region) − H(infilled masked region)|
         where H is the Shannon entropy (bits) of the 12-class pitch histogram.
         Range: [0, ∞).  Lower is better.
         Reference: Wu & Yang (2020).

Usage (standalone):
    python scripts/evaluation/compute_infilling_metrics.py \\
        --original  outputs/infilling_v2/seed_2/original.mid \\
        --infilled  outputs/infilling_v2/seed_2/reconstructed.mid \\
        --bar_start 3 \\
        --bar_end   5 \\
        [--grid_res 16]

Usage (as importable module):
    from scripts.evaluation.compute_infilling_metrics import compute_metrics
    results = compute_metrics(
        original_midi = "original.mid",
        infilled_midi = "reconstructed.mid",
        bar_start     = 3,   # first masked bar (0-indexed, inclusive)
        bar_end       = 5,   # last  masked bar (0-indexed, inclusive)
        grid_res      = 16,  # grid subdivisions per whole note for GS
    )
    print(results)
    # {'CP': 0.72, 'GS': 0.58, 'PCHE': 0.31,
    #  'n_bars': 3, 'notes_orig': 45, 'notes_inf': 48}

Notes:
  - bar_start and bar_end are both inclusive, 0-indexed from the start of the
    MIDI file (which corresponds to bar 0 of the 8-bar chunk output by infer.py).
  - Bar boundaries are computed from the MIDI time signatures; defaults to 4/4.
  - All tracks (instruments) are pooled together before computing metrics,
    consistent with the multi-track REMI+ setup used in MIDIT.
  - If a bar has no notes in either file, CP and GS default to 0 for that bar
    (conservative — penalises empty infilling).
  - PCHE is computed over all masked bars combined, not per-bar.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import symusic


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _load(path: str | Path) -> symusic.Score:
    return symusic.Score.from_file(str(path))


def _bar_boundaries(score: symusic.Score, max_bars: int = 256) -> List[int]:
    """Return tick positions of bar-start boundaries (length = max_bars + 1).

    Handles time signature changes correctly: each bar's length is determined
    by the time signature in effect at its start tick.
    """
    tpq = score.ticks_per_quarter
    time_sigs = sorted(score.time_signatures, key=lambda ts: ts.time)

    # Fall back to 4/4 if no time signature events are present
    if not time_sigs:
        time_sigs = [_FakeTS(0, 4, 4)]

    boundaries = [0]
    current_tick = 0
    ts_idx = 0

    for _ in range(max_bars):
        # Advance to the time signature that governs the current bar start
        while (ts_idx + 1 < len(time_sigs)
               and time_sigs[ts_idx + 1].time <= current_tick):
            ts_idx += 1
        ts = time_sigs[ts_idx]
        # bar length in ticks = numerator * tpq * (4 / denominator)
        bar_ticks = int(round(ts.numerator * tpq * 4 / ts.denominator))
        current_tick += bar_ticks
        boundaries.append(current_tick)

    return boundaries


class _FakeTS:
    """Minimal stand-in for a symusic TimeSignature object."""
    def __init__(self, time: int, numerator: int, denominator: int):
        self.time = time
        self.numerator = numerator
        self.denominator = denominator


def _notes_in_range(
    score: symusic.Score,
    t_start: int,
    t_end: int,
) -> List:
    """Collect all notes from all tracks whose onset is in [t_start, t_end)."""
    notes = []
    for track in score.tracks:
        for note in track.notes:
            if t_start <= note.time < t_end:
                notes.append(note)
    return notes


# ── Metric primitives ─────────────────────────────────────────────────────────

def _chroma(notes: List, t_start: int, t_end: int) -> np.ndarray:
    """12-dim pitch chroma vector weighted by note duration, L1-normalised."""
    chroma = np.zeros(12, dtype=float)
    for note in notes:
        chroma[note.pitch % 12] += note.duration
    total = chroma.sum()
    if total > 0:
        chroma /= total
    return chroma


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _onset_grid(
    notes: List,
    t_start: int,
    t_end: int,
    grid_ticks: int,
) -> np.ndarray:
    """Binary onset grid at grid_ticks resolution over [t_start, t_end)."""
    bar_len = t_end - t_start
    n_cells = max(1, int(round(bar_len / grid_ticks)))
    grid = np.zeros(n_cells, dtype=bool)
    for note in notes:
        if t_start <= note.time < t_end:
            cell = min(
                int((note.time - t_start) / grid_ticks),
                n_cells - 1,
            )
            grid[cell] = True
    return grid


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard similarity between two boolean arrays (matched to shorter length)."""
    # Align lengths in case bar boundaries differ slightly between files
    length = min(len(a), len(b))
    a, b = a[:length], b[:length]
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0  # both bars empty → identical groove
    return float(np.logical_and(a, b).sum() / union)


def _pitch_histogram(notes: List) -> np.ndarray:
    """12-class pitch histogram, L1-normalised."""
    hist = np.zeros(12, dtype=float)
    for note in notes:
        hist[note.pitch % 12] += 1
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _entropy(hist: np.ndarray) -> float:
    """Shannon entropy in bits."""
    p = hist[hist > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


# ── Per-metric computation ────────────────────────────────────────────────────

def compute_CP(
    orig: symusic.Score,
    inf: symusic.Score,
    bar_range: List[Tuple[int, int]],
) -> Tuple[float, List[float]]:
    """Content Preservation: average per-bar chroma cosine similarity.

    Args:
        orig:      Original MIDI score.
        inf:       Infilled MIDI score.
        bar_range: List of (t_start, t_end) tick pairs for each masked bar.

    Returns:
        (mean_CP, per_bar_values)
    """
    values = []
    for t_start, t_end in bar_range:
        notes_o = _notes_in_range(orig, t_start, t_end)
        notes_i = _notes_in_range(inf,  t_start, t_end)
        chroma_o = _chroma(notes_o, t_start, t_end)
        chroma_i = _chroma(notes_i, t_start, t_end)
        values.append(_cosine(chroma_o, chroma_i))
    mean = float(np.mean(values)) if values else 0.0
    return mean, values


def compute_GS(
    orig: symusic.Score,
    inf: symusic.Score,
    bar_range: List[Tuple[int, int]],
    tpq: int,
    grid_res: int = 16,
) -> Tuple[float, List[float]]:
    """Groove Similarity: average per-bar Jaccard similarity of onset grids.

    Args:
        orig:     Original MIDI score.
        inf:      Infilled MIDI score.
        bar_range: List of (t_start, t_end) tick pairs.
        tpq:      Ticks per quarter note (from the score).
        grid_res: Grid subdivisions per whole note (default 16 = 16th notes).

    Returns:
        (mean_GS, per_bar_values)
    """
    grid_ticks = max(1, int(round(tpq * 4 / grid_res)))
    values = []
    for t_start, t_end in bar_range:
        notes_o = _notes_in_range(orig, t_start, t_end)
        notes_i = _notes_in_range(inf,  t_start, t_end)
        grid_o = _onset_grid(notes_o, t_start, t_end, grid_ticks)
        grid_i = _onset_grid(notes_i, t_start, t_end, grid_ticks)
        values.append(_jaccard(grid_o, grid_i))
    mean = float(np.mean(values)) if values else 0.0
    return mean, values


def compute_PCHE(
    orig: symusic.Score,
    inf: symusic.Score,
    bar_range: List[Tuple[int, int]],
) -> float:
    """Pitch Class Histogram Entropy Difference.

    Computed over all masked bars pooled together (not per-bar).

    Args:
        orig:     Original MIDI score.
        inf:      Infilled MIDI score.
        bar_range: List of (t_start, t_end) tick pairs.

    Returns:
        |H_original − H_infilled|
    """
    all_orig, all_inf = [], []
    for t_start, t_end in bar_range:
        all_orig.extend(_notes_in_range(orig, t_start, t_end))
        all_inf.extend(_notes_in_range(inf,  t_start, t_end))

    hist_o = _pitch_histogram(all_orig)
    hist_i = _pitch_histogram(all_inf)
    return abs(_entropy(hist_o) - _entropy(hist_i))


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_metrics(
    original_midi: str | Path,
    infilled_midi: str | Path,
    bar_start: int,
    bar_end: int,
    grid_res: int = 16,
) -> Dict[str, float]:
    """Compute CP, GS, PCHE for the masked bar range.

    Args:
        original_midi: Path to the original (unmasked) MIDI file.
        infilled_midi: Path to the reconstructed MIDI file.
        bar_start:     First masked bar index, 0-indexed, inclusive.
        bar_end:       Last  masked bar index, 0-indexed, inclusive.
        grid_res:      Grid subdivisions per whole note for GS (default 16).

    Returns:
        dict with keys: CP, GS, PCHE, n_bars, notes_orig, notes_inf,
                        CP_per_bar, GS_per_bar
    """
    orig = _load(original_midi)
    inf  = _load(infilled_midi)

    # Use original score's bar boundaries (infilled has same structure)
    n_masked = bar_end - bar_start + 1
    boundaries = _bar_boundaries(orig, max_bars=max(bar_end + 2, 64))

    if bar_end + 1 >= len(boundaries):
        raise ValueError(
            f"bar_end={bar_end} exceeds detected bar count "
            f"({len(boundaries) - 1} bars in {original_midi})."
        )

    bar_range = [
        (boundaries[b], boundaries[b + 1])
        for b in range(bar_start, bar_end + 1)
    ]

    tpq = orig.ticks_per_quarter

    cp,   cp_pb  = compute_CP(orig, inf, bar_range)
    gs,   gs_pb  = compute_GS(orig, inf, bar_range, tpq, grid_res)
    pche         = compute_PCHE(orig, inf, bar_range)

    # Count notes in masked region for diagnostics
    t_all_start = bar_range[0][0]
    t_all_end   = bar_range[-1][1]
    n_orig = len(_notes_in_range(orig, t_all_start, t_all_end))
    n_inf  = len(_notes_in_range(inf,  t_all_start, t_all_end))

    return {
        'CP':         cp,
        'GS':         gs,
        'PCHE':       pche,
        'n_bars':     n_masked,
        'notes_orig': n_orig,
        'notes_inf':  n_inf,
        'CP_per_bar': cp_pb,
        'GS_per_bar': gs_pb,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_results(results: Dict, bar_start: int, bar_end: int) -> None:
    print()
    print("=" * 56)
    print("  MIDIT Infilling Evaluation Metrics")
    print("=" * 56)
    print(f"  Masked bars : {bar_start}-{bar_end}  ({results['n_bars']} bars)")
    print(f"  Notes orig  : {results['notes_orig']}")
    print(f"  Notes infill: {results['notes_inf']}")
    print("-" * 56)
    print(f"  CP   (^ better) : {results['CP']:.4f}")
    print(f"  GS   (^ better) : {results['GS']:.4f}")
    print(f"  PCHE (v better) : {results['PCHE']:.4f}")
    print("-" * 56)
    print("  Per-bar breakdown:")
    for i, (cp_b, gs_b) in enumerate(zip(results['CP_per_bar'],
                                          results['GS_per_bar'])):
        bar_idx = bar_start + i
        print(f"    bar {bar_idx}:  CP={cp_b:.3f}  GS={gs_b:.3f}")
    print("=" * 56)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute CP / GS / PCHE infilling evaluation metrics.\n\n"
            "CP   (Content Preservation) — chroma cosine similarity, higher is better.\n"
            "GS   (Groove Similarity)    — onset Jaccard similarity,  higher is better.\n"
            "PCHE (Pitch entropy diff)   — |H_orig − H_infill|,        lower  is better.\n\n"
            "References: Lu & Su (2018); Wu & Yang (2020)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--original', required=True,
        help='Path to original (unmasked) MIDI file.',
    )
    parser.add_argument(
        '--infilled', required=True,
        help='Path to reconstructed (infilled) MIDI file.',
    )
    parser.add_argument(
        '--bar_start', type=int, required=True,
        help='First masked bar index, 0-indexed, inclusive.',
    )
    parser.add_argument(
        '--bar_end', type=int, required=True,
        help='Last masked bar index, 0-indexed, inclusive.',
    )
    parser.add_argument(
        '--grid_res', type=int, default=16,
        help='Grid subdivisions per whole note for GS (default: 16 = 16th notes).',
    )
    args = parser.parse_args()

    results = compute_metrics(
        original_midi = args.original,
        infilled_midi = args.infilled,
        bar_start     = args.bar_start,
        bar_end       = args.bar_end,
        grid_res      = args.grid_res,
    )
    _print_results(results, args.bar_start, args.bar_end)


if __name__ == '__main__':
    main()
