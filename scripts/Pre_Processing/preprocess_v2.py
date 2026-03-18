"""
preprocess_v2.py -- MIDIT Phase 1 Preprocessing (V2)
=====================================================
Tokenizes the full Lakh MIDI dataset (lmd_full) with the V2 tokenizer config,
applies quality filters, generates pitch-shifting augmentation, and saves
.pt files compatible with the existing lakh_dataset.py loader.

Output .pt format (same as preprocess_batch_improved.py):
    {
      'tokens':        torch.Tensor([N], dtype=torch.long),
      'bar_positions': list[int],   # ascending, [0 ... len(tokens)]
      'metadata':      dict,
    }

V2 tokenizer changes vs V1:
    beat_res: {(0,4):8, (4,12):4}   (+32 Duration tokens, vocab=550)
    delete_equal_successive_tempo_changes=True
    delete_equal_successive_time_sig_changes=True

Quality filters (applied before tokenizing, fast miditoolkit check):
    F1: note count < 50         -> skip
    F2: duration < 15 seconds   -> skip
    F3: top-1 velocity > 90%    -> skip

Pitch augmentation: 11 shifts per file (-5 ... +5 semitones)
    Notes clipped to [21, 108], drum tracks unchanged.

Usage:
    # Dry run (filter stats only, nothing saved):
    python scripts/Pre_Processing/preprocess_v2.py --dry_run

    # Test on 10 files:
    python scripts/Pre_Processing/preprocess_v2.py --max_files 10 --dry_run

    # Full run on cluster:
    python scripts/Pre_Processing/preprocess_v2.py \\
        --output_dir /nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2 \\
        --num_workers 32

Run from MIDIT project root.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import tempfile
import time
from collections import Counter
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import torch

# ── Constants ─────────────────────────────────────────────────────────────────

DATASET_DIR = Path("data/midi_files/lmd_full")

SHIFTS    = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
PITCH_MIN = 21
PITCH_MAX = 108

# Quality filter thresholds
F1_MIN_NOTES      = 50
F2_MIN_DURATION_S = 15.0
F3_MAX_VEL_PCT    = 0.90

PROGRESS_INTERVAL = 100


# ── Worker globals (one per worker process) ───────────────────────────────────

_tokenizer   = None
_id_to_token = None


def _init_worker():
    """Initialize tokenizer once per worker process (V2 config)."""
    global _tokenizer, _id_to_token
    from miditok import REMI, TokenizerConfig

    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 8, (4, 12): 4},   # V2: extended range
        num_velocities=32,
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_programs=True,
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
        use_controls=False,
        delete_equal_successive_tempo_changes=True,
        delete_equal_successive_time_sig_changes=True,
    )
    _tokenizer   = REMI(config)
    _id_to_token = {v: k for k, v in _tokenizer.vocab.items()}


# ── Per-file worker ───────────────────────────────────────────────────────────

def _process_file(args: tuple) -> dict:
    """
    Process one MIDI file: apply quality filters, tokenize with pitch shifting.

    Args:
        args: (midi_path, output_dir, shifts, dry_run)

    Returns:
        dict with keys:
            path       : str
            status     : 'ok' | 'skipped_f1' | 'skipped_f2' | 'skipped_f3'
                       | 'skipped_unreadable' | 'error'
            pt_count   : int  (number of .pt files written, 0 in dry_run)
            error      : str | None
    """
    midi_path, output_dir, shifts, dry_run = args

    try:
        import miditoolkit
        from collections import Counter as _Counter

        # ── Load MIDI ─────────────────────────────────────────────────────────
        try:
            midi = miditoolkit.MidiFile(str(midi_path))
        except Exception as e:
            return {
                "path": str(midi_path),
                "status": "skipped_unreadable",
                "pt_count": 0,
                "error": str(e),
            }

        tpq = midi.ticks_per_beat
        bpm = float(midi.tempo_changes[0].tempo) if midi.tempo_changes else 120.0

        all_notes = [n for inst in midi.instruments for n in inst.notes]

        # ── Quality filters ────────────────────────────────────────────────────

        # F1: too sparse
        if len(all_notes) < F1_MIN_NOTES:
            return {"path": str(midi_path), "status": "skipped_f1", "pt_count": 0, "error": None}

        # F2: too short
        if all_notes:
            max_tick     = max(n.end for n in all_notes)
            duration_sec = (max_tick / tpq) * 60.0 / bpm
        else:
            duration_sec = 0.0

        if duration_sec < F2_MIN_DURATION_S:
            return {"path": str(midi_path), "status": "skipped_f2", "pt_count": 0, "error": None}

        # F3: near-constant velocity
        vel_counts  = _Counter(n.velocity for n in all_notes)
        top_vel_pct = max(vel_counts.values()) / len(all_notes) if all_notes else 0.0
        if top_vel_pct > F3_MAX_VEL_PCT:
            return {"path": str(midi_path), "status": "skipped_f3", "pt_count": 0, "error": None}

        # ── Dry run: filters passed, nothing to save ───────────────────────────
        if dry_run:
            return {"path": str(midi_path), "status": "ok", "pt_count": 0, "error": None}

        # ── Tokenize each shift and save .pt ──────────────────────────────────
        stem     = Path(midi_path).stem
        pt_count = 0
        errors   = []

        for shift in shifts:
            try:
                # Build shifted MIDI (deepcopy, then clip non-drum pitches)
                if shift == 0:
                    midi_to_use = midi
                else:
                    midi_to_use = copy.deepcopy(midi)
                    for inst in midi_to_use.instruments:
                        if not inst.is_drum:
                            for note in inst.notes:
                                note.pitch = max(PITCH_MIN, min(PITCH_MAX, note.pitch + shift))

                # Save to temp file, tokenize, delete temp
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mid")
                try:
                    os.close(tmp_fd)
                    midi_to_use.dump(tmp_path)

                    tokens = _tokenizer(tmp_path)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

                # Flatten token IDs
                if isinstance(tokens, list):
                    all_ids: list[int] = []
                    for seq in tokens:
                        all_ids.extend(seq.ids)
                else:
                    all_ids = tokens.ids

                if len(all_ids) < 10:
                    continue  # empty / degenerate result

                # Build bar_positions (same logic as preprocess_batch_improved.py)
                bar_positions: list[int] = []
                for idx, tok_id in enumerate(all_ids):
                    tok_name = _id_to_token.get(tok_id)
                    if tok_name and "Bar" in tok_name:
                        bar_positions.append(idx)

                if not bar_positions or bar_positions[0] != 0:
                    bar_positions.insert(0, 0)
                bar_positions.append(len(all_ids))

                num_bars = len(bar_positions) - 1

                # Build metadata
                metadata = {
                    "source_file":   Path(midi_path).name,
                    "source_path":   str(midi_path),
                    "num_tokens":    len(all_ids),
                    "num_bars":      num_bars,
                    "pitch_shift":   shift,
                    "tokenizer_ver": "v2",
                }

                # Output filename
                if shift == 0:
                    out_name = f"{stem}.pt"
                elif shift > 0:
                    out_name = f"{stem}_shift+{shift}.pt"
                else:
                    out_name = f"{stem}_shift{shift}.pt"

                data = {
                    "tokens":        torch.tensor(all_ids, dtype=torch.long),
                    "bar_positions": bar_positions,
                    "metadata":      metadata,
                }
                torch.save(data, Path(output_dir) / out_name)
                pt_count += 1

            except Exception as e:
                errors.append(f"shift={shift}: {e}")

        if errors:
            # Partial success is still 'ok'; errors are logged separately
            return {
                "path": str(midi_path),
                "status": "ok",
                "pt_count": pt_count,
                "error": "; ".join(errors),
            }

        return {"path": str(midi_path), "status": "ok", "pt_count": pt_count, "error": None}

    except Exception as e:
        return {
            "path": str(midi_path),
            "status": "error",
            "pt_count": 0,
            "error": str(e),
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIDIT V2 Preprocessing — tokenize + filter + augment Lakh MIDI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="data/preprocessed_v2",
        help="Directory to save .pt files",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--max_files", type=int, default=0,
        help="Process at most N files (0 = all). Use for local testing.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Apply filters but do not tokenize or save anything",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for --max_files sampling",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    dry_run    = args.dry_run
    t0         = time.time()

    # ── Header ────────────────────────────────────────────────────────────────
    print("=" * 72)
    print("MIDIT V2 PREPROCESSING")
    print(f"  Dataset    : {DATASET_DIR.resolve()}")
    print(f"  Output dir : {output_dir.resolve()}")
    print(f"  Workers    : {args.num_workers}")
    print(f"  Shifts     : {SHIFTS}  ({len(SHIFTS)} per file)")
    print(f"  Dry run    : {dry_run}")
    print(f"  Max files  : {args.max_files if args.max_files > 0 else 'all'}")
    print(f"  Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)
    print()

    # ── Discover files ────────────────────────────────────────────────────────
    print("Scanning dataset directory...")
    all_files  = sorted(DATASET_DIR.rglob("*.mid"))
    total_disk = len(all_files)
    print(f"  Found {total_disk:,} .mid files")

    if args.max_files > 0:
        rng    = random.Random(args.seed)
        sample = rng.sample(all_files, min(args.max_files, total_disk))
        print(f"  Sampling {len(sample)} files (seed={args.seed})")
    else:
        sample = all_files

    total = len(sample)
    print(f"  Processing {total:,} files  ->  up to {total * len(SHIFTS):,} .pt files")
    print()

    # ── Create output directory ────────────────────────────────────────────────
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output directory ready: {output_dir}")
        print()

    # ── Build args for workers ─────────────────────────────────────────────────
    args_list = [
        (str(midi_path), str(output_dir), SHIFTS, dry_run)
        for midi_path in sample
    ]

    # ── Process ───────────────────────────────────────────────────────────────
    print(f"Processing with {args.num_workers} workers...")
    print(f"  (progress every {PROGRESS_INTERVAL} files)")
    print()

    ok_count        = 0
    skip_f1         = 0
    skip_f2         = 0
    skip_f3         = 0
    skip_unreadable = 0
    error_count     = 0
    pt_created      = 0
    failed_paths: list[str] = []

    with Pool(args.num_workers, initializer=_init_worker) as pool:
        for i, result in enumerate(
            pool.imap_unordered(_process_file, args_list), 1
        ):
            status = result.get("status", "error")

            if status == "ok":
                ok_count += 1
                pt_created += result.get("pt_count", 0)
                if result.get("error"):
                    # Partial errors (some shifts failed)
                    failed_paths.append(
                        f"{result['path']}  [partial: {result['error']}]"
                    )
            elif status == "skipped_f1":
                skip_f1 += 1
            elif status == "skipped_f2":
                skip_f2 += 1
            elif status == "skipped_f3":
                skip_f3 += 1
            elif status == "skipped_unreadable":
                skip_unreadable += 1
                failed_paths.append(
                    f"{result['path']}  [unreadable: {result.get('error','')}]"
                )
            else:
                error_count += 1
                failed_paths.append(
                    f"{result['path']}  [error: {result.get('error','')}]"
                )

            if i % PROGRESS_INTERVAL == 0 or i == total:
                elapsed  = time.time() - t0
                rate     = i / elapsed if elapsed > 0 else 0
                eta_s    = (total - i) / rate if rate > 0 else 0
                skipped  = skip_f1 + skip_f2 + skip_f3 + skip_unreadable
                print(
                    f"  [{i:>7,}/{total:,}]  "
                    f"ok={ok_count:,}  "
                    f"skip={skipped:,}  "
                    f"err={error_count:,}  "
                    f"pt={pt_created:,}  "
                    f"elapsed={elapsed:.0f}s  "
                    f"eta={eta_s:.0f}s"
                )

    total_elapsed = time.time() - t0
    total_skipped = skip_f1 + skip_f2 + skip_f3 + skip_unreadable

    # ── Save failed_files.txt ─────────────────────────────────────────────────
    if not dry_run and failed_paths:
        failed_log = output_dir / "failed_files.txt"
        with open(failed_log, "w", encoding="utf-8") as f:
            f.write(f"# Failed / partial files from preprocess_v2.py\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Total: {len(failed_paths)}\n\n")
            for line in failed_paths:
                f.write(line + "\n")
        print(f"\n  Failed/partial log: {failed_log}  ({len(failed_paths)} entries)")

    # ── Save preprocessing_stats.json ─────────────────────────────────────────
    stats = {
        "timestamp":              datetime.now().isoformat(),
        "dry_run":                dry_run,
        "total_files_on_disk":    total_disk,
        "files_submitted":        total,
        "results": {
            "ok":                 ok_count,
            "skipped_f1_sparse":  skip_f1,
            "skipped_f2_short":   skip_f2,
            "skipped_f3_vel":     skip_f3,
            "skipped_unreadable": skip_unreadable,
            "error":              error_count,
        },
        "survival_rate_pct":      round(100.0 * ok_count / max(1, total), 2),
        "pt_files_created":       pt_created,
        "augmentation_factor":    len(SHIFTS),
        "elapsed_seconds":        round(total_elapsed, 1),
        "config": {
            "shifts":        SHIFTS,
            "pitch_min":     PITCH_MIN,
            "pitch_max":     PITCH_MAX,
            "tokenizer": {
                "beat_res":   "V2: {(0,4):8, (4,12):4}",
                "vocab_size": "550 (V2)",
                "delete_equal_successive_tempo_changes": True,
                "delete_equal_successive_time_sig_changes": True,
            },
            "filters": {
                "f1_min_notes":       F1_MIN_NOTES,
                "f2_min_duration_s":  F2_MIN_DURATION_S,
                "f3_max_vel_pct":     F3_MAX_VEL_PCT,
            },
        },
    }

    if not dry_run:
        stats_path = output_dir / "preprocessing_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"  Stats JSON: {stats_path}")
    else:
        print()
        print("  Dry run stats (not saved):")
        print(json.dumps(stats, indent=4))

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Files submitted      : {total:>10,}")
    print(f"  OK (all filters pass): {ok_count:>10,}  ({100*ok_count/max(1,total):.1f}%)")
    print(f"  Skipped F1 (sparse)  : {skip_f1:>10,}  ({100*skip_f1/max(1,total):.1f}%)")
    print(f"  Skipped F2 (short)   : {skip_f2:>10,}  ({100*skip_f2/max(1,total):.1f}%)")
    print(f"  Skipped F3 (vel)     : {skip_f3:>10,}  ({100*skip_f3/max(1,total):.1f}%)")
    print(f"  Unreadable           : {skip_unreadable:>10,}  ({100*skip_unreadable/max(1,total):.1f}%)")
    print(f"  Errors               : {error_count:>10,}  ({100*error_count/max(1,total):.1f}%)")
    print(f"  Total skipped        : {total_skipped:>10,}  ({100*total_skipped/max(1,total):.1f}%)")
    print()
    if not dry_run:
        print(f"  .pt files created    : {pt_created:>10,}")
        if ok_count > 0:
            print(f"  Avg .pt/ok file      : {pt_created/ok_count:>10.1f}  (expected {len(SHIFTS)})")
    else:
        est_pt = ok_count * len(SHIFTS)
        print(f"  Est. .pt files (full run): ~{est_pt:,}  ({ok_count} files x {len(SHIFTS)} shifts)")
        if total_disk > total:
            est_full_ok = round(ok_count / total * total_disk)
            est_full_pt = est_full_ok * len(SHIFTS)
            print(f"  Est. for full dataset    : ~{est_full_pt:,}  ({est_full_ok:,} files pass filters)")
    print()
    print(f"  Total time           : {total_elapsed:.1f}s")
    if not dry_run:
        print(f"  Output directory     : {output_dir.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
