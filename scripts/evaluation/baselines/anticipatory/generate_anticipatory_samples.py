"""Batch infilling inference with the Anticipatory Music Transformer.

SOTA baseline reproduction for the MIDIT thesis (Fase B). This is experimental
comparison code — NOT part of the core MIDIT model/training pipeline.

For each MIDIT sample (input_dir/seed_N/original.mid) this produces an
infilling reconstruction (output_dir/seed_N/reconstructed.mid) using
Anticipatory MT span infilling, with the surrounding context supplied as
anticipated controls.

Reference: Thickstun et al., "Anticipatory Music Transformer",
NeurIPS 2023, arXiv:2306.08620.

Run inside the dedicated .venv_anticipation environment (see README.md);
do NOT use the main MIDIT .venv.
"""

import argparse
import sys
import time
from pathlib import Path

import mido
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from anticipation.convert import midi_to_events, events_to_midi
from anticipation.sample import generate
from anticipation import ops
from anticipation.vocab import CONTROL_OFFSET


DEFAULT_TEMPO = 500000  # microseconds per beat (120 BPM), MIDI default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anticipatory MT batch infilling for MIDIT SOTA comparison.",
    )
    parser.add_argument(
        "--input_dir", type=Path, required=True,
        help="Directory containing seed_N/ subdirs, each with original.mid",
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Directory for output seed_N/reconstructed.mid",
    )
    parser.add_argument(
        "--model_name", type=str, default="stanford-crfm/music-medium-800k",
        help="HuggingFace model id for the Anticipatory MT checkpoint",
    )
    parser.add_argument(
        "--mask_bars", type=int, nargs="+", default=[3, 4, 5],
        help="Bar indices defining the masked (infilled) span",
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Process only the first N samples (default: all)",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95,
        help="Nucleus sampling top-p for generation",
    )
    return parser.parse_args()


def read_first_tempo(mid: mido.MidiFile) -> int:
    """Return the first set_tempo value (microseconds/beat), or the default."""
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                return msg.tempo
    return DEFAULT_TEMPO


def find_seed_dirs(input_dir: Path) -> list[Path]:
    """Return sorted seed_N/ subdirectories that contain an original.mid."""
    seed_dirs = [
        d for d in input_dir.glob("seed_*")
        if d.is_dir() and (d / "original.mid").exists()
    ]
    # Sort numerically by the integer after "seed_" so seed_2 < seed_10.
    def seed_index(d: Path) -> int:
        try:
            return int(d.name.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1
    return sorted(seed_dirs, key=seed_index)


def process_one(
    seed_dir: Path,
    output_path: Path,
    model,
    mask_bars: list[int],
    top_p: float,
) -> str:
    """Run infilling for a single seed dir.

    Returns one of: 'success', 'skipped_short', 'skipped_exists'.
    Raises on malformed input — the caller logs and continues.
    """
    if output_path.exists():
        return "skipped_exists"

    midi_path = seed_dir / "original.mid"

    mid = mido.MidiFile(str(midi_path))
    total_duration = mid.length  # seconds
    tempo = read_first_tempo(mid)

    # 4/4 assumed: one bar = 4 beats.
    bar_seconds = (tempo / 1_000_000) * 4
    full_window = 8 * bar_seconds

    # Skip files too short to contain the 8-bar context window.
    if total_duration < 0.9 * full_window:
        return "skipped_short"

    t1 = mask_bars[0] * bar_seconds
    t2 = (mask_bars[-1] + 1) * bar_seconds
    t_ant_end = min(full_window, total_duration)

    events = midi_to_events(str(midi_path))

    # Prefix before the mask is given as plain history.
    history = ops.clip(events, 0, t1, clip_duration=False)

    # Context after the mask is supplied as anticipated controls.
    anticipated_raw = ops.clip(events, t2, t_ant_end, clip_duration=False)
    anticipated = [CONTROL_OFFSET + tok for tok in anticipated_raw]

    inpainted = generate(
        model,
        start_time=t1,
        end_time=t2,
        inputs=history,
        controls=anticipated,
        top_p=top_p,
    )

    # generate() re-encodes the prefix; combine with the anticipated controls.
    # The resulting span may run [~bar_1 -> t2] — expected, do not compensate.
    output = ops.combine(inpainted, anticipated)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid_out = events_to_midi(output)
    mid_out.save(str(output_path))
    return "success"


def main() -> None:
    args = parse_args()

    seed_dirs = find_seed_dirs(args.input_dir)
    if not seed_dirs:
        print(f"[ERROR] No seed_N/ dirs with original.mid found in {args.input_dir}")
        sys.exit(1)

    if args.n_samples is not None:
        seed_dirs = seed_dirs[: args.n_samples]

    print(f"[INFO] Found {len(seed_dirs)} samples to process")
    print(f"[INFO] mask_bars={args.mask_bars}  top_p={args.top_p}")
    print(f"[INFO] Loading model: {args.model_name}")

    model = AutoModelForCausalLM.from_pretrained(args.model_name).cuda()
    model.eval()
    print("[INFO] Model loaded.\n")

    n_success = n_failed = n_skipped = 0
    start_time = time.time()

    for seed_dir in tqdm(seed_dirs, desc="Anticipatory infilling"):
        output_path = args.output_dir / seed_dir.name / "reconstructed.mid"
        try:
            result = process_one(
                seed_dir, output_path, model, args.mask_bars, args.top_p
            )
            if result == "success":
                n_success += 1
            elif result == "skipped_short":
                n_skipped += 1
                tqdm.write(f"[SKIP] {seed_dir.name}: too short")
            elif result == "skipped_exists":
                n_skipped += 1
        except Exception as e:  # noqa: BLE001 — robustness to malformed MIDI
            n_failed += 1
            tqdm.write(f"[FAIL] {seed_dir.name}: {type(e).__name__}: {e}")
            continue

    elapsed = time.time() - start_time
    processed = max(n_success, 1)

    print("\n" + "=" * 60)
    print("ANTICIPATORY MT INFILLING — SUMMARY")
    print("=" * 60)
    print(f"  Success: {n_success}")
    print(f"  Failed:  {n_failed}")
    print(f"  Skipped: {n_skipped}")
    print(f"  Total:   {len(seed_dirs)}")
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Avg/sample (success): {elapsed / processed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
