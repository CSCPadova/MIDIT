"""
scripts/generate_samples_batch.py

Batch infilling: loads the model checkpoint ONCE and runs all seeds
in-process, avoiding 1000x checkpoint reload overhead (~10-20s each).

Usage:
    python scripts/generate_samples_batch.py \
        --checkpoint /nfsd/.../infilling-epoch=05-val_loss=8.5836.ckpt \
        --data_dir   /nfsd/.../preprocessed_v2_50k \
        --output_dir /nfsd/.../outputs/samples_1000 \
        --n_seeds    1000 \
        --start_seed 1 \
        --mask_bars  3 4 5
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import List, Optional

import torch

from midit.model.masked_transformer import MaskedTransformerREMI
from midit.infer import (
    build_tokenizer,
    build_vocab_lookups,
    get_val_files,
    extract_chunk,
    apply_bar_mask,
    reconstruct,
    save_midi,
)


def process_seed(
    seed: int,
    val_files: List[Path],
    model: MaskedTransformerREMI,
    tokenizer,
    lookups: dict,
    mask_bars: List[int],
    output_dir: Path,
    device: torch.device,
) -> Optional[bool]:
    """Run the full infilling pipeline for one seed.

    Returns:
        None  — skipped (reconstructed.mid already exists)
        True  — success
        False — failure
    """
    seed_dir = output_dir / f"seed_{seed}"

    if (seed_dir / "reconstructed.mid").exists():
        return None  # already done

    # Find a val file with >=8 bars, trying up to 20 offsets
    chunk = None
    file_path = None
    for offset in range(20):
        candidate = val_files[(seed + offset) % len(val_files)]
        chunk = extract_chunk(candidate)
        if chunk is not None:
            file_path = candidate
            break

    if chunk is None:
        print(f"  [seed {seed:>4}] ERROR: no valid chunk in 20 attempts", flush=True)
        return False

    original_tokens = chunk["tokens"]
    bar_positions   = chunk["bar_positions"]

    if len(original_tokens) >= 8000:
        print(f"  [seed {seed:>4}] SKIP: sequence too long ({len(original_tokens)} tokens)", flush=True)
        return False

    masked_tokens, mask_labels = apply_bar_mask(
        original_tokens, bar_positions, mask_bars, lookups
    )
    n_masked = mask_labels.sum().item()

    # Suppress per-token verbose output from reconstruct() — captured to buf
    buf = io.StringIO()
    with redirect_stdout(buf):
        reconstructed_tokens = reconstruct(
            model, original_tokens, masked_tokens, mask_labels, lookups, device
        )

    print(
        f"  [seed {seed:>4}]  {file_path.name:<40}  masked={n_masked:>4} tokens",
        flush=True,
    )

    seed_dir.mkdir(parents=True, exist_ok=True)
    save_midi(original_tokens,      tokenizer, seed_dir / "original.mid")
    save_midi(masked_tokens,         tokenizer, seed_dir / "masked.mid")
    save_midi(reconstructed_tokens,  tokenizer, seed_dir / "reconstructed.mid")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch infilling: loads model once, runs all seeds in-process.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint",  required=True,                         help="Path to .ckpt file.")
    parser.add_argument("--data_dir",    required=True,                         help="Directory with preprocessed .pt files.")
    parser.add_argument("--output_dir",  required=True,                         help="Base output directory (seed_N/ subdirs created inside).")
    parser.add_argument("--n_seeds",     type=int,   default=1000,              help="Number of seeds to generate.")
    parser.add_argument("--start_seed",  type=int,   default=1,                 help="First seed index.")
    parser.add_argument("--mask_bars",   type=int,   nargs="+", default=[3, 4, 5], help="0-indexed bars to mask.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    end_seed   = args.start_seed + args.n_seeds - 1

    print("\n[generate_samples_batch]")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  data_dir   : {args.data_dir}")
    print(f"  output_dir : {output_dir}")
    print(f"  seeds      : {args.start_seed} -> {end_seed}  ({args.n_seeds} total)")
    print(f"  mask_bars  : {args.mask_bars}")
    print()

    # 1. Load model once
    print("[1/3] Loading model checkpoint...")
    t0 = time.time()
    model = MaskedTransformerREMI.load_from_checkpoint(args.checkpoint)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    print(f"  Device    : {device}")
    print(f"  Params    : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Load time : {time.time() - t0:.1f}s")
    print()

    # 2. Build tokenizer and val split once
    print("[2/3] Building tokenizer and loading val split...")
    tokenizer = build_tokenizer()
    lookups   = build_vocab_lookups(tokenizer)
    val_files = get_val_files(args.data_dir)
    print(f"  Vocab size : {len(tokenizer.vocab)}")
    print(f"  Val files  : {len(val_files)}")
    print()

    # 3. Inference loop
    print("[3/3] Running inference loop...")
    t_loop    = time.time()
    n_done    = 0
    n_skipped = 0
    n_failed  = 0

    for seed in range(args.start_seed, args.start_seed + args.n_seeds):
        result = process_seed(
            seed=seed,
            val_files=val_files,
            model=model,
            tokenizer=tokenizer,
            lookups=lookups,
            mask_bars=args.mask_bars,
            output_dir=output_dir,
            device=device,
        )

        if result is None:
            n_skipped += 1
        elif result:
            n_done += 1
        else:
            n_failed += 1

        processed = n_done + n_failed
        if processed > 0 and processed % 100 == 0:
            elapsed   = time.time() - t_loop
            rate      = processed / elapsed               # seeds/s
            remaining = (args.n_seeds - n_skipped - processed) / rate if rate > 0 else 0
            print(
                f"\n  --- [{processed}/{args.n_seeds - n_skipped}] "
                f"done={n_done}  skipped={n_skipped}  failed={n_failed} "
                f"| {rate * 60:.1f} seeds/min "
                f"| ETA {remaining / 60:.0f} min ---\n",
                flush=True,
            )

    elapsed_total = time.time() - t_loop
    print("\n[generate_samples_batch] Complete.")
    print(f"  Generated : {n_done}")
    print(f"  Skipped   : {n_skipped} (already existed)")
    print(f"  Failed    : {n_failed}")
    print(f"  Total time: {elapsed_total / 60:.1f} min")
    if n_done > 0:
        print(f"  Avg/seed  : {elapsed_total / n_done:.1f}s")


if __name__ == "__main__":
    main()
