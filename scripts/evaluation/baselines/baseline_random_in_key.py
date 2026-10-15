"""
scripts/evaluation/baselines/baseline_random_in_key.py

Baseline for CP/GS/PCHE floor. Randomizes ONLY pitch; onsets/duration/velocity/
program taken from ground truth. Isolates the pitch component. This is a random
lower-bound FAVORABLE to the baseline (perfect scaffold), so any MIDIT margin
over it is conservative.

THROWAWAY audit script (no model, no GPU). Reuses midit.infer UNCHANGED
(build_tokenizer, build_vocab_lookups, get_val_files, extract_chunk,
apply_bar_mask, save_midi). Writes only into --output_dir/<mode>/seed_N/;
never touches outputs/samples_1000.

Pipeline (identical to MIDIT except the fill step):
    get_val_files -> (seed+offset selection, same as generate_samples_batch)
    -> extract_chunk -> apply_bar_mask -> fill_random (NO model)
    -> save original.mid / masked.mid / reconstructed.mid

Fill rules (family-preserving):
    * masked slot with head=='pitch':
        - original PitchDrum_* -> random PitchDrum_* id (uniform, both modes:
          drums have no key-based pitch class)
        - original Pitch_*     -> random Pitch_* id
            b1_uniform : uniform over the melodic Pitch_* id pool
            b2_context : draw a pitch-class from the histogram of Pitch_* tokens
                         in the UNMASKED context bars (bars != mask_bars), then
                         pick a Pitch_* id of that class
    * any other masked slot (Velocity/Duration/Program): copy the ORIGINAL id
      (oracle scaffold — isolates pitch as the only randomized variable)

RNG is seeded per seed (rng_seed + seed) for reproducibility.

Metrics afterwards (UNCHANGED pipeline), e.g.:
    python scripts/evaluation/compute_muspy_metrics.py \
        --results_dir outputs/baseline_random_1000/b1_uniform \
        --output_csv  outputs/baseline_random_1000/metrics_b1.csv
    (then filter to the 973 seeds in sample_manifest.csv)

Usage:
    python scripts/evaluation/baselines/baseline_random_in_key.py \
        --mode b1_uniform \
        --data_dir /nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_50k \
        --output_dir outputs/baseline_random_1000 \
        --n_seeds 1000 --start_seed 1 --mask_bars 3 4 5
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from midit.infer import (
    build_tokenizer,
    build_vocab_lookups,
    get_val_files,
    extract_chunk,
    apply_bar_mask,
    save_midi,
)


# ── Pitch id pools (depend only on the vocab) ─────────────────────────────────

def build_pitch_pools(lookups: Dict):
    """Return (melodic_ids, drum_ids, pc_to_ids).

    melodic_ids : list[int]           ids of Pitch_* tokens
    drum_ids    : list[int]           ids of PitchDrum_* tokens
    pc_to_ids   : dict[int, list[int]] pitch-class (0-11) -> melodic ids of that class
    """
    class_to_name = lookups["class_to_name"]
    vocab = lookups["vocab"]

    melodic_names = [n for n in class_to_name["pitch"] if n.startswith("Pitch_")]
    drum_names    = [n for n in class_to_name["pitch"] if n.startswith("PitchDrum_")]

    melodic_ids = [vocab[n] for n in melodic_names]
    drum_ids    = [vocab[n] for n in drum_names]

    pc_to_ids: Dict[int, List[int]] = {}
    for n in melodic_names:
        val = int(n[len("Pitch_"):])          # Pitch_69 -> 69
        pc_to_ids.setdefault(val % 12, []).append(vocab[n])

    return melodic_ids, drum_ids, pc_to_ids


def masked_bar_token_set(bar_positions: List[int], mask_bars: List[int]) -> set:
    """Token indices belonging to the masked bars (handles non-contiguous)."""
    idx = set()
    n_bars = len(bar_positions) - 1
    for b in mask_bars:
        if 0 <= b < n_bars:
            idx.update(range(bar_positions[b], bar_positions[b + 1]))
    return idx


def context_pc_hist(
    original_tokens: torch.Tensor,
    bar_positions: List[int],
    mask_bars: List[int],
    id_to_token: Dict[int, str],
) -> Counter:
    """Pitch-class histogram of Pitch_* tokens in UNMASKED context bars."""
    masked_idx = masked_bar_token_set(bar_positions, mask_bars)
    hist: Counter = Counter()
    for i in range(len(original_tokens)):
        if i in masked_idx:
            continue
        name = id_to_token.get(original_tokens[i].item(), "")
        if name.startswith("Pitch_"):
            hist[int(name[len("Pitch_"):]) % 12] += 1
    return hist


# ── Fill (replaces reconstruct(); NO model) ───────────────────────────────────

def fill_random(
    original_tokens: torch.Tensor,
    masked_tokens: torch.Tensor,
    mask_labels: torch.Tensor,
    lookups: Dict,
    mode: str,
    pools: Tuple[List[int], List[int], Dict[int, List[int]]],
    ctx_hist: Counter,
    rng: random.Random,
) -> torch.Tensor:
    """Fill masked slots: random pitch (family-preserving), scaffold copied."""
    type_for_id = lookups["type_for_id"]
    id_to_token = lookups["id_to_token"]
    melodic_ids, drum_ids, pc_to_ids = pools

    reconstructed = masked_tokens.clone()
    masked_positions = mask_labels.nonzero(as_tuple=True)[0].tolist()

    pcs = list(ctx_hist.keys())
    weights = [ctx_hist[p] for p in pcs]

    for pos in masked_positions:
        orig_id = original_tokens[pos].item()
        head = type_for_id.get(orig_id)

        if head == "pitch":
            name = id_to_token.get(orig_id, "")
            if name.startswith("PitchDrum_"):
                reconstructed[pos] = rng.choice(drum_ids) if drum_ids else orig_id
            else:
                if mode == "b2_context" and pcs:
                    pc = rng.choices(pcs, weights=weights, k=1)[0]
                    ids = pc_to_ids.get(pc) or melodic_ids
                    reconstructed[pos] = rng.choice(ids)
                else:  # b1_uniform, or b2 with empty context histogram
                    reconstructed[pos] = rng.choice(melodic_ids) if melodic_ids else orig_id
        else:
            # Velocity / Duration / Program (and any other): oracle scaffold
            reconstructed[pos] = orig_id

    return reconstructed


# ── Per-seed processing (mirrors generate_samples_batch.process_seed) ─────────

def process_seed(
    seed: int,
    val_files,
    lookups: Dict,
    tokenizer,
    pools,
    mode: str,
    mask_bars: List[int],
    out_root: Path,
    rng_seed: int,
) -> Tuple[Optional[bool], bool]:
    """Return (result, used_fallback).

    result: None = skipped (already done); True = success; False = failure.
    used_fallback: True iff mode==b2_context AND the context pitch-class
                   histogram was empty (whole seed fell back to uniform).
                   Meaningful only when result is True.
    """
    seed_dir = out_root / f"seed_{seed}"
    if (seed_dir / "reconstructed.mid").exists():
        return None, False

    # seed->file selection: identical to generate_samples_batch.process_seed
    chunk = None
    for offset in range(20):
        candidate = val_files[(seed + offset) % len(val_files)]
        chunk = extract_chunk(candidate)
        if chunk is not None:
            break
    if chunk is None:
        print(f"  [seed {seed:>4}] ERROR: no valid chunk in 20 attempts", flush=True)
        return False, False

    original_tokens = chunk["tokens"]
    bar_positions   = chunk["bar_positions"]

    if len(original_tokens) >= 8000:
        print(f"  [seed {seed:>4}] SKIP: too long ({len(original_tokens)} tokens)", flush=True)
        return False, False

    masked_tokens, mask_labels = apply_bar_mask(
        original_tokens, bar_positions, mask_bars, lookups
    )

    rng = random.Random(rng_seed + seed)
    ctx_hist = (context_pc_hist(original_tokens, bar_positions, mask_bars, lookups["id_to_token"])
                if mode == "b2_context" else Counter())
    used_fallback = (mode == "b2_context" and len(ctx_hist) == 0)

    reconstructed = fill_random(
        original_tokens, masked_tokens, mask_labels,
        lookups, mode, pools, ctx_hist, rng,
    )

    seed_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    ok &= save_midi(original_tokens, tokenizer, seed_dir / "original.mid")
    ok &= save_midi(masked_tokens,    tokenizer, seed_dir / "masked.mid")
    ok &= save_midi(reconstructed,    tokenizer, seed_dir / "reconstructed.mid")

    n_masked = int(mask_labels.sum().item())
    fb = "  [b2 empty-ctx fallback]" if used_fallback else ""
    print(f"  [seed {seed:>4}] masked={n_masked:>4}  {'OK' if ok else 'SAVE-FAIL'}{fb}", flush=True)
    return bool(ok), used_fallback


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Random-in-key pitch baseline (CP/GS/PCHE floor). No model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mode", required=True, choices=["b1_uniform", "b2_context"])
    ap.add_argument("--data_dir", required=True, help="Preprocessed .pt dir (same as MIDIT).")
    ap.add_argument("--output_dir", default="outputs/baseline_random_1000",
                    help="Base dir; <mode>/seed_N/ created inside.")
    ap.add_argument("--n_seeds", type=int, default=1000)
    ap.add_argument("--start_seed", type=int, default=1)
    ap.add_argument("--mask_bars", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--rng_seed", type=int, default=1234)
    args = ap.parse_args()

    out_root = Path(args.output_dir) / args.mode
    end_seed = args.start_seed + args.n_seeds - 1

    print("\n[baseline_random_in_key]")
    print(f"  mode       : {args.mode}")
    print(f"  data_dir   : {args.data_dir}")
    print(f"  output     : {out_root}")
    print(f"  seeds      : {args.start_seed} -> {end_seed}  ({args.n_seeds} total)")
    print(f"  mask_bars  : {args.mask_bars}")
    print(f"  rng_seed   : {args.rng_seed}\n")

    tokenizer = build_tokenizer()
    lookups   = build_vocab_lookups(tokenizer)
    pools     = build_pitch_pools(lookups)
    val_files = get_val_files(args.data_dir)
    melodic, drum, pc_to_ids = pools
    print(f"  val_files  : {len(val_files)}")
    print(f"  pitch pool : melodic={len(melodic)}  drum={len(drum)}  "
          f"pitch_classes={len(pc_to_ids)}\n")

    n_done = n_skipped = n_failed = n_fallback = 0
    for seed in range(args.start_seed, args.start_seed + args.n_seeds):
        result, used_fallback = process_seed(
            seed, val_files, lookups, tokenizer, pools,
            args.mode, args.mask_bars, out_root, args.rng_seed,
        )
        if result is None:
            n_skipped += 1
        elif result:
            n_done += 1
            if used_fallback:
                n_fallback += 1
        else:
            n_failed += 1

    print("\n[baseline_random_in_key] Complete.")
    print(f"  Generated : {n_done}")
    print(f"  Skipped   : {n_skipped} (already existed)")
    print(f"  Failed    : {n_failed}")
    if args.mode == "b2_context":
        print(f"  B2 empty-context fallback: {n_fallback}/{n_done} seeds")
    print(f"  Output    : {out_root}")


if __name__ == "__main__":
    main()
