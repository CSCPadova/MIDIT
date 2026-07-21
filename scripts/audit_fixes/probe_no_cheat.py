"""
scripts/audit_fixes/probe_no_cheat.py

THROWAWAY diagnostic (read-only; CPU is fine). Verifies mechanically that
inference cannot "cheat":
  PART B — at masked CONTENT slots the model INPUT is the MASK id (not the
           original id), and predictions come from logits (mostly != original).
  PART C — perturbing ONLY the context bars (keeping masked bars + scaffold
           identical) changes predictions -> the model conditions on context,
           it is not replaying anything.

It reproduces the pipeline's seed->file selection and masking EXACTLY by reusing
infer.py's own functions (get_val_files, extract_chunk, apply_bar_mask), so the
tokens/mask match what generated samples_1000/seed_N.

Usage (cluster; CPU ok, or GPU if available):
    python scripts/audit_fixes/probe_no_cheat.py \
        --checkpoint /nfsd/.../checkpoints_infilling/infilling-epoch=05-val_loss=8.5836.ckpt \
        --data_dir   /nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_50k \
        --seed 79 --n_slots 10 --mask_bars 3 4 5
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from midit.infer import (          # noqa: E402
    MASK_ID, SPECIAL_IDS,
    build_tokenizer, build_vocab_lookups,
    get_val_files, extract_chunk, apply_bar_mask,
)
from midit.model.masked_transformer import MaskedTransformerREMI  # noqa: E402

CONTENT_HEADS = ("pitch", "velocity", "duration", "program")


def select_chunk_for_seed(seed: int, val_files):
    """Replicate generate_samples_batch.process_seed file selection."""
    for offset in range(20):
        candidate = val_files[(seed + offset) % len(val_files)]
        chunk = extract_chunk(candidate)
        if chunk is not None:
            return candidate, chunk
    raise RuntimeError(f"No valid chunk for seed {seed} in 20 attempts.")


def run_forward(model, tokens: torch.Tensor, device):
    """Return logits dict {head: (1, seq, vocab)} for a 1-D token tensor."""
    with torch.no_grad():
        hidden = model(tokens.unsqueeze(0).to(device))
        return model.multi_head_output(hidden)


def predict_slot(logits, lookups, pos_int: int, orig_id: int):
    """Mirror infer.reconstruct's per-slot decision. Returns (pred_id, source)."""
    type_for_id = lookups["type_for_id"]
    class_to_name = lookups["class_to_name"]
    vocab = lookups["vocab"]

    if orig_id in SPECIAL_IDS:
        return orig_id, "special_passthrough"
    head_name = type_for_id.get(orig_id)
    if head_name is None:
        return None, "unrecognized"
    if pos_int >= logits[head_name].shape[1]:
        return orig_id, "fallback_shape"
    class_idx = logits[head_name][0, pos_int].argmax().item()
    names = class_to_name[head_name]
    if class_idx >= len(names):
        return orig_id, "fallback_oob"
    return vocab.get(names[class_idx]), "logits_argmax"


def main() -> None:
    ap = argparse.ArgumentParser(description="Anti-cheat + context-dependence probe.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seed", type=int, default=79)
    ap.add_argument("--n_slots", type=int, default=10)
    ap.add_argument("--mask_bars", type=int, nargs="+", default=[3, 4, 5])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device={device}  seed={args.seed}  mask_bars={args.mask_bars}")

    # ── model + tokenizer ─────────────────────────────────────────────────────
    model = MaskedTransformerREMI.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    model = model.to(device)
    tokenizer = build_tokenizer()
    lookups = build_vocab_lookups(tokenizer)
    id_to_token = lookups["id_to_token"]
    type_for_id = lookups["type_for_id"]

    # ── reproduce the exact chunk + masking for this seed ─────────────────────
    val_files = get_val_files(args.data_dir)
    file_path, chunk = select_chunk_for_seed(args.seed, val_files)
    original_tokens = chunk["tokens"]
    bar_positions = chunk["bar_positions"]
    print(f"[probe] file={Path(file_path).name}  tokens={len(original_tokens)}  "
          f"bars={len(bar_positions)-1}")

    masked_tokens, mask_labels = apply_bar_mask(
        original_tokens, bar_positions, args.mask_bars, lookups
    )

    masked_positions = mask_labels.nonzero(as_tuple=True)[0].tolist()
    # keep only CONTENT slots (all masked slots are content by apply_bar_mask,
    # but filter explicitly for safety), take first n_slots
    content_slots = [
        p for p in masked_positions
        if type_for_id.get(original_tokens[p].item()) in CONTENT_HEADS
    ][: args.n_slots]

    # ── PART B: input==MASK, predictions from logits ──────────────────────────
    logits = run_forward(model, masked_tokens, device)

    print("\n" + "=" * 74)
    print(f"  PART B — first {len(content_slots)} masked CONTENT slots (seed {args.seed})")
    print("=" * 74)
    print(f"  {'slot':>5} {'input_id':>9} {'MASK?':>6} {'orig_id':>8} "
          f"{'pred_id':>8} {'equal?':>7}  {'orig->pred (names)':<32} {'source'}")
    print("  " + "-" * 108)

    baseline_pred = {}
    n_equal = n_input_ok = 0
    for p in content_slots:
        in_id = masked_tokens[p].item()
        orig_id = original_tokens[p].item()
        pred_id, source = predict_slot(logits, lookups, p, orig_id)
        baseline_pred[p] = pred_id
        eq = (pred_id == orig_id)
        n_equal += int(eq)
        input_is_mask = (in_id == MASK_ID)
        n_input_ok += int(input_is_mask)
        names = f"{id_to_token.get(orig_id,'?')} -> {id_to_token.get(pred_id,'?')}"
        print(f"  {p:>5} {in_id:>9} {('YES' if input_is_mask else 'NO!'):>6} "
              f"{orig_id:>8} {str(pred_id):>8} {('YES' if eq else 'no'):>7}  "
              f"{names:<32} {source}")

    print("  " + "-" * 108)
    print(f"  input==MASK: {n_input_ok}/{len(content_slots)}   "
          f"predicted==original: {n_equal}/{len(content_slots)} "
          f"(expect a MINORITY equal)")
    if n_input_ok != len(content_slots):
        print("  *** ALERT: some masked slot did NOT contain MASK in the input — CHEAT POSSIBLE ***")

    # ── PART C: perturb ONLY context bars, keep masked bars + scaffold ─────────
    # Masked-bar token span (token space): [bar_positions[min], bar_positions[max+1])
    start_bar, end_bar = min(args.mask_bars), max(args.mask_bars)
    mask_span = range(bar_positions[start_bar], bar_positions[end_bar + 1])
    mask_span_set = set(mask_span)

    # Context CONTENT positions = content tokens OUTSIDE the masked-bar span.
    context_content_pos = [
        i for i in range(len(masked_tokens))
        if i not in mask_span_set
        and type_for_id.get(masked_tokens[i].item()) in CONTENT_HEADS
    ]

    perturbed = masked_tokens.clone()
    rng = random.Random(1234)
    vals = [perturbed[i].item() for i in context_content_pos]
    shuffled = vals[:]
    rng.shuffle(shuffled)
    for i, v in zip(context_content_pos, shuffled):
        perturbed[i] = v

    # sanity: masked-bar region must be byte-identical (scaffold + MASK unchanged)
    same_mask_region = all(
        perturbed[i].item() == masked_tokens[i].item() for i in mask_span_set
    )

    logits_p = run_forward(model, perturbed, device)

    print("\n" + "=" * 74)
    print(f"  PART C — shuffle CONTEXT content ({len(context_content_pos)} tokens), "
          f"re-predict same {len(content_slots)} slots")
    print("=" * 74)
    print(f"  masked-bar region unchanged: {'YES' if same_mask_region else 'NO!!'}")
    print(f"  {'slot':>5} {'orig_id':>8} {'pred_base':>10} {'pred_shuf':>10} {'changed?':>9}")
    print("  " + "-" * 50)

    n_changed = 0
    for p in content_slots:
        orig_id = original_tokens[p].item()
        base = baseline_pred[p]
        new_pred, _ = predict_slot(logits_p, lookups, p, orig_id)
        changed = (new_pred != base)
        n_changed += int(changed)
        print(f"  {p:>5} {orig_id:>8} {str(base):>10} {str(new_pred):>10} "
              f"{('CHANGED' if changed else 'same'):>9}")

    print("  " + "-" * 50)
    print(f"  predictions changed by context shuffle: {n_changed}/{len(content_slots)}")
    print("\n[INTERPRETATION]")
    print("  PART B: input==MASK everywhere + predicted!=original majority -> no cheat.")
    print("  PART C: n_changed > 0 -> model conditions on context (not replaying).")
    if n_changed == 0:
        print("  *** ALERT: predictions did NOT change with context — investigate. ***")


if __name__ == "__main__":
    main()
