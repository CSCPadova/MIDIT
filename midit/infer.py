"""
MIDIT Infilling — qualitative evaluation of the masked transformer.

Masks a contiguous set of bars from an 8-bar chunk, runs the model,
reconstructs the missing tokens and saves three MIDI files:
original, masked, and reconstructed.

Usage:
    python midit/infer.py \
        --checkpoint logs/checkpoints_phase1/phase1-epoch=04-val_loss=4.1604.ckpt \
        --data_dir /nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_50k \
        --output_dir outputs/infilling/
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from miditok import REMI, TokenizerConfig, TokSequence

from midit.model.masked_transformer import MaskedTransformerREMI


MASK_ID = 3
BOS_ID  = 1
EOS_ID  = 2
PAD_ID  = 0
SPECIAL_IDS = {PAD_ID, BOS_ID, EOS_ID, MASK_ID}


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def build_tokenizer() -> REMI:
    """Build V2 tokenizer — identical config to lakh_dataset.py."""
    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 8, (4, 12): 4},
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
    return REMI(config)


# ── Vocab lookups ─────────────────────────────────────────────────────────────

def build_vocab_lookups(tokenizer: REMI) -> Dict:
    """Build all vocab lookup structures needed for token reconstruction.

    Returns a dict with:
        id_to_token   — {token_id: token_name}
        class_to_name — {head_name: [token_name, ...]}  index = class_idx
        type_for_id   — {token_id: head_name}  fast dispatch at masked positions
        vocab         — {token_name: token_id}
    """
    vocab = tokenizer.vocab
    vocab_keys = list(vocab.keys())

    token_lists = {
        'pitch':     sorted([k for k in vocab_keys if k.startswith('Pitch_')]),
        'pitchdrum': sorted([k for k in vocab_keys if k.startswith('PitchDrum_')]),
        'velocity':  sorted([k for k in vocab_keys if k.startswith('Velocity_')]),
        'duration':  sorted([k for k in vocab_keys if k.startswith('Duration_')]),
        'program':   sorted([k for k in vocab_keys if k.startswith('Program_')]),
        'position':  sorted([k for k in vocab_keys if k.startswith('Position_')]),
        'bar':       sorted([k for k in vocab_keys if k.startswith('Bar_')]),
        'tempo':     sorted([k for k in vocab_keys if k.startswith('Tempo_')]),
        'timesig':   sorted([k for k in vocab_keys if k.startswith('TimeSig_')]),
        'rest':      sorted([k for k in vocab_keys if k.startswith('Rest_')]),
    }

    # Pitch head covers Pitch_ and PitchDrum_ combined (same as training)
    class_to_name = {
        'pitch':    token_lists['pitch'] + token_lists['pitchdrum'],
        'velocity': token_lists['velocity'],
        'duration': token_lists['duration'],
        'program':  token_lists['program'],
        'position': token_lists['position'],
        'bar':      token_lists['bar'],
        'tempo':    token_lists['tempo'],
        'timesig':  token_lists['timesig'],
        'rest':     token_lists['rest'],
    }

    # Fast lookup: token_id → head name
    type_for_id: Dict[int, str] = {}
    for type_name, lst in token_lists.items():
        head = 'pitch' if type_name == 'pitchdrum' else type_name
        for tok_name in lst:
            type_for_id[vocab[tok_name]] = head

    return {
        'id_to_token':   {v: k for k, v in vocab.items()},
        'class_to_name': class_to_name,
        'type_for_id':   type_for_id,
        'vocab':         vocab,
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def get_val_files(data_dir: str) -> List[Path]:
    """Return val split file list using same seed=42 logic as LakhDataset."""
    all_files = sorted(Path(data_dir).glob("*.pt"))
    if not all_files:
        raise ValueError(f"No .pt files found in {data_dir}")
    rng = np.random.RandomState(42)
    indices = np.arange(len(all_files))
    rng.shuffle(indices)
    n_train = int(0.8 * len(all_files))
    n_val   = int(0.1 * len(all_files))
    return [all_files[i] for i in indices[n_train:n_train + n_val]]


def extract_chunk(file_path: Path, bars_per_chunk: int = 8) -> Optional[Dict]:
    """Load .pt file and extract the first bars_per_chunk bars."""
    data = torch.load(file_path, weights_only=False)
    bar_positions = data['bar_positions']
    num_bars = len(bar_positions) - 1

    if num_bars < bars_per_chunk:
        return None

    start_idx = bar_positions[0]
    end_idx   = bar_positions[bars_per_chunk]
    chunk_tokens = data['tokens'][start_idx:end_idx]
    chunk_bar_positions = [pos - start_idx for pos in bar_positions[:bars_per_chunk + 1]]

    return {
        'tokens':        chunk_tokens,
        'bar_positions': chunk_bar_positions,
        'source_file':   str(file_path),
    }


# ── Masking ───────────────────────────────────────────────────────────────────

def apply_bar_mask(
    tokens: torch.Tensor,
    bar_positions: List[int],
    mask_bars: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace all tokens in the specified bars with MASK_ID.

    Args:
        tokens:        Original token IDs [seq_len]
        bar_positions: Bar boundary positions (length = num_bars + 1)
        mask_bars:     0-indexed bar numbers within the chunk to mask

    Returns:
        masked_tokens  [seq_len]
        mask_labels    bool tensor, True at masked positions [seq_len]
    """
    masked = tokens.clone()
    mask_labels = torch.zeros(len(tokens), dtype=torch.bool)

    for bar_idx in mask_bars:
        if bar_idx + 1 >= len(bar_positions):
            print(f"  Warning: bar {bar_idx} out of range (chunk has {len(bar_positions)-1} bars), skipping")
            continue
        start = bar_positions[bar_idx]
        end   = bar_positions[bar_idx + 1]
        masked[start:end] = MASK_ID
        mask_labels[start:end] = True

    return masked, mask_labels


# ── Reconstruction ────────────────────────────────────────────────────────────

def reconstruct(
    model: MaskedTransformerREMI,
    original_tokens: torch.Tensor,
    masked_tokens: torch.Tensor,
    mask_labels: torch.Tensor,
    lookups: Dict,
    device: torch.device,
) -> torch.Tensor:
    """Run model forward pass and fill masked positions with argmax predictions.

    For each masked position:
      1. Read the original token type (which head to query)
      2. Take argmax from that head's logits at this position
      3. Map class_idx → token_name → token_id
    """
    type_for_id   = lookups['type_for_id']
    class_to_name = lookups['class_to_name']
    vocab         = lookups['vocab']

    with torch.no_grad():
        hidden = model(masked_tokens.unsqueeze(0).to(device))
        logits = model.multi_head_output(hidden)   # {head: (1, seq, vocab_size)}

    reconstructed = masked_tokens.clone()
    masked_positions = mask_labels.nonzero(as_tuple=True)[0]

    head_counts: Dict[str, int] = {}

    for pos in masked_positions:
        pos_int = pos.item()
        orig_id = original_tokens[pos_int].item()

        # Special tokens are kept as-is
        if orig_id in SPECIAL_IDS:
            reconstructed[pos_int] = orig_id
            continue

        head_name = type_for_id.get(orig_id)
        if head_name is None:
            # Unrecognised token type — leave as MASK
            continue

        class_idx = logits[head_name][0, pos_int].argmax().item()
        names = class_to_name[head_name]

        if class_idx >= len(names):
            # Model predicted an out-of-range class — keep original token
            reconstructed[pos_int] = orig_id
            continue

        pred_id = vocab.get(names[class_idx])
        if pred_id is not None:
            reconstructed[pos_int] = pred_id
            head_counts[head_name] = head_counts.get(head_name, 0) + 1

    print(f"  Reconstructed {len(masked_positions)} tokens:")
    for head, count in sorted(head_counts.items()):
        print(f"    {head:10s}: {count}")

    return reconstructed


# ── MIDI export ───────────────────────────────────────────────────────────────

def save_midi(tokens: torch.Tensor, tokenizer: REMI, path: Path) -> bool:
    """Convert token IDs to MIDI and save. Returns True on success."""
    ids = tokens.tolist()
    # Try TokSequence API (MidiTok 3.x)
    try:
        score = tokenizer(TokSequence(ids=ids))
        score.dump_midi(str(path))
        return True
    except Exception:
        pass
    # Fallback: pass raw list directly
    try:
        score = tokenizer(ids)
        score.dump_midi(str(path))
        return True
    except Exception as e:
        print(f"  Warning: could not save {path.name}: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MIDIT infilling — mask bars and reconstruct")
    parser.add_argument('--checkpoint', required=True,            help='Path to .ckpt file')
    parser.add_argument('--data_dir',   required=True,            help='Directory with preprocessed .pt files')
    parser.add_argument('--output_dir', default='outputs/infilling', help='Output directory for MIDI files')
    parser.add_argument('--seed',       type=int, default=0,      help='Index into val split for file selection')
    parser.add_argument('--mask_bars',  default='3,4,5',          help='Comma-separated 0-indexed bars to mask')
    args = parser.parse_args()

    mask_bar_list = [int(b.strip()) for b in args.mask_bars.split(',')]
    output_dir    = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading model checkpoint...")
    model = MaskedTransformerREMI.load_from_checkpoint(args.checkpoint)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 2. Build tokenizer and vocab lookups ──────────────────────────────────
    print("\n[2/6] Building tokenizer and vocab lookups...")
    tokenizer = build_tokenizer()
    lookups   = build_vocab_lookups(tokenizer)
    print(f"  Vocab size: {len(tokenizer.vocab)}")

    # ── 3. Select val file ────────────────────────────────────────────────────
    print(f"\n[3/6] Selecting val file (seed={args.seed})...")
    val_files = get_val_files(args.data_dir)
    print(f"  Val split: {len(val_files)} files")

    # Try seed, then seed+1, seed+2, ... until we find a file with ≥8 bars
    chunk = None
    file_path = None
    for offset in range(20):
        candidate = val_files[(args.seed + offset) % len(val_files)]
        chunk = extract_chunk(candidate)
        if chunk is not None:
            file_path = candidate
            if offset > 0:
                print(f"  Skipped {offset} file(s) with <8 bars")
            break

    if chunk is None:
        raise RuntimeError("Could not find a val file with at least 8 bars in the first 20 attempts.")

    print(f"  File: {file_path.name}")

    # ── 4. Inspect chunk ──────────────────────────────────────────────────────
    print("\n[4/6] Chunk info...")
    original_tokens = chunk['tokens']
    bar_positions   = chunk['bar_positions']
    print(f"  Tokens:     {len(original_tokens)}")
    print(f"  Bars:       {len(bar_positions) - 1}")
    print(f"  Mask bars:  {mask_bar_list}")
    for i, (s, e) in enumerate(zip(bar_positions, bar_positions[1:])):
        marker = " ← MASKED" if i in mask_bar_list else ""
        print(f"    bar {i}: tokens [{s}:{e}]  ({e-s} tokens){marker}")

    # ── 5. Apply masking ──────────────────────────────────────────────────────
    print("\n[5/6] Applying masking...")
    masked_tokens, mask_labels = apply_bar_mask(original_tokens, bar_positions, mask_bar_list)
    n_masked = mask_labels.sum().item()
    n_total  = len(original_tokens)
    print(f"  Masked {n_masked}/{n_total} tokens ({100 * n_masked / n_total:.1f}%)")

    # ── 6. Reconstruct ────────────────────────────────────────────────────────
    print("\n[6/6] Running model inference...")
    reconstructed_tokens = reconstruct(
        model, original_tokens, masked_tokens, mask_labels, lookups, device
    )

    # ── Save MIDI files ────────────────────────────────────────────────────────
    print("\n[Saving MIDI files]")
    files_to_save = {
        'original.mid':      original_tokens,
        'masked.mid':        masked_tokens,
        'reconstructed.mid': reconstructed_tokens,
    }
    for filename, tokens in files_to_save.items():
        ok = save_midi(tokens, tokenizer, output_dir / filename)
        print(f"  {'✓' if ok else '✗'}  {output_dir / filename}")

    print(f"\nDone. Output directory: {output_dir.resolve()}")
    print("Listen to original.mid and reconstructed.mid to compare.")


if __name__ == '__main__':
    main()
