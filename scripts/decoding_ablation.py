"""
scripts/decoding_ablation.py

Decoding-strategy ablation for MIDIT infilling (argmax vs temperature/top-k/top-p
vs MaskGIT-style iterative decoding). Analogous to MIDI-RWKV Appendix E.3 Table 15.

Self-contained: imports utilities from midit.infer and metric functions from
scripts/Evaluation_Infilling/compute_infilling_metrics.py WITHOUT modifying them.
Every decoding condition keeps the same oracle token-type read-out and the
Position/Bar/Tempo/TimeSig/Rest scaffolding as midit.infer.reconstruct(); only the
per-slot read-out changes, so results are comparable to the thesis main table.

Runnable locally (small --n_seeds) and via sbatch (slurm/decoding_ablation.sh).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))                       # repo root
sys.path.insert(0, str(Path(__file__).parent.parent /
                        "scripts" / "Evaluation_Infilling"))                # metrics

from compute_infilling_metrics import compute_CP, compute_GS, compute_PCHE, _bar_boundaries
from midit.infer import (
    build_tokenizer, build_vocab_lookups, get_val_files,
    extract_chunk, apply_bar_mask, save_midi,
    MASK_ID, BOS_ID, EOS_ID, PAD_ID, SPECIAL_IDS,
)
from midit.model.masked_transformer import MaskedTransformerREMI

import symusic
import torch
import torch.nn.functional as F
import numpy as np
import csv
import argparse
from collections import defaultdict

# ── Constants ─────────────────────────────────────────────────────────────────
SEED_TORCH = 2024
MAX_SEQ_LEN = 8000   # skip chunks longer than this (matches generate_samples_batch.py)


# ── Decoding read-outs ────────────────────────────────────────────────────────

def sample_from_logits(logit_vec, strategy, temperature=1.0, top_k=0, top_p=1.0):
    """Pick a class index from a per-head logit vector [C_head]."""
    if strategy == "argmax":
        return int(logit_vec.argmax().item())

    if strategy == "temperature":
        probs = F.softmax(logit_vec / temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    if strategy == "top_k":
        v = logit_vec.clone()
        k = min(top_k, v.numel())
        if 0 < k < v.numel():
            kth = torch.topk(v, k).values.min()
            v[v < kth] = float("-inf")
        probs = F.softmax(v, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    if strategy == "top_p":
        sorted_logits, sorted_idx = torch.sort(logit_vec, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p                       # keep cumulative prob <= top_p
        remove[0] = False                          # always keep at least 1 token
        sorted_logits = sorted_logits.clone()
        sorted_logits[remove] = float("-inf")
        probs = F.softmax(sorted_logits, dim=-1)
        choice = torch.multinomial(probs, 1)
        return int(sorted_idx[choice].item())

    raise ValueError(f"Unknown strategy: {strategy}")


def reconstruct_with_strategy(model, original_tokens, masked_tokens, mask_labels,
                              lookups, device, strategy, temperature, top_k, top_p,
                              condition_idx, chunk_seed):
    """Single forward pass; per masked slot, sample from the ORACLE-type head."""
    torch.manual_seed(SEED_TORCH + chunk_seed * 100 + condition_idx)

    type_for_id = lookups["type_for_id"]
    class_to_name = lookups["class_to_name"]
    vocab = lookups["vocab"]

    with torch.no_grad():
        hidden = model(masked_tokens.unsqueeze(0).to(device))
        logits = model.multi_head_output(hidden)

    reconstructed = masked_tokens.clone()
    for pos in mask_labels.nonzero(as_tuple=True)[0]:
        pos_int = pos.item()
        orig_id = original_tokens[pos_int].item()          # ORACLE type source
        if orig_id in SPECIAL_IDS:
            reconstructed[pos_int] = orig_id
            continue
        head_name = type_for_id.get(orig_id)
        if head_name is None:
            continue
        if pos_int >= logits[head_name].shape[1]:
            reconstructed[pos_int] = orig_id
            continue
        class_idx = sample_from_logits(logits[head_name][0, pos_int],
                                       strategy, temperature, top_k, top_p)
        names = class_to_name[head_name]
        if class_idx >= len(names):
            reconstructed[pos_int] = orig_id
            continue
        pred_id = vocab.get(names[class_idx])
        if pred_id is not None:
            reconstructed[pos_int] = pred_id
    return reconstructed


def reconstruct_maskgit(model, original_tokens, masked_tokens_init, mask_labels,
                        lookups, device, n_iterations, condition_idx, chunk_seed):
    """MaskGIT-style iterative decoding: reveal highest-confidence slots first."""
    torch.manual_seed(SEED_TORCH + chunk_seed * 100 + condition_idx)

    type_for_id = lookups["type_for_id"]
    class_to_name = lookups["class_to_name"]
    vocab = lookups["vocab"]

    current = masked_tokens_init.clone()
    still_masked = mask_labels.clone()                     # bool; excludes scaffold
    n_total_masked = int(still_masked.sum().item())
    if n_total_masked == 0:
        return current

    def predict_at(pos_int, logits):
        """Return (pred_id, confidence) using the oracle head at pos_int."""
        orig_id = original_tokens[pos_int].item()
        if orig_id in SPECIAL_IDS:
            return orig_id, 1.0
        head_name = type_for_id.get(orig_id)
        if head_name is None or pos_int >= logits[head_name].shape[1]:
            return orig_id, 1.0
        probs = F.softmax(logits[head_name][0, pos_int], dim=-1)
        conf, cidx = probs.max(dim=-1)
        names = class_to_name[head_name]
        ci = int(cidx.item())
        if ci >= len(names):
            return orig_id, float(conf.item())
        return vocab.get(names[ci], orig_id), float(conf.item())

    for t in range(1, n_iterations + 1):
        with torch.no_grad():
            hidden = model(current.unsqueeze(0).to(device))
            logits = model.multi_head_output(hidden)

        positions = still_masked.nonzero(as_tuple=True)[0].tolist()
        preds = {p: predict_at(p, logits) for p in positions}   # p -> (pred_id, conf)

        # cumulative reveal schedule
        target = max(1, int((t / n_iterations) * n_total_masked))
        if t == n_iterations:
            target = n_total_masked
        revealed = n_total_masked - int(still_masked.sum().item())
        n_new = target - revealed
        if n_new <= 0:
            continue

        ranked = sorted(preds.items(), key=lambda kv: kv[1][1], reverse=True)
        for pos_int, (pred_id, _conf) in ranked[:n_new]:
            current[pos_int] = pred_id
            still_masked[pos_int] = False

    # final argmax pass for any remaining masked slots (safety)
    if int(still_masked.sum().item()) > 0:
        with torch.no_grad():
            hidden = model(current.unsqueeze(0).to(device))
            logits = model.multi_head_output(hidden)
        for pos in still_masked.nonzero(as_tuple=True)[0]:
            pos_int = pos.item()
            pred_id, _ = predict_at(pos_int, logits)
            current[pos_int] = pred_id
            still_masked[pos_int] = False
    return current


# ── Conditions ────────────────────────────────────────────────────────────────

def get_conditions():
    return [
        {'name': 'argmax',          'type': 'standard', 'strategy': 'argmax',      'temperature': 1.0, 'top_k': 0,  'top_p': 1.0},
        {'name': 'temperature_0.8', 'type': 'standard', 'strategy': 'temperature', 'temperature': 0.8, 'top_k': 0,  'top_p': 1.0},
        {'name': 'temperature_1.0', 'type': 'standard', 'strategy': 'temperature', 'temperature': 1.0, 'top_k': 0,  'top_p': 1.0},
        {'name': 'temperature_1.2', 'type': 'standard', 'strategy': 'temperature', 'temperature': 1.2, 'top_k': 0,  'top_p': 1.0},
        {'name': 'top_k_10',        'type': 'standard', 'strategy': 'top_k',       'temperature': 1.0, 'top_k': 10, 'top_p': 1.0},
        {'name': 'top_k_20',        'type': 'standard', 'strategy': 'top_k',       'temperature': 1.0, 'top_k': 20, 'top_p': 1.0},
        {'name': 'top_p_0.90',      'type': 'standard', 'strategy': 'top_p',       'temperature': 1.0, 'top_k': 0,  'top_p': 0.90},
        {'name': 'top_p_0.95',      'type': 'standard', 'strategy': 'top_p',       'temperature': 1.0, 'top_k': 0,  'top_p': 0.95},
        {'name': 'maskgit_k3',      'type': 'maskgit',  'n_iterations': 3},
        {'name': 'maskgit_k5',      'type': 'maskgit',  'n_iterations': 5},
        {'name': 'maskgit_k10',     'type': 'maskgit',  'n_iterations': 10},
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="MIDIT decoding-strategy ablation.")
    ap.add_argument("--checkpoint", required=True, help="Infilling checkpoint (.ckpt).")
    ap.add_argument("--data_dir", required=True, help="preprocessed_v2_50k dir.")
    ap.add_argument("--output_dir", default="outputs/decoding_ablation")
    ap.add_argument("--n_seeds", type=int, default=50)
    ap.add_argument("--mask_bars", type=int, nargs="+", default=[3, 4, 5])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[decoding_ablation] device={device}")

    model = MaskedTransformerREMI.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    model = model.to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    tokenizer = build_tokenizer()
    lookups = build_vocab_lookups(tokenizer)
    val_files = get_val_files(args.data_dir)
    conditions = get_conditions()
    print(f"  val files: {len(val_files)}  ·  conditions: {len(conditions)}  ·  seeds: {args.n_seeds}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    agg = defaultdict(lambda: {"cp": [], "gs": [], "pche": []})

    for seed_idx in range(args.n_seeds):
        # seed -> file selection (mirror generate_samples_batch.process_seed)
        chunk, file_path = None, None
        for offset in range(20):
            cand = val_files[(seed_idx + offset) % len(val_files)]
            chunk = extract_chunk(cand)
            if chunk is not None:
                file_path = cand
                break

        if chunk is None or len(chunk["tokens"]) >= MAX_SEQ_LEN:
            print(f"  Skipping seed {seed_idx}: chunk too long or None")
            continue

        original_tokens = chunk["tokens"]
        bar_positions = chunk["bar_positions"]
        masked_tokens, mask_labels = apply_bar_mask(original_tokens, bar_positions,
                                                    args.mask_bars, lookups)

        seed_dir = out / f"seed_{seed_idx}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        orig_path = seed_dir / "original.mid"
        masked_path = seed_dir / "masked.mid"
        save_midi(original_tokens, tokenizer, orig_path)      # written ONCE per seed
        save_midi(masked_tokens, tokenizer, masked_path)      # written ONCE per seed

        # bar_range from the original score's own tick grid
        orig_score = symusic.Score(str(orig_path))
        tpq = orig_score.ticks_per_quarter                    # ~16 for MIDIT
        boundaries = _bar_boundaries(orig_score)
        bar_range = [(boundaries[b], boundaries[b + 1])
                     for b in args.mask_bars if b + 1 < len(boundaries)]
        if not bar_range:
            print(f"  Skipping seed {seed_idx}: empty bar_range (score too short)")
            continue

        for ci, cond in enumerate(conditions):
            if cond["type"] == "standard":
                recon = reconstruct_with_strategy(
                    model, original_tokens, masked_tokens, mask_labels, lookups, device,
                    cond["strategy"], cond["temperature"], cond["top_k"], cond["top_p"],
                    condition_idx=ci, chunk_seed=seed_idx)
            else:
                recon = reconstruct_maskgit(
                    model, original_tokens, masked_tokens, mask_labels, lookups, device,
                    n_iterations=cond["n_iterations"], condition_idx=ci, chunk_seed=seed_idx)

            cond_dir = seed_dir / cond["name"]
            cond_dir.mkdir(parents=True, exist_ok=True)
            recon_path = cond_dir / "reconstructed.mid"
            if not save_midi(recon, tokenizer, recon_path):
                print(f"  seed {seed_idx} / {cond['name']}: save_midi failed — skip")
                continue

            try:
                recon_score = symusic.Score(str(recon_path))
                cp, _ = compute_CP(orig_score, recon_score, bar_range)
                gs, _ = compute_GS(orig_score, recon_score, bar_range, tpq)
                pche = compute_PCHE(orig_score, recon_score, bar_range)
            except Exception as e:  # noqa: BLE001
                print(f"  seed {seed_idx} / {cond['name']}: metric failure ({e}) — skip")
                continue

            rows.append({"condition": cond["name"], "seed_idx": seed_idx,
                         "cp": cp, "gs": gs, "pche": pche})
            agg[cond["name"]]["cp"].append(cp)
            agg[cond["name"]]["gs"].append(gs)
            agg[cond["name"]]["pche"].append(pche)

        print(f"  seed {seed_idx} done  ({file_path.name})")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = out / "decoding_ablation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "seed_idx", "cp", "gs", "pche"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {csv_path}  ({len(rows)} rows)")

    # ── summary table ─────────────────────────────────────────────────────────
    def ms(xs):
        a = np.asarray(xs, float)
        return (f"{a.mean():.3f} ± {a.std(ddof=1):.3f}"
                if a.size > 1 else (f"{a.mean():.3f} ± 0.000" if a.size == 1 else "N/A"))

    print("\n" + "=" * 78)
    print(f"  {'Condition':<20} | {'CP (mean±std)':<15} | {'GS (mean±std)':<15} | {'PCHE (mean±std)'}")
    print("  " + "-" * 74)
    for cond in conditions:
        name = cond["name"]
        d = agg.get(name, {"cp": [], "gs": [], "pche": []})
        tag = "  ← baseline" if name == "argmax" else ""
        label = "argmax (baseline)" if name == "argmax" else name
        print(f"  {label:<20} | {ms(d['cp']):<15} | {ms(d['gs']):<15} | {ms(d['pche'])}{tag}")
    print("=" * 78)


if __name__ == "__main__":
    main()
