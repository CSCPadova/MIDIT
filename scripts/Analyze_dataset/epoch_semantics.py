"""
scripts/Analyze_dataset/epoch_semantics.py

READ-ONLY. Computes the exact training-loop / epoch-coverage numbers for the
thesis and writes /out_dir/epoch_semantics.md.

- Batch/accumulate/limit values are read from the config files (the authoritative
  source; they are NOT stored in the model checkpoint).
- Total training chunks is computed with the SAME logic training uses, by
  constructing LakhDataset(split='train', apply_masking=False, preload=False)
  and reading len(dataset.chunks).
- num_training_steps / warmup_steps are read from the checkpoint hyper_parameters
  as a cross-check on the schedule.

Nothing is modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

import sys
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from midit.data.lakh_dataset import LakhDataset   # noqa: E402


def ckpt_field(path, key):
    try:
        d = torch.load(path, map_location="cpu")
    except Exception as e:  # noqa: BLE001
        return f"(load failed: {e})"
    hp = d.get("hyper_parameters") or d.get("hparams") or {}
    return hp.get(key, "(absent)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--phase1_ckpt", default=None)
    ap.add_argument("--infill_ckpt", default=None)
    # config-derived values (defaults match the repo configs; override if needed)
    ap.add_argument("--p1_batch", type=int, default=8)
    ap.add_argument("--p1_accum", type=int, default=4)
    ap.add_argument("--p1_limit", type=int, default=25000)
    ap.add_argument("--p1_epochs", type=int, default=50)
    ap.add_argument("--p1_ckpt_epoch", type=int, default=14)
    ap.add_argument("--p1b_batch", type=int, default=8)
    ap.add_argument("--p1b_accum", type=int, default=4)
    ap.add_argument("--p1b_limit", type=int, default=50000)
    ap.add_argument("--p1b_epochs", type=int, default=10)
    ap.add_argument("--p1b_ckpt_epoch", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ── exact chunk total via the training code path ──────────────────────────
    cfg = OmegaConf.create({
        "chunking": {"preprocessed_dir": args.data_dir, "strategy": "bar_based",
                     "bars_per_chunk": 8, "tokens_per_chunk": 4096, "max_tokens": 8192},
        "masking": {},
    })
    ds = LakhDataset(cfg, split="train", apply_masking=False, preload=False)
    n_files = len(ds.files)
    total_chunks = len(ds.chunks)
    avg_cpf = total_chunks / max(1, n_files)

    def block(name, batch, accum, limit, epochs, ckpt_epoch, ckpt_path):
        eff = batch * accum
        opt_steps = limit // accum
        chunks_ep = limit * batch
        cov = 100.0 * chunks_ep / max(1, total_chunks)
        reps = ckpt_epoch * chunks_ep / max(1, total_chunks)
        nts = ckpt_field(ckpt_path, "num_training_steps") if ckpt_path else "(no ckpt)"
        wus = ckpt_field(ckpt_path, "warmup_steps") if ckpt_path else "(no ckpt)"
        return dict(name=name, batch=batch, accum=accum, eff=eff, limit=limit,
                    opt_steps=opt_steps, epochs=epochs, ckpt_epoch=ckpt_epoch,
                    chunks_ep=chunks_ep, cov=cov, reps=reps,
                    planned_steps=opt_steps * epochs, ckpt_nts=nts, ckpt_wus=wus)

    p1 = block("Phase 1", args.p1_batch, args.p1_accum, args.p1_limit,
               args.p1_epochs, args.p1_ckpt_epoch, args.phase1_ckpt)
    p1b = block("Phase 1b", args.p1b_batch, args.p1b_accum, args.p1b_limit,
                args.p1b_epochs, args.p1b_ckpt_epoch, args.infill_ckpt)

    L = []
    L.append("# MIDIT — Epoch semantics & dataset coverage (exact)\n")
    L.append(f"Training files (song-level split): **{n_files:,}**  ·  "
             f"total 8-bar chunks (stride-1 overlapping): **{total_chunks:,}**  ·  "
             f"avg **{avg_cpf:.1f}** chunks/file.\n")
    L.append("_Chunk total from LakhDataset(split='train', preload=False), the same "
             "logic used in training (lakh_dataset.py:255-275)._\n")

    for p in (p1, p1b):
        L.append(f"## {p['name']}\n")
        L.append(f"- physical batch = **{p['batch']}**, accumulate = **{p['accum']}** "
                 f"→ effective batch = **{p['eff']}**")
        L.append(f"- limit_train_batches = **{p['limit']:,}** (dataloader batches) "
                 f"→ optimizer steps/epoch = **{p['opt_steps']:,}**")
        L.append(f"- chunks (samples) per epoch = limit × batch = **{p['chunks_ep']:,}**")
        L.append(f"- coverage per epoch = **{p['cov']:.2f}%** of all chunks")
        L.append(f"- planned optimizer steps (epochs×steps) = **{p['planned_steps']:,}**  "
                 f"| checkpoint num_training_steps = **{p['ckpt_nts']}**, "
                 f"warmup_steps = **{p['ckpt_wus']}**")
        L.append(f"- over {p['ckpt_epoch']} epochs (checkpoint): expected repetitions "
                 f"per chunk ≈ **{p['reps']:.3f}**  (i.e. ~{100*p['reps']:.1f}% cumulative chunk-visits)\n")

    L.append("## Summary table\n")
    L.append("| Phase | phys batch | accum | eff batch | limit (batches) | opt steps/epoch | "
             "chunks/epoch | coverage/epoch | ckpt epoch | cov×epochs |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for p in (p1, p1b):
        L.append(f"| {p['name']} | {p['batch']} | {p['accum']} | {p['eff']} | {p['limit']:,} | "
                 f"{p['opt_steps']:,} | {p['chunks_ep']:,} | {p['cov']:.2f}% | "
                 f"{p['ckpt_epoch']} | {100*p['reps']:.1f}% |")

    L.append("\n**Notes.** limit_train_batches counts dataloader batches (Lightning "
             "≥2.5.5); an epoch is a fixed segment of limit×batch chunks, not a full "
             "dataset pass. Chunks are stride-1 overlapping (lakh_dataset.py:269); "
             "batch_size/accumulate are config-only (not in the checkpoint).")

    md = out / "epoch_semantics.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("Saved:", md)
    print("\n".join(L))


if __name__ == "__main__":
    main()
