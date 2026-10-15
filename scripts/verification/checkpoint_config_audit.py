"""
scripts/verification/checkpoint_config_audit.py

READ-ONLY. Verifies the EXACT training configuration of the two thesis
checkpoints by reading their hyper_parameters directly (authoritative), then
hunts SLURM logs / W&B config snapshots for the Trainer-level params that are
NOT stored in the checkpoint (batch_size, accumulate_grad_batches,
limit_train_batches). Writes checkpoint_config_audit.md.

Nothing is modified.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

TRAINER_PATTERNS = re.compile(
    r"(batch_size|accumulate_grad_batches|limit_train_batches|"
    r"total_training_steps|num_training_steps|steps/epoch|Batch size|effective batch)",
    re.IGNORECASE,
)
LOG_EXTS = {".txt", ".out", ".err", ".log", ".yaml", ".yml"}
MAX_FILE_BYTES = 8_000_000
MAX_HITS_PER_FILE = 12


def dump_ckpt(path: Path):
    d = torch.load(path, map_location="cpu")
    keys = list(d.keys())
    hp = d.get("hyper_parameters") or d.get("hparams") or {}
    return keys, dict(hp)


def scan_logs(dirs, patterns=TRAINER_PATTERNS):
    hits = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for f in sorted(d.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in LOG_EXTS:
                continue
            try:
                if f.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = f.read_text(errors="ignore")
            except Exception:
                continue
            n = 0
            for i, line in enumerate(text.splitlines(), 1):
                if patterns.search(line):
                    hits.append(f"{f}:{i}: {line.strip()[:200]}")
                    n += 1
                    if n >= MAX_HITS_PER_FILE:
                        break
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1_ckpt", required=True)
    ap.add_argument("--infill_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--conf_dir", default=str(Path.home() / "MIDIT" / "conf"))
    ap.add_argument("--log_dirs", nargs="+", default=[
        str(Path.home() / "MIDIT" / "logs"),
        "/nfsd/tesi/zanettisim/MIDIT/logs",
        str(Path.home() / "MIDIT" / "wandb"),
        "/nfsd/tesi/zanettisim/MIDIT/logs_v2",
    ])
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    L = ["# MIDIT — Checkpoint configuration audit\n"]

    ckpts = [("Phase-1", args.phase1_ckpt), ("Phase-1b", args.infill_ckpt)]
    hp_store = {}
    for name, path in ckpts:
        L.append(f"## {name} checkpoint hparams\n")
        L.append(f"`{path}`\n")
        try:
            keys, hp = dump_ckpt(Path(path))
        except Exception as e:  # noqa: BLE001
            L.append(f"**LOAD FAILED:** {e}\n")
            hp_store[name] = None
            continue
        hp_store[name] = hp
        L.append(f"- top-level checkpoint keys: `{keys}`")
        L.append("- **hyper_parameters (all keys):**\n")
        L.append("```")
        for k in sorted(hp.keys()):
            v = hp[k]
            vs = str(v)
            if len(vs) > 300:
                vs = vs[:300] + " …(truncated)"
            L.append(f"{k} = {vs}")
        L.append("```")
        # trainer-level presence
        present = {k: (k in hp) for k in ("batch_size", "accumulate_grad_batches", "limit_train_batches")}
        L.append(f"- batch_size in hparams: **{present['batch_size']}**  ·  "
                 f"accumulate_grad_batches: **{present['accumulate_grad_batches']}**  ·  "
                 f"limit_train_batches: **{present['limit_train_batches']}**  "
                 f"(these are Trainer/DataLoader params; expected ABSENT).")
        L.append(f"- num_training_steps = **{hp.get('num_training_steps', '(absent)')}**  ·  "
                 f"warmup_steps = **{hp.get('warmup_steps', '(absent)')}**  ·  "
                 f"lr = **{hp.get('lr', '(absent)')}**  ·  "
                 f"weight_decay = **{hp.get('weight_decay', '(absent)')}**")
        lam = hp.get("lambda_weights")
        L.append("- **lambda_weights (authoritative):**")
        if isinstance(lam, dict):
            L.append("```")
            for k in ("pitch", "velocity", "duration", "program", "position", "bar", "tempo", "timesig", "rest"):
                L.append(f"{k} = {lam.get(k, '(absent)')}")
            L.append("```")
        else:
            L.append(f"  {lam}")
        L.append("")

    # ── conf/ listing ─────────────────────────────────────────────────────────
    L.append("## Config files in conf/\n")
    cd = Path(args.conf_dir)
    if cd.exists():
        for f in sorted(cd.rglob("*.yaml")):
            L.append(f"- `{f.relative_to(cd)}`")
    else:
        L.append(f"(conf dir not found: {cd})")
    L.append("")

    # ── log / wandb search for trainer params ────────────────────────────────
    L.append("## SLURM log / W&B search (batch_size, accumulate, limit_train_batches)\n")
    hits = scan_logs(args.log_dirs)
    if hits:
        L.append("```")
        L.extend(hits[:120])
        L.append("```")
    else:
        L.append("_No matching lines found in the searched dirs: "
                 + ", ".join(args.log_dirs) + "._")
    L.append("")

    # ── verdict scaffold ──────────────────────────────────────────────────────
    L.append("## Verdict\n")
    for name in ("Phase-1", "Phase-1b"):
        hp = hp_store.get(name) or {}
        nts = hp.get("num_training_steps", "?")
        L.append(f"- **{name}** — num_training_steps (ckpt) = {nts}, "
                 f"warmup_steps = {hp.get('warmup_steps','?')}. "
                 f"batch_size / accumulate / limit_train_batches are NOT in the ckpt — "
                 f"fill from the log/config lines above.")
    L.append("\n_Note: num_training_steps = (limit_train_batches // accumulate) × epochs, "
             "so it lets you back out limit/accumulate given epochs; the log lines above "
             "should also state them verbatim (train_phase1.py prints "
             "'total_training_steps: N (B batches // A accumulate = S steps/epoch x E epochs)')._")

    md = out / "checkpoint_config_audit.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("Saved:", md)
    print("\n".join(L))


if __name__ == "__main__":
    main()
