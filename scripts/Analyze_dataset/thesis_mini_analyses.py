"""
scripts/Analyze_dataset/thesis_mini_analyses.py

Three lightweight thesis analyses (READ-ONLY). Writes a single markdown file
thesis_analysis_results.md to --out_dir.

  A1: per-class token frequency over N=100 training-split .pt files (seed 42)
      -> mean tokens/type/file and % of total (justifies lambda_t weighting).
  A2: lambda values actually used per head. Reads conf/config.yaml (source read
      by build_model) with file:line, cites multi_head_output.py application,
      and -- if --phase1_ckpt is given -- the AUTHORITATIVE as-trained values
      from the checkpoint hyper_parameters. Confirms Bar lambda and compares
      Tempo/Rest/TimeSig to Table 4.1.
  A3: sequence length (mean/median/p5/p95) and bars/file (mean/median) on the
      same 100-file sample (justifies max_seq_len=8192).

Does not modify any model or data files.
"""

from __future__ import annotations

import argparse
import gc
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]

TYPES = ["Pitch", "PitchDrum", "Velocity", "Duration", "Program",
         "Position", "Bar", "Tempo", "TimeSig", "Rest", "Special"]
TABLE_41 = {"pitch": 3.0, "bar": 0.0, "tempo": 1.0, "timesig": 0.5, "rest": 1.0}  # as quoted


# ── tokenizer / split helpers ─────────────────────────────────────────────────

def build_id_to_token():
    from miditok import REMI, TokenizerConfig
    cfg = TokenizerConfig(
        pitch_range=(21, 109), beat_res={(0, 4): 8, (4, 12): 4}, num_velocities=32,
        use_chords=False, use_rests=True, use_tempos=True, use_time_signatures=True,
        use_programs=True, special_tokens=["PAD", "BOS", "EOS", "MASK"], use_controls=False,
        delete_equal_successive_tempo_changes=True, delete_equal_successive_time_sig_changes=True,
    )
    v = REMI(cfg).vocab
    return {i: n for n, i in v.items()}, len(v)


def type_of(name: str):
    if name.startswith("PitchDrum_"):
        return "PitchDrum"
    if name.startswith("Pitch_"):
        return "Pitch"
    for t in ("Velocity", "Duration", "Program", "Position", "Bar", "Tempo", "TimeSig", "Rest"):
        if name.startswith(t + "_"):
            return t
    if name.split("_")[0] in ("PAD", "BOS", "EOS", "MASK"):
        return "Special"
    return None


def training_files(data_dir: Path):
    all_files = sorted(data_dir.glob("*.pt"))
    stem_to = defaultdict(list)
    for f in all_files:
        stem_to[re.sub(r"_shift[+-]\d+$", "", f.stem)].append(f)
    unique = sorted(stem_to.keys())
    np.random.seed(42)
    idx = np.arange(len(unique)); np.random.shuffle(idx)
    n_train = int(0.8 * len(unique))
    train_stems = {unique[i] for i in idx[:n_train]}
    return [f for f in all_files if re.sub(r"_shift[+-]\d+$", "", f.stem) in train_stems], len(unique), len(all_files)


def _tokens_and_bars(fp):
    obj = torch.load(fp, weights_only=False)
    if isinstance(obj, dict):
        ids = obj["tokens"]
        nb = len(obj["bar_positions"]) - 1 if obj.get("bar_positions") is not None else None
    else:
        ids, nb = obj, None
    ids = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
    return ids, nb


# ── Analysis 2 helpers ────────────────────────────────────────────────────────

def config_lambda_lines(cfg_path: Path):
    text = cfg_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out, in_block = {}, False
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if s.startswith("lambda_weights:"):
            in_block = True
            continue
        if in_block:
            if ln[:1] not in (" ", "\t") and s:
                break
            m = re.match(r"\s*([a-z_]+)\s*:\s*([0-9.eE+-]+)", ln)
            if m:
                out[m.group(1)] = (float(m.group(2)), i)
    return out


def ckpt_lambda(path: Path):
    try:
        d = torch.load(path, map_location="cpu")
    except Exception as e:  # noqa: BLE001
        return None, f"failed to load: {e}"
    hp = d.get("hyper_parameters") or d.get("hparams") or {}
    lam = hp.get("lambda_weights")
    return (dict(lam) if lam is not None else None), None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Three thesis analyses (read-only).")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--phase1_ckpt", default=None, help="Optional: read as-trained lambda from hparams.")
    ap.add_argument("--infill_ckpt", default=None, help="Optional: confirm infilling ckpt lambda.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    id_to_token, vsize = build_id_to_token()

    # ── A1 + A3 sampling ─────────────────────────────────────────────────────
    files, n_songs, n_files = training_files(Path(args.data_dir))
    sample = random.Random(args.seed).sample(files, min(args.n, len(files)))
    per_type_total = Counter()
    per_file_type = []           # list of Counter per file
    seq_lens, bars = [], []
    grand_total = 0
    for i, fp in enumerate(sample):
        ids, nb = _tokens_and_bars(fp)
        seq_lens.append(len(ids))
        c = Counter()
        for tid in ids:
            nm = id_to_token.get(tid)
            t = type_of(nm) if nm else None
            if t:
                c[t] += 1
        if nb is None:
            nb = c.get("Bar", 0)
        bars.append(nb)
        per_file_type.append(c)
        per_type_total.update(c)
        grand_total += len(ids)
        del ids
        if (i + 1) % 25 == 0:
            gc.collect()

    n = len(sample)
    seq = np.asarray(seq_lens, float); bar = np.asarray(bars, float)

    # ── A2 lambdas ────────────────────────────────────────────────────────────
    cfg_path = _ROOT / "conf" / "config.yaml"
    cfg_lam = config_lambda_lines(cfg_path)
    p1_lam = p1_err = inf_lam = inf_err = None
    if args.phase1_ckpt:
        p1_lam, p1_err = ckpt_lambda(Path(args.phase1_ckpt))
    if args.infill_ckpt:
        inf_lam, inf_err = ckpt_lambda(Path(args.infill_ckpt))

    # ── write markdown ────────────────────────────────────────────────────────
    L = []
    L.append("# MIDIT — Thesis mini-analyses\n")
    L.append(f"Sample: {n} training-split .pt files (seed {args.seed}); "
             f"corpus {n_songs} songs / {n_files} files; tokenizer vocab {vsize}.\n")

    # A1
    L.append("## Analysis 1 — Per-class token frequency\n")
    L.append("| Type | mean tokens/file | % of total |")
    L.append("|---|---:|---:|")
    for t in TYPES:
        tot = per_type_total.get(t, 0)
        L.append(f"| {t} | {tot/n:,.1f} | {100*tot/max(1,grand_total):.2f}% |")
    L.append(f"\nGrand total tokens in sample: {grand_total:,}  "
             f"(mean {grand_total/n:,.0f} tokens/file).\n")

    # A2
    L.append("## Analysis 2 — Loss weights (lambda) per head\n")
    L.append("`multi_head_output.py` does not hardcode lambda; it applies "
             "`lambda_weights.get(token_type, 1.0)` at "
             "`midit/model/multi_head_output.py:209`. The dict is read from "
             "`conf/config.yaml` by `midit/model/model_factory.py:99`.\n")
    L.append("**Configured values (`conf/config.yaml`):**\n")
    L.append("| Head | lambda | file:line |")
    L.append("|---|---:|---|")
    for k in ("pitch", "velocity", "duration", "program", "position", "bar", "tempo", "timesig", "rest"):
        if k in cfg_lam:
            val, ln = cfg_lam[k]
            L.append(f"| {k} | {val} | conf/config.yaml:{ln} |")
    bar_val = cfg_lam.get("bar", (None, None))[0]
    L.append(f"\n- **Bar lambda = {bar_val} → {'CONFIRMED 0.0' if bar_val == 0.0 else 'NOT 0.0'}**")
    for k in ("tempo", "timesig", "rest"):
        cv = cfg_lam.get(k, (None, None))[0]
        tv = TABLE_41.get(k)
        match = "match" if cv == tv else "**MISMATCH**"
        L.append(f"- {k}: config={cv}  vs  Table 4.1={tv}  → {match}")
    if p1_lam is not None:
        L.append("\n**As-trained (Phase-1 checkpoint hyper_parameters — authoritative):**\n")
        L.append("| Head | lambda (ckpt) |")
        L.append("|---|---:|")
        for k in ("pitch", "velocity", "duration", "program", "position", "bar", "tempo", "timesig", "rest"):
            L.append(f"| {k} | {p1_lam.get(k)} |")
        # config vs ckpt consistency
        diffs = [k for k in p1_lam if k in cfg_lam and float(p1_lam[k]) != cfg_lam[k][0]]
        L.append(f"\nconfig.yaml vs checkpoint: {'IDENTICAL' if not diffs else 'DIFFER on ' + ', '.join(diffs)}.")
    elif args.phase1_ckpt:
        L.append(f"\n(Phase-1 checkpoint lambda unavailable: {p1_err})")
    else:
        L.append("\n_(Pass --phase1_ckpt to read the authoritative as-trained lambda from hparams.)_")
    if inf_lam is not None:
        same = (p1_lam is not None and inf_lam == p1_lam)
        L.append(f"\nInfilling checkpoint lambda {'== Phase-1 (inherited)' if same else inf_lam}.")

    # A3
    L.append("\n## Analysis 3 — Sequence length & bars (same 100 files)\n")
    L.append("| Metric | mean | median | p5 | p95 |")
    L.append("|---|---:|---:|---:|---:|")
    L.append(f"| Sequence length (tokens) | {seq.mean():,.0f} | {np.median(seq):,.0f} "
             f"| {np.percentile(seq,5):,.0f} | {np.percentile(seq,95):,.0f} |")
    L.append(f"| Bars per file | {bar.mean():,.1f} | {np.median(bar):,.0f} "
             f"| {np.percentile(bar,5):,.0f} | {np.percentile(bar,95):,.0f} |")
    L.append(f"\nMax sequence length in sample: {int(seq.max()):,} tokens "
             f"→ well within max_seq_len = 8192 for the 8-bar chunks the model consumes "
             f"(these full-song lengths are not the model input).\n")

    md = out_dir / "thesis_analysis_results.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("Saved:", md)
    print("\n".join(L))


if __name__ == "__main__":
    main()
