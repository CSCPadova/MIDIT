"""
scripts/Analyze_dataset/analyze_preprocessed_v2_dataset.py

Chapter-3 dataset analysis of the MIDIT preprocessed training data (READ-ONLY).
Samples N=5000 .pt files from the TRAINING split (song-level split replicated
from lakh_dataset.py, seed=42), computes per-file statistics + token-type and
GM-program frequencies, and emits 4 thesis-quality plots + a per-file CSV.

Does NOT load all files at once; does NOT modify any data.

Usage (cluster):
    python scripts/Analyze_dataset/analyze_preprocessed_v2_dataset.py \
        --data_dir /nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_50k \
        --out_dir  /nfsd/tesi/zanettisim/MIDIT/outputs/dataset_analysis \
        --n 5000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import gc
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

# General MIDI program names (program id 0-127); -1 = drum channel.
GM_NAMES = [
    "Acoustic Grand Piano","Bright Acoustic Piano","Electric Grand Piano","Honky-tonk Piano",
    "Electric Piano 1","Electric Piano 2","Harpsichord","Clavi","Celesta","Glockenspiel",
    "Music Box","Vibraphone","Marimba","Xylophone","Tubular Bells","Dulcimer","Drawbar Organ",
    "Percussive Organ","Rock Organ","Church Organ","Reed Organ","Accordion","Harmonica",
    "Tango Accordion","Acoustic Guitar (nylon)","Acoustic Guitar (steel)","Electric Guitar (jazz)",
    "Electric Guitar (clean)","Electric Guitar (muted)","Overdriven Guitar","Distortion Guitar",
    "Guitar harmonics","Acoustic Bass","Electric Bass (finger)","Electric Bass (pick)",
    "Fretless Bass","Slap Bass 1","Slap Bass 2","Synth Bass 1","Synth Bass 2","Violin","Viola",
    "Cello","Contrabass","Tremolo Strings","Pizzicato Strings","Orchestral Harp","Timpani",
    "String Ensemble 1","String Ensemble 2","SynthStrings 1","SynthStrings 2","Choir Aahs",
    "Voice Oohs","Synth Voice","Orchestra Hit","Trumpet","Trombone","Tuba","Muted Trumpet",
    "French Horn","Brass Section","SynthBrass 1","SynthBrass 2","Soprano Sax","Alto Sax",
    "Tenor Sax","Baritone Sax","Oboe","English Horn","Bassoon","Clarinet","Piccolo","Flute",
    "Recorder","Pan Flute","Blown Bottle","Shakuhachi","Whistle","Ocarina","Lead 1 (square)",
    "Lead 2 (sawtooth)","Lead 3 (calliope)","Lead 4 (chiff)","Lead 5 (charang)","Lead 6 (voice)",
    "Lead 7 (fifths)","Lead 8 (bass + lead)","Pad 1 (new age)","Pad 2 (warm)","Pad 3 (polysynth)",
    "Pad 4 (choir)","Pad 5 (bowed)","Pad 6 (metallic)","Pad 7 (halo)","Pad 8 (sweep)",
    "FX 1 (rain)","FX 2 (soundtrack)","FX 3 (crystal)","FX 4 (atmosphere)","FX 5 (brightness)",
    "FX 6 (goblins)","FX 7 (echoes)","FX 8 (sci-fi)","Sitar","Banjo","Shamisen","Koto","Kalimba",
    "Bag pipe","Fiddle","Shanai","Tinkle Bell","Agogo","Steel Drums","Woodblock","Taiko Drum",
    "Melodic Tom","Synth Drum","Reverse Cymbal","Guitar Fret Noise","Breath Noise","Seashore",
    "Bird Tweet","Telephone Ring","Helicopter","Applause","Gunshot",
]

SEMANTIC_TYPES = ("pitch", "velocity", "duration", "program", "position",
                  "bar", "tempo", "timesig", "rest")


def gm_name(pid: int) -> str:
    if pid == -1:
        return "Drums"
    return GM_NAMES[pid] if 0 <= pid < len(GM_NAMES) else f"Program_{pid}"


def build_tokenizer_vocab():
    """REMI+ V2 tokenizer vocab, identical config to preprocess_v2.py."""
    from miditok import REMI, TokenizerConfig
    cfg = TokenizerConfig(
        pitch_range=(21, 109), beat_res={(0, 4): 8, (4, 12): 4}, num_velocities=32,
        use_chords=False, use_rests=True, use_tempos=True, use_time_signatures=True,
        use_programs=True, special_tokens=["PAD", "BOS", "EOS", "MASK"], use_controls=False,
        delete_equal_successive_tempo_changes=True, delete_equal_successive_time_sig_changes=True,
    )
    vocab = REMI(cfg).vocab
    return {v: k for k, v in vocab.items()}, len(vocab)


def get_stem(path: Path) -> str:
    return re.sub(r"_shift[+-]\d+$", "", path.stem)


def training_files(data_dir: Path) -> list:
    """Replicate lakh_dataset.py song-level split (seed 42, 80/10/10) -> train files."""
    all_files = sorted(data_dir.glob("*.pt"))
    if not all_files:
        raise SystemExit(f"No .pt files under {data_dir}")
    stem_to_files = defaultdict(list)
    for f in all_files:
        stem_to_files[get_stem(f)].append(f)
    unique = sorted(stem_to_files.keys())
    np.random.seed(42)
    idx = np.arange(len(unique)); np.random.shuffle(idx)
    n_train = int(0.8 * len(unique))
    train_stems = {unique[i] for i in idx[:n_train]}
    files = [f for f in all_files if get_stem(f) in train_stems]
    print(f"[split] {len(unique)} stems, {len(all_files)} files, "
          f"train stems={n_train}, train files={len(files)}")
    return files


def type_of(name: str) -> str | None:
    if name.startswith(("Pitch_", "PitchDrum_")):
        return "pitch"
    for t in ("velocity", "duration", "program", "position", "bar", "tempo", "timesig", "rest"):
        if name.lower().startswith(t):
            return t
    return None


def _tokens_from_pt(obj) -> list:
    """Handle both dict payloads ({'tokens':...}) and bare tensors/lists."""
    if isinstance(obj, dict):
        obj = obj.get("tokens")
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return list(obj)


def analyze(files, id_to_token, n, seed):
    rng = random.Random(seed)
    sample = rng.sample(files, min(n, len(files)))
    print(f"[sample] {len(sample)} files (seed={seed})")

    rows = []
    type_freq = Counter()
    program_freq = Counter()
    n_bad = 0

    for i, fp in enumerate(sample):
        try:
            ids = _tokens_from_pt(torch.load(fp, weights_only=False))
        except Exception as e:  # noqa: BLE001
            n_bad += 1
            if n_bad <= 5:
                print(f"  [warn] failed {fp.name}: {e}")
            continue

        n_bar = n_pitch = n_drum = 0
        programs = set()
        tempo_ctr = Counter()
        for tid in ids:
            name = id_to_token.get(tid)
            if name is None:
                continue
            t = type_of(name)
            if t is None:
                continue
            type_freq[t] += 1
            if t == "bar":
                n_bar += 1
            elif t == "pitch":
                n_pitch += 1
                if name.startswith("PitchDrum_"):
                    n_drum += 1
            elif t == "program":
                try:
                    pid = int(name.split("_", 1)[1])
                except ValueError:
                    continue
                programs.add(pid)
                program_freq[pid] += 1
            elif t == "tempo":
                tempo_ctr[name.split("_", 1)[1]] += 1

        seq_len = len(ids)
        note_density = n_pitch / n_bar if n_bar > 0 else float("nan")
        frac_drum = n_drum / n_pitch if n_pitch > 0 else float("nan")
        top_tempo = tempo_ctr.most_common(1)[0][0] if tempo_ctr else ""
        rows.append({
            "file": fp.name, "seq_len": seq_len, "n_bars": n_bar,
            "n_instruments": len(programs), "n_notes": n_pitch,
            "note_density": note_density, "frac_drum": frac_drum,
            "top_tempo_bin": top_tempo,
        })
        del ids
        if (i + 1) % 500 == 0:
            gc.collect()
            print(f"  ...{i+1}/{len(sample)}", flush=True)

    print(f"[done] ok={len(rows)} bad={n_bad}")
    return rows, type_freq, program_freq


def _summ(vals):
    v = np.asarray([x for x in vals if x is not None and not np.isnan(x)], float)
    if v.size == 0:
        return dict(n=0, mean=float("nan"), median=float("nan"), std=float("nan"),
                    p5=float("nan"), p95=float("nan"))
    return dict(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)),
                std=float(v.std(ddof=1)) if v.size > 1 else float("nan"),
                p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)))


def main() -> None:
    ap = argparse.ArgumentParser(description="MIDIT preprocessed dataset analysis (read-only).")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    id_to_token, vsize = build_tokenizer_vocab()
    print(f"[tokenizer] vocab size = {vsize}")

    files = training_files(data_dir)
    rows, type_freq, program_freq = analyze(files, id_to_token, args.n, args.seed)
    if not rows:
        raise SystemExit("No files analyzed.")

    # ── per-file CSV ──────────────────────────────────────────────────────────
    csv_path = out_dir / "per_file_stats.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[saved] {csv_path}")

    # ── numeric summaries ─────────────────────────────────────────────────────
    numeric = ["seq_len", "n_bars", "n_instruments", "n_notes", "note_density", "frac_drum"]
    print("\n=== NUMERIC SUMMARY (per file) ===")
    print(f"{'var':<14} {'n':>6} {'mean':>10} {'median':>10} {'std':>10} {'p5':>10} {'p95':>10}")
    summ_lines = []
    for k in numeric:
        s = _summ([r[k] for r in rows])
        line = (f"{k:<14} {s['n']:>6} {s['mean']:>10.3f} {s['median']:>10.3f} "
                f"{s['std']:>10.3f} {s['p5']:>10.3f} {s['p95']:>10.3f}")
        print(line); summ_lines.append(line)
    (out_dir / "numeric_summary.txt").write_text(
        "var            n      mean     median      std        p5        p95\n" +
        "\n".join(summ_lines) + "\n", encoding="utf-8")

    # ── token-type frequency (PART B) ─────────────────────────────────────────
    total_tok = sum(type_freq.values())
    print("\n=== TOKEN-TYPE FREQUENCY (9 semantic types; pitch incl. PitchDrum) ===")
    tf_lines = []
    for t in SEMANTIC_TYPES:
        c = type_freq.get(t, 0)
        line = f"{t:<10} {c:>12}  ({100*c/total_tok:5.2f}%)"
        print(line); tf_lines.append(line)
    (out_dir / "token_type_frequency.txt").write_text("\n".join(tf_lines) + "\n", encoding="utf-8")

    # ── program frequency (PART B) ────────────────────────────────────────────
    prog_rows = [(pid, gm_name(pid), cnt) for pid, cnt in program_freq.most_common()]
    with open(out_dir / "program_frequency.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["program_id", "gm_name", "count"])
        w.writerows(prog_rows)
    print(f"[saved] {out_dir/'program_frequency.csv'}  ({len(prog_rows)} programs)")

    # ── PART C plots ──────────────────────────────────────────────────────────
    def _hist(vals, title, xlabel, fname, logx=False, bins=50):
        v = np.asarray([x for x in vals if x is not None and not np.isnan(x)], float)
        fig, ax = plt.subplots(figsize=(6, 4))
        if logx:
            v = v[v > 0]
            ax.hist(v, bins=np.logspace(np.log10(max(1, v.min())), np.log10(v.max()), bins))
            ax.set_xscale("log")
        else:
            ax.hist(v, bins=bins)
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("number of files")
        fig.tight_layout(); fig.savefig(out_dir / fname, dpi=300); plt.close(fig)
        print(f"[plot] {out_dir/fname}")

    _hist([r["seq_len"] for r in rows], "Sequence length distribution",
          "tokens per file (log)", "fig_seq_length.png", logx=True)
    _hist([r["note_density"] for r in rows], "Note density distribution",
          "notes per bar", "fig_note_density.png")
    _hist([r["n_instruments"] for r in rows], "Active instruments per file",
          "distinct programs", "fig_instruments.png",
          bins=range(0, max(2, max(r["n_instruments"] for r in rows)) + 2))

    top20 = prog_rows[:20]
    fig, ax = plt.subplots(figsize=(7, 6))
    names = [f"{n} ({p})" for p, n, _ in top20][::-1]
    counts = [c for _, _, c in top20][::-1]
    ax.barh(names, counts)
    ax.set_title("Top-20 most frequent GM programs")
    ax.set_xlabel("token count (5000-file sample)")
    fig.tight_layout(); fig.savefig(out_dir / "fig_top20_programs.png", dpi=300); plt.close(fig)
    print(f"[plot] {out_dir/'fig_top20_programs.png'}")

    # ── flags ─────────────────────────────────────────────────────────────────
    print("\n=== FLAGS (surprising findings) ===")
    sl = _summ([r["seq_len"] for r in rows])
    print(f"  seq_len p95/median ratio = {sl['p95']/max(1,sl['median']):.1f} "
          f"(>5 => heavy right tail)")
    drums_top = prog_rows[0] if prog_rows else None
    if drums_top:
        share = 100 * drums_top[2] / sum(program_freq.values())
        print(f"  most frequent program: {drums_top[1]} (id {drums_top[0]}) = {share:.1f}% of program tokens")
    fd = _summ([r["frac_drum"] for r in rows])
    print(f"  mean drum-note fraction = {fd['mean']:.3f}")
    print("\n[COMPLETE] outputs in", out_dir)


if __name__ == "__main__":
    main()
