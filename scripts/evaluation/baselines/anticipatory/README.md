# Anticipatory Music Transformer — SOTA Baseline (Fase B)

Batch infilling inference with the **Anticipatory Music Transformer (AMT)**,
used as a state-of-the-art baseline for the symbolic-music infilling comparison
in the MIDIT thesis. This is **experimental comparison code**, not part of the
core MIDIT model or training pipeline.

**Reference:** Thickstun et al., *Anticipatory Music Transformer*,
NeurIPS 2023 — [arXiv:2306.08620](https://arxiv.org/abs/2306.08620).
Code: <https://github.com/jthickstun/anticipation>.

---

## Setup

AMT pulls in its own dependency set; install it in a **separate virtualenv** on
the cluster so it never touches the main MIDIT environment (`.venv`).

```bash
cd ~/MIDIT
python -m venv .venv_anticipation
source .venv_anticipation/bin/activate
git clone https://github.com/jthickstun/anticipation.git /tmp/anticipation
pip install /tmp/anticipation
pip install torch transformers huggingface_hub mido tqdm
```

> Do **not** install these into the main MIDIT `.venv`.

---

## How to run

```bash
sbatch slurm/sota_baselines/generate_anticipatory_samples.sh
```

The script is **idempotent / resumable**: samples whose `reconstructed.mid`
already exists are skipped, so a requeued job continues where it stopped.

- **Input:**  `/nfsd/tesi/zanettisim/MIDIT/outputs/samples_1000/seed_N/original.mid`
- **Output:** `/nfsd/tesi/zanettisim/MIDIT/outputs/samples_anticipatory_1000/seed_N/reconstructed.mid`

Per-sample inference is ~10 s on an L40S; the full ~989-sample run takes
roughly 3 hours. The model uses ~3–4 GB VRAM.

---

## What the script does

For each `seed_N/`:

1. Read `original.mid`, extract the first tempo (default `500000` µs/beat if
   none present).
2. Compute `bar_seconds = (tempo / 1_000_000) * 4` (4/4 assumed).
3. Mask span = bars `mask_bars[0]` … `mask_bars[-1]`
   (`t1 = mask_bars[0] * bar_seconds`, `t2 = (mask_bars[-1] + 1) * bar_seconds`).
4. Supply the pre-mask prefix as `history` and the post-mask context (up to
   `min(8 * bar_seconds, total_duration)`) as **anticipated controls**.
5. Generate the masked span and save the reconstruction.

Files shorter than `0.9 × (8 * bar_seconds)` are skipped (logged as "too short").

---

## Methodological note

Anticipatory MT **re-encodes the prefix during generation** — a different
paradigm from MIDIT's *mask-preserve* approach, where the unmasked context is
held fixed. As a result, `generate()` may emit events spanning
`[~bar_1 → t2]` rather than only the masked region; this is expected AMT
behavior and is **not** compensated for here.

Because of this paradigm difference, metrics are computed **only on the masked
region**, isolated via `compute_model_agg.py`'s `_detect_masked_bars()`, so the
two systems are compared on the same target span.
