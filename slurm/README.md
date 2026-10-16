# SLURM launchers (cluster-only)

The job scripts of the runs behind the thesis results, kept as run records. They
contain DEI cluster paths (`/home/zanettisim/MIDIT`, `/nfsd/tesi/zanettisim/...`),
partitions and GPU types, and do not run elsewhere without editing. To run a step on
another machine, use the `uv run` commands in the [root README](../README.md).

Submit from the repository root, after creating the log directories (SLURM does not
create the `--output` / `--error` directories, and the job fails without them):

```bash
mkdir -p logs/out logs/err
sbatch slurm/<launcher>.sh
```

| Launcher | Runs | Produced | Backs (thesis) |
|---|---|---|---|
| `train_phase1_v2_50k_safe.sh` | `midit/train_phase1.py` (50 epochs, W&B, `preprocessed_v2_50k`) | Phase-1 pre-training run; best checkpoint epoch 14 (`midit-phase1-epoch14.ckpt`). **Inferred:** this is the launcher in the repository when the run was set up (commit `6da32ce`, 2026-03-25); not confirmed from the job log. | Phase-1 results (cloze accuracy, zero-shot) |
| `train_infilling.sh` | `midit/train_infilling.py` with `conf/config_infilling.yaml` | Phase-1b BarFillMasking fine-tuning, `midit-infilling-epoch05.ckpt`. It passes no checkpoint override; the config pointed at epoch 14 then and points at `epoch=49` now (see [`conf/README.md`](../conf/README.md)). | all fine-tuned results |
| `generate_samples.sh` | `scripts/inference/generate_samples_batch.py`, fine-tuned checkpoint, bars 3–5 | fine-tuned evaluation sample set (`samples_1000`, 989 samples) | Tables 5.15, 5.17, 5.18 |
| `generate_samples_zs.sh` | `scripts/inference/generate_samples_batch.py`, bars 3–5 | zero-shot sample set (`samples_zs_1000`), **generated from the Phase-1 `epoch=49` checkpoint** (see [`scripts/README.md`](../scripts/README.md)) | Table 5.15 (zero-shot, via `scripts/evaluation/js/run_js_zs.sh`) |
| `extreme_infilling.sh` | `midit/infer.py`, seeds 1–5 and 10, first/last 1–3 bars | extreme-infilling outputs | Table 5.11, Figure 5.5 (see [`docs/ERRATA.md`](../docs/ERRATA.md)) |
| `decoding_ablation.sh` | `scripts/evaluation/decoding/decoding_ablation.py`, 50 seeds | decoding-strategy ablation | Table 5.13 |
| `analyze_dataset.sh` | `scripts/analysis/analyze_preprocessed_v2_dataset.py` | statistics of the preprocessed V2 dataset | Chapter 3 dataset analysis |
| `thesis_analysis.sh` | `scripts/analysis/thesis_mini_analyses.py` | per-class frequencies, loss-weight audit, sequence lengths | Chapters 3–4 analyses |
| `sota_baselines/generate_anticipatory_samples.sh` | `scripts/evaluation/baselines/anticipatory/generate_anticipatory_samples.py` (separate venv) | Anticipatory Music Transformer samples | Tables 5.15, 5.17, 5.18 |

No recorded launcher exists for `scripts/preprocessing/preprocess_v2.py` (the V2
dataset) or for the cloze evaluation
(`scripts/evaluation/evaluate_phase1_accuracy_DIRECT.py`).

The other launchers (project templates, test runs, post-thesis and V1 runs, audit
jobs) are preserved at the `archive-full-tree` tag.
