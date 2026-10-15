# Script guide

This directory combines the current research workflow with preserved experiment,
diagnostic, and thesis-support material. Directory names are retained to preserve
historical commands and references. A script's placement here is not evidence that it
is safe to delete, rename, or use with a different tokenizer/checkpoint.

Run scripts from the repository root with `uv run python ...`. Commands that need a
dataset, checkpoint, raw MIDI corpus, or cluster filesystem state that requirement in
their help text or source. Do not download data or checkpoints implicitly.

## Canonical workflows

| Task | Script | Required inputs |
| --- | --- | --- |
| Infilling a user-provided MIDI file | `infer_from_midi.py` | A MIDI file and compatible checkpoint; no preprocessed dataset |
| Infilling a validation example | `../midit/infer.py` | Compatible checkpoint and preprocessed dataset |
| Batch sample generation | `generate_samples_batch.py` | Compatible checkpoint and preprocessed dataset |
| Current preprocessing pipeline | `Pre_Processing/preprocess_v2.py` | Lakh MIDI files at the script's configured input path |
| Phase 1 training | `../midit/train_phase1.py` | Preprocessed dataset and configured accelerator |
| Phase 1b infilling fine-tuning | `../midit/train_infilling.py` | Base checkpoint, preprocessed dataset, and configured accelerator |
| Cloze evaluation | `Evaluation_Accuracy/evaluate_phase1_accuracy_DIRECT.py` | Compatible checkpoint and preprocessed dataset |
| Infilling metric report | `Evaluation_Infilling/compute_muspy_metrics.py` | Generated sample directory |

The root [README](../README.md) contains command examples and distinguishes local
inference from dataset-backed workflows.

## Evaluation and reproducibility utilities

- `Evaluation_Infilling/Matteo/` contains the reference aggregate, model aggregate,
  and JS-score pipeline. It needs the optional `eval` dependency group
  (`muspy`, `scipy`): `uv sync --group eval`.
- `audit_fixes/` contains integrity probes, manifests, recomputation, baseline, and
  paired-statistics tools created during research auditing. Preserve their outputs and
  use them only with the matching experiment artifacts.
- `Analyze_dataset/` contains dataset and checkpoint analyses. Some scripts are
  intended for the original cluster storage layout.

## Historical V1 and exploratory material

- `Pre_Processing/preprocess_500.py`, `preprocess_batch_improved.py`,
  `preprocess_single_track.py`, and related validation helpers record earlier pipeline
  stages. The V2 preprocessing script is the documented current pipeline.
- `Generation/`, `Tokenization/`, and `Masking/` include generation and inspection
  utilities from multiple iterations. Check the tokenizer and data assumptions before
  running them.
- `debug/`, `test/`, `Tests/`, and `Smoke_test_training/` are diagnostic scripts, not
  a conventional automated test suite. Several require a local preprocessed dataset.
- `figures/` produces thesis/research figures. Curated figures are tracked in
  `../figures/`; other generated output belongs in `../outputs/` (gitignored).
  `figures/generate_piano_roll_figure.py` and `run_extreme_metrics.py` read from
  hard-coded local paths — edit them before running.

### Tokenizer version per script

The model and released checkpoints use the **V2** tokenizer (vocabulary 550,
`beat_res={(0,4):8, (4,12):4}`). Scripts that build their own tokenizer:

| Version | Scripts |
| --- | --- |
| **V2** (compatible with the released checkpoints) | `Analyze_dataset/analyze_config_and_pitch.py`, `Analyze_dataset/analyze_preprocessed_v2_dataset.py`, `Analyze_dataset/thesis_mini_analyses.py`, `Evaluation_Accuracy/evaluate_phase1_accuracy_DIRECT.py`, `Pre_Processing/preprocess_v2.py`, `Tests/validate_v2_preprocessing.py`, `Tokenization/visualize_tokenization_v2.py` |
| **V1** (historical, `beat_res={(0,4):8}`; do not use with V2 checkpoints or data) | `Analyze_dataset/analyze_program_distribution.py`, `analyze_token_counts.py`, `analyze_token_frequency.py`, `analyze_velocity_distribution.py`; `Evaluation_Accuracy/evaluate_phase1_EXTENDED.py`; `Generation/generate_PITCH_ONLY_v3.py`, `generate_from_checkpoint_fixed_v3.py`; `Masking/test_masking.py`; `Pre_Processing/filter_preprocessed_by_program.py`, `preprocess_500.py`, `preprocess_batch_improved.py`, `preprocess_single_track.py`, `test_preprocess_one_file.py`, `validate_preprocessed_tokenization.py`, `validate_preprocessing.py`; `Tokenization/analyze_token_frequencies.py`, `verify_token_head_mapping.py`, `visualize_tokenization.py`; `debug/compute_class_weights.py` |

Scripts that import `midit.infer` (e.g. `infer_from_midi.py`,
`generate_samples_batch.py`) use the package's V2 tokenizer.

### Known-broken or misleading scripts

| Script | Problem |
| --- | --- |
| `test/test_class_weights.py` | Imports `midit.model.masked_transformer1`, which no longer exists. |
| `debug/debug_per_class_accuracy.py`, `Tokenization/verify_token_head_mapping.py` | Hard-code the invalid `phase1-epoch=44-val_loss=0.0982.ckpt`. |
| `../slurm/preprocess_full.sh` | Historical V1 preprocessing launcher (`Pre_Processing/preprocess_500.py`); output is not compatible with the V2 model. |
| `../slurm/generate_samples_zs.sh` | Generates "zero-shot" samples from the `epoch=49` checkpoint, not the thesis epoch-14 model. |

## Cluster-specific scripts

The `../slurm/` directory preserves DEI SLURM launchers. Many contain historical
absolute paths, module/virtual-environment assumptions, checkpoint names, and resource
requests. They are reproducibility records rather than portable commands. Do not alter
their paths or use them on another cluster without review.

`sota_baselines/anticipatory/` is likewise a separate baseline workflow with its own
model and environment assumptions. Consult its local README before use.

## Maintenance policy

Before changing a script, identify its tokenizer version, required checkpoint/data
lineage, and any referenced thesis result. Preserve V1, audit, baseline, and cluster
scripts unless their removal or relocation has explicit research approval.
