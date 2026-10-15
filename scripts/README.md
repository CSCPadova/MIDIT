# Script guide

This directory holds the scripts needed to reproduce the MIDIT pipeline and the
thesis results: preprocessing, inference, evaluation, verification probes and
figures. Historical, exploratory and V1-tokenizer scripts were moved to
[`../research/`](../research/README.md); the original layout is preserved in the
`thesis-final` tag.

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

- `Evaluation_Infilling/compute_infilling_metrics.py` (CP / GS / PCHE) and
  `compute_muspy_metrics.py` are also imported by other scripts through `sys.path`;
  keep them in place.
- `Evaluation_Infilling/Matteo/` contains the reference aggregate, model aggregate,
  and JS-score pipeline (`run_js_upperbound.sh`, `run_js_zs.sh` drive it on the
  cluster). It needs the optional `eval` dependency group (`muspy`, `scipy`):
  `uv sync --group eval`.
- `audit_fixes/` contains integrity probes, manifests, recomputation, baseline, and
  paired-statistics tools created during research auditing. Preserve their outputs and
  use them only with the matching experiment artifacts.
- `decoding_ablation.py` / `decoding_ablation_postprocess.py`: decoding-strategy
  comparison (argmax, sampling, MaskGIT-style). `run_extreme_metrics.py`: metrics for
  the extreme-infilling configurations.
- `Analyze_dataset/` contains dataset and checkpoint analyses used for the thesis.
  Some scripts are intended for the original cluster storage layout.
- `Tests/validate_v2_preprocessing.py` validates the V2 preprocessing output.
  The automated smoke tests live in `../tests/` (`uv run pytest`).
- `figures/` produces thesis figures (see [`../figures/README.md`](../figures/README.md)).
  `figures/generate_piano_roll_figure.py` (`--input_dir`) and `run_extreme_metrics.py`
  (`--base_dir`, `--out_dir`) take their input/output locations as arguments.

### Tokenizer version per script

The model and released checkpoints use the **V2** tokenizer (vocabulary 550,
`beat_res={(0,4):8, (4,12):4}`). Scripts here that build their own tokenizer are all
V2: `Analyze_dataset/analyze_config_and_pitch.py`,
`Analyze_dataset/analyze_preprocessed_v2_dataset.py`,
`Analyze_dataset/thesis_mini_analyses.py`,
`Evaluation_Accuracy/evaluate_phase1_accuracy_DIRECT.py`,
`Pre_Processing/preprocess_v2.py`, `Tests/validate_v2_preprocessing.py`,
`Tokenization/visualize_tokenization_v2.py`. Scripts that import `midit.infer`
(e.g. `infer_from_midi.py`, `generate_samples_batch.py`) use the package's V2
tokenizer. The V1 scripts are listed in [`../research/README.md`](../research/README.md).

### Known caveats

| Script | Caveat |
| --- | --- |
| `../slurm/generate_samples_zs.sh` | Generates "zero-shot" samples from the `epoch=49` checkpoint, not the thesis epoch-14 model. |
| `audit_fixes/recompute_anticipatory_metrics_fixed.py`, `audit_fixes/token_diff_masked_bars.py` | Default `--manifest scripts/audit_fixes/sample_manifest.csv`, which is written by `build_sample_manifest.py` and is not versioned. |

## Cluster-specific scripts

The `../slurm/` directory preserves DEI SLURM launchers. Many contain historical
absolute paths, module/virtual-environment assumptions, checkpoint names, and resource
requests. They are reproducibility records rather than portable commands. Do not alter
their paths or use them on another cluster without review.

`sota_baselines/anticipatory/` is likewise a separate baseline workflow with its own
model and environment assumptions. Consult its local README before use.

## Maintenance policy

Before changing a script, identify its tokenizer version, required checkpoint/data
lineage, and any referenced thesis result. Preserve audit, baseline, and cluster
scripts unless their removal or relocation has explicit research approval.
