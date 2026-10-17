# Script guide

Scripts that reproduce the MIDIT pipeline and the thesis results. Historical,
exploratory and V1-tokenizer scripts are in `research/` at the `archive-full-tree`
tag; the pre-reorganisation layout is preserved in the `thesis-final` tag.

Run everything from the repository root with `uv run python ...`. Scripts that need
a dataset, checkpoint or cluster storage say so in their `--help`. The evaluation
scripts need the optional dependency group: `uv sync --group eval`. All scripts that
build a tokenizer use the **V2** configuration (vocabulary 550) of the released
checkpoints.

| Folder | Script | Purpose | Backs (thesis) |
|---|---|---|---|
| `preprocessing/` | `preprocess_v2.py` | Lakh MIDI → filtered, pitch-shifted, REMI+ `.pt` dataset | Fig. 3.4 pipeline; dataset counts |
| | `visualize_tokenization_v2.py` | Inspect the token stream of a preprocessed `.pt` file | — |
| `analysis/` | `analyze_lakh_dataset.py` | Raw Lakh statistics (500-file sample) | dataset description (`docs/analysis/dataset_analysis.txt`) |
| | `analyze_quality_filter.py` | Quality-filter simulation (500-file sample) | filter design (`docs/analysis/quality_filter_simulation.txt`) |
| | `analyze_config_and_pitch.py` | V2 tokenizer vocabulary impact and pitch-shift validation | V2 config, ±5 augmentation (`docs/analysis/config_v2_analysis.txt`, `pitch_shift_analysis.txt`) |
| | `analyze_preprocessed_v2_dataset.py` | Statistics of the preprocessed V2 dataset | Ch. 3 dataset analysis |
| | `thesis_mini_analyses.py` | Per-class frequencies, loss-weight audit, sequence lengths | Ch. 3–4 analyses |
| `inference/` | `infer_from_midi.py` | Infill bars of any MIDI file (also importable: `infill_midi()`) | README quickstart |
| | `inspect_infill.py` | Print original vs MIDIT notes per onset in the masked bars of an `infer_from_midi.py` output (writes no files) | README "Inspecting a reconstruction" |
| | `generate_samples_batch.py` | Generate N validation samples with one model load | evaluation sample sets |
| `evaluation/` | `evaluate_phase1_accuracy_DIRECT.py` | Cloze Top-1/Top-5 accuracy per token type | Fig. 5.1, cloze results |
| | `compute_infilling_metrics.py` | CP / GS / PCHE implementation (library + CLI) | Ch. 5 infilling metrics |
| | `compute_muspy_metrics.py` | Per-sample CP/GS/PCHE + S-I metrics for a results folder (library + CLI) | Figs. 5.4–5.5, Table 5.11 |
| | `merge_metrics.py` | Merge per-seed CSVs into summary/LaTeX tables | Ch. 5 tables |
| | `run_extreme_metrics.py` | Metrics for the first/last-bar extreme-infilling runs | Table 5.11, Fig. 5.5 |
| | `paired_stats_fixed.py` | Paired t-test / Wilcoxon / d_z, MIDIT vs Anticipatory MT | Table 5.17 |
| `evaluation/js/` | `compute_reference_agg.py`, `compute_model_agg.py`, `compute_js_score.py`, `js_similarity.py`, `muspy_metrics.py` | JS Similarity pipeline (author: Matteo Spanio, see [`README`](evaluation/js/README.md)) | Table 5.15 |
| | `run_js_upperbound.sh`, `run_js_zs.sh` | Cluster-only wrappers (upper bound, zero-shot) | Table 5.15 |
| `evaluation/decoding/` | `decoding_ablation.py`, `decoding_ablation_postprocess.py` | Argmax vs sampling vs MaskGIT-style decoding, + S-I metrics | §5.11 decoding |
| `evaluation/baselines/` | `recompute_anticipatory_metrics_fixed.py` | Anticipatory MT metrics with time-aligned crop (tpq fix) | Tables 5.17, 5.18 |
| | `baseline_random_in_key.py` | Random-in-key pitch baselines B1/B2 (CP floor) | baseline comparison |
| | `anticipatory/` | Anticipatory MT sample generation (separate venv, see its README) | Tables 5.15, 5.17 |
| `verification/` | `validate_v2_preprocessing.py` | Validate the V2 preprocessed dataset | `docs/analysis/v2_preprocessing_validation.txt` |
| | `checkpoint_config_audit.py` | Dump checkpoint hyperparameters vs config | Table 4.1 correction (`docs/ERRATA.md`) |
| | `check_sample_integrity.py` | Integrity of generated sample folders | evaluation sample sets |
| | `probe_no_cheat.py` | Runtime probe: inference cannot copy masked tokens | "genuinely generative" check |
| | `token_diff_masked_bars.py` | Token-level diff of masked bars vs ground truth | "genuinely generative" check |
| | `build_sample_manifest.py` | Builds `sample_manifest.csv` (matched sample set) | Table 5.17 sample set |
| `figures/` | `generate_architecture.py`, `preprocessing_pipeline_v2.py`, `generate_piano_roll_figure.py` | Figure scripts (default output `docs/img/`, not versioned). The output of `generate_architecture.py` is no longer used in the README, which shows the architecture as a Mermaid diagram. The thesis figures are in the thesis PDF (corrections in [`../docs/ERRATA.md`](../docs/ERRATA.md)); the original PNGs are at the `thesis-final` tag, e.g. `git show thesis-final:figures/val_loss_curve.png > val_loss_curve.png` | Fig. 3.4 |

## Notes

- `evaluation/compute_infilling_metrics.py` and `evaluation/compute_muspy_metrics.py`
  are imported through `sys.path` by `evaluation/js/compute_model_agg.py`, the
  decoding scripts, `evaluation/baselines/recompute_anticipatory_metrics_fixed.py`,
  `verification/build_sample_manifest.py` and `verification/token_diff_masked_bars.py`.
  Move them only together with these importers.
- `verification/sample_manifest.csv` (written by `build_sample_manifest.py`, read by
  default by `recompute_anticipatory_metrics_fixed.py` and `token_diff_masked_bars.py`)
  is **not versioned**; on the cluster it was written to the former
  `scripts/audit_fixes/` folder.
- `../slurm/generate_samples_zs.sh` generated the "zero-shot" samples from the
  `epoch=49` checkpoint, not the thesis epoch-14 model.
- The `../slurm/` launchers contain DEI cluster paths and are kept as run records;
  only their script paths are updated.
