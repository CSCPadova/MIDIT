# Research archive

**Historical, not maintained.** These scripts and files are kept for provenance:
they document earlier pipeline stages, exploratory analyses and debugging work
behind the thesis. Many use the **V1 tokenizer** (`beat_res={(0,4):8}`) and are not
compatible with the released V2 checkpoints or the V2 preprocessed data; some
hard-code checkpoints or cluster paths that are not available. They are not tested.

The original layout (everything under `scripts/` and `tests/`) is preserved in the
`thesis-final` tag. Sub-folder names mirror the former `scripts/<folder>/`, so
relative paths inside the scripts resolve to the same depth as before.

| Folder | Contents |
| --- | --- |
| `Analyze_dataset/` | V1 token/velocity/program analyses (outputs in `../docs/analysis/`), raw-MIDI pitch/token exploration, MidiCaps exploration, dataloader debugging, `epoch_semantics.py` (its config-based conclusions are superseded) |
| `Evaluation_Accuracy/` | `evaluate_phase1_EXTENDED.py` — V1 cloze evaluation, superseded by `scripts/Evaluation_Accuracy/evaluate_phase1_accuracy_DIRECT.py` |
| `Generation/` | V1 generation scripts |
| `Masking/` | V1 masking test |
| `Pre_Processing/` | V1 preprocessing and validation (`preprocess_500.py` is still launched by `../slurm/preprocess_full.sh`) |
| `Smoke_test_training/` | Early forward/loss smoke test |
| `Tokenization/` | V1 tokenization inspection; `verify_token_head_mapping.py` hard-codes the invalid `epoch=44` checkpoint |
| `debug/` | Class-weight and mode-collapse investigations; `debug_per_class_accuracy.py` hard-codes the invalid `epoch=44` checkpoint |
| `test/` | `test_class_weights.py` — **broken**: imports the removed module `midit.model.masked_transformer1` |
| `tests/` | `debug_dataset.py` — downloads a MidiCaps sample (Phase 2, deferred) |
| `figures/` | `preprocessing_pipeline.py` (first version of the pipeline figure, superseded by `scripts/figures/preprocessing_pipeline_v2.py`); `extreme_infilling_sc.png` (its source data is not verified) |
| `thesis_materials/` | Earlier approaches (diffusion models, MidiCaps loaders), early analysis scripts and results, round-trip validation MIDI files (moved from the repository root) |
| `generate_1000_samples.sh` | Per-seed sample generation, superseded by `scripts/inference/generate_samples_batch.py` |
| `test_config.py` | Config-loading check, superseded by `tests/test_smoke.py` |
