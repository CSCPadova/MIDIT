# JS Similarity pipeline

**Author: Matteo Spanio.** The JS-similarity metric (`js_similarity.py`) and the
MusPy metric set (`muspy_metrics.py`) are his evaluation utilities, ported from the
mxlGPT evaluation code (see the module docstrings). The driver scripts adapt them
to MIDIT: metrics are computed on the masked bars only (`compute_model_agg.py`) and
`groove_consistency` uses `measure_resolution=4` (see `git log` for the changes).

Previously located in `scripts/Evaluation_Infilling/Matteo/`.

## Steps

Requires the `eval` dependency group (`uv sync --group eval`). Run from the
repository root.

```bash
# 1. reference aggregate over LMD files
uv run python scripts/evaluation/js/compute_reference_agg.py \
    --midi_dir data/midi_files/lmd_full/ --output_json outputs/ref_agg.json \
    --max_files 1000 --seed 42

# 2. model aggregate over the masked bars of generated samples (seed_N/ folders)
uv run python scripts/evaluation/js/compute_model_agg.py \
    --samples_dir outputs/samples_1000 --output_json outputs/model_agg.json

# 3. JS Similarity score
uv run python scripts/evaluation/js/compute_js_score.py \
    --ref_agg outputs/ref_agg.json --model_agg outputs/model_agg.json
```

| File | Role |
|---|---|
| `js_similarity.py` | JS-divergence similarity between a model aggregate and the reference aggregate |
| `muspy_metrics.py` | MusPy metrics with a `muspy_` prefix |
| `compute_reference_agg.py` | Step 1: reference aggregate |
| `compute_model_agg.py` | Step 2: model aggregate (masked bars only); imports `../compute_muspy_metrics.py` |
| `compute_js_score.py` | Step 3: final score |
| `run_js_upperbound.sh` | **Cluster-only.** Upper bound: scores the ground-truth bars as if they were the reconstruction (thesis Table 5.15) |
| `run_js_zs.sh` | **Cluster-only.** Zero-shot score; uses samples from `slurm/generate_samples_zs.sh`, which used the `epoch=49` checkpoint |
