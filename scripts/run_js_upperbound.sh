#!/bin/bash
# scripts/run_js_upperbound.sh
#
# Upper-bound JS Similarity for the infilling pipeline.
#
# Symlinks original.mid -> reconstructed.mid for every seed, then runs the
# standard model_agg + js_score pipeline. The result is the JS Similarity
# score the model would obtain if it perfectly reconstructed the masked bars,
# providing a ceiling for any real model score.

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
SAMPLES=/nfsd/tesi/zanettisim/MIDIT/outputs/samples_1000
UPPER=/nfsd/tesi/zanettisim/MIDIT/outputs/upperbound
OUTPUTS=/nfsd/tesi/zanettisim/MIDIT/outputs

REF_AGG="$OUTPUTS/ref_agg.json"
MODEL_AGG="$OUTPUTS/model_agg_upperbound.json"
JS_OUT="$OUTPUTS/js_score_upperbound.json"

# ── Sanity ───────────────────────────────────────────────────────────────────
if [ ! -d "$SAMPLES" ]; then
    echo "Error: SAMPLES dir not found: $SAMPLES" >&2
    exit 1
fi
if [ ! -f "$REF_AGG" ]; then
    echo "Error: reference aggregate not found: $REF_AGG" >&2
    echo "       Run compute_reference_agg.py first." >&2
    exit 1
fi

# ── Activate venv ────────────────────────────────────────────────────────────
source .venv/bin/activate

# ── 1. Build symlink tree ────────────────────────────────────────────────────
echo "[1/3] Building upper-bound symlink tree at $UPPER"
mkdir -p "$UPPER"

n_seeds=0
for seed_dir in "$SAMPLES"/seed_*; do
    [ -d "$seed_dir" ] || continue
    seed_name=$(basename "$seed_dir")
    dst="$UPPER/$seed_name"
    mkdir -p "$dst"

    # original.mid and masked.mid are needed by _detect_masked_bars().
    # reconstructed.mid -> original.mid is the upper-bound (perfect copy).
    ln -sf "$seed_dir/original.mid" "$dst/original.mid"
    ln -sf "$seed_dir/masked.mid"   "$dst/masked.mid"
    ln -sf "$seed_dir/original.mid" "$dst/reconstructed.mid"
    n_seeds=$((n_seeds + 1))
done
echo "      Linked $n_seeds seed directories."
echo

# ── 2. Compute model aggregate over upper-bound dir ──────────────────────────
echo "[2/3] Running compute_model_agg.py"
python -u scripts/Evaluation_Infilling/Matteo/compute_model_agg.py \
    --samples_dir "$UPPER" \
    --output_json "$MODEL_AGG" \
    --max_files   1000
echo

# ── 3. Compute JS similarity score ───────────────────────────────────────────
echo "[3/3] Running compute_js_score.py"
python -u scripts/Evaluation_Infilling/Matteo/compute_js_score.py \
    --model_agg   "$MODEL_AGG" \
    --ref_agg     "$REF_AGG" \
    --model_name  "Upper_Bound" \
    --output_json "$JS_OUT"
echo

# ── Final score ──────────────────────────────────────────────────────────────
js=$(python -c "import json,sys; print(json.load(open('$JS_OUT'))['js_similarity'])")
echo "════════════════════════════════════════════════════════════════"
echo "  Upper-Bound JS Similarity = $js"
echo "════════════════════════════════════════════════════════════════"
