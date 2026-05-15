#!/bin/bash
# Generate 1000 infilling samples from the validation split.
#
# Runs midit/infer.py once per seed (1..1000).
# Seeds index into the val split (seed=42 split, unseen data confirmed).
# Skips seeds whose output already exists (safe to restart).
#
# Usage (from ~/MIDIT on cluster):
#   bash scripts/generate_1000_samples.sh

CHECKPOINT=/nfsd/tesi/zanettisim/MIDIT/logs/checkpoints_infilling/infilling-epoch=05-val_loss=8.5836.ckpt
DATA_DIR=/nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_50k
BASE_OUTPUT=/nfsd/tesi/zanettisim/MIDIT/outputs/samples_1000

N_TOTAL=1000
N_DONE=0
N_SKIPPED=0
N_FAILED=0

echo "[generate_1000_samples] Starting — seeds 1 to ${N_TOTAL}"
echo "  checkpoint : ${CHECKPOINT}"
echo "  data_dir   : ${DATA_DIR}"
echo "  output_base: ${BASE_OUTPUT}"
echo ""

for SEED in $(seq 1 ${N_TOTAL}); do
    OUTPUT_DIR="${BASE_OUTPUT}/seed_${SEED}"

    # Skip if already completed
    if [ -f "${OUTPUT_DIR}/reconstructed.mid" ]; then
        N_SKIPPED=$((N_SKIPPED + 1))
        continue
    fi

    python -u midit/infer.py \
        --checkpoint "${CHECKPOINT}" \
        --data_dir   "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --seed       "${SEED}" \
        --mask_bars  3 4 5

    if [ $? -eq 0 ]; then
        N_DONE=$((N_DONE + 1))
    else
        N_FAILED=$((N_FAILED + 1))
        echo "  [WARNING] seed ${SEED} failed — continuing"
    fi

    # Progress every 100 seeds
    PROCESSED=$((N_DONE + N_FAILED))
    if [ $((PROCESSED % 100)) -eq 0 ]; then
        echo ""
        echo "  --- Progress: ${PROCESSED}/${N_TOTAL} processed, ${N_SKIPPED} skipped, ${N_FAILED} failed ---"
        echo ""
    fi
done

echo ""
echo "[generate_1000_samples] Done."
echo "  Generated : ${N_DONE}"
echo "  Skipped   : ${N_SKIPPED} (already existed)"
echo "  Failed    : ${N_FAILED}"
