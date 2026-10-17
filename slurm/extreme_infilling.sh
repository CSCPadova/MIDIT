#!/bin/bash
#SBATCH --job-name=midit-extreme-infer
#SBATCH --output=logs/out/extreme_%j.txt
#SBATCH --error=logs/err/extreme_%j.txt
#SBATCH --mail-type=ALL
#SBATCH --time=2-20:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --partition=allgroups
#SBATCH --mem=10G
#SBATCH --gres=gpu:rtx

cd /home/zanettisim/MIDIT
source .venv/bin/activate
export PYTHONPATH=/home/zanettisim/MIDIT:$PYTHONPATH

CHECKPOINT=/nfsd/tesi/zanettisim/MIDIT/logs/checkpoints_infilling/infilling-epoch=05-val_loss=8.5836.ckpt
DATA_DIR=/nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_50k
OUTPUT_BASE=/nfsd/tesi/zanettisim/MIDIT/outputs/extreme

SEEDS=(1 2 3 4 5 10)

# Config name → space-separated bar indices (0-indexed, 8-bar chunk)
declare -A MASK_BARS
MASK_BARS=(
    [first1]="0"
    [first2]="0 1"
    [first3]="0 1 2"
    [last1]="7"
    [last2]="6 7"
    [last3]="5 6 7"
)
CONFIGS=(first1 first2 first3 last1 last2 last3)

TOTAL=$(( ${#SEEDS[@]} * ${#CONFIGS[@]} ))
CURRENT=0

echo "========================================"
echo " MIDIT Extreme Infilling Experiments"
echo " Checkpoint: $CHECKPOINT"
echo " Seeds:      ${SEEDS[*]}"
echo " Configs:    ${CONFIGS[*]}"
echo " Total runs: $TOTAL"
echo "========================================"
echo ""

for SEED in "${SEEDS[@]}"; do
    for CONFIG in "${CONFIGS[@]}"; do
        CURRENT=$(( CURRENT + 1 ))
        BARS="${MASK_BARS[$CONFIG]}"
        OUT_DIR="${OUTPUT_BASE}/${CONFIG}/seed_${SEED}"

        echo "[${CURRENT}/${TOTAL}] seed=${SEED}  config=${CONFIG}  mask_bars=[${BARS}]"
        echo "         -> ${OUT_DIR}"

        python midit/infer.py \
            --checkpoint "$CHECKPOINT" \
            --data_dir   "$DATA_DIR"   \
            --output_dir "$OUT_DIR"    \
            --seed       "$SEED"       \
            --mask_bars  $BARS

        echo "  Done: seed=${SEED} config=${CONFIG}"
        echo ""
    done
done

echo "========================================"
echo " All ${TOTAL} experiments complete."
echo " Results in: ${OUTPUT_BASE}"
echo "========================================"
