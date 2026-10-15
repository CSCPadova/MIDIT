#!/bin/bash
# Historical V1 preprocessing (beat_res={(0,4):8}); not compatible with the V2
# (vocab 550) model. The current pipeline is scripts/preprocessing/preprocess_v2.py.
#SBATCH --job-name=preprocess-lakh
#SBATCH --partition=allgroups
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/out/preprocess_%j.txt
#SBATCH --error=logs/err/preprocess_%j.txt

cd $HOME/MIDIT

~/.local/bin/uv run python research/Pre_Processing/preprocess_500.py \
    --input_dir data/lakh_clean \
    --output_dir data/preprocessed_phase1 \
    --num_workers 16

echo "✅ Preprocessing complete!"
