#!/bin/bash
#SBATCH --job-name=preprocess-lakh
#SBATCH --partition=allgroups
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/out/preprocess_%j.txt
#SBATCH --error=logs/err/preprocess_%j.txt

cd $HOME/MIDIT

~/.local/bin/uv run python scripts/preprocess_500.py \
    --input_dir data/lakh_clean \
    --output_dir data/preprocessed_phase1 \
    --num_workers 16

echo "✅ Preprocessing complete!"
