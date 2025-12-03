#!/bin/bash

#SBATCH --job-name=midit_comparison
#SBATCH --output=logs/comparison/slurm_%j.out
#SBATCH --error=logs/comparison/slurm_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=allgroups
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1

# Load modules (se necessario)
# module load cuda/...

# Activate environment
cd ~/MIDIT

# Create logs directory
mkdir -p logs/comparison

# Run training
uv run python scripts/train_comparison.py

echo "✅ Training completed!"
