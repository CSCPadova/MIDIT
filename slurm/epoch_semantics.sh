#!/bin/bash
#SBATCH --job-name=midit-epoch-semantics
#SBATCH --output=logs/out/epoch_semantics_%j.txt
#SBATCH --error=logs/err/epoch_semantics_%j.txt
#SBATCH --partition=allgroups
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

# Read-only: exact epoch/coverage numbers. Runs on a COMPUTE node (login node
# forbids Python). Scans all training .pt files to count chunks with the exact
# training logic; reads checkpoint num_training_steps for cross-check.
cd ~/MIDIT
source .venv/bin/activate

python scripts/Analyze_dataset/epoch_semantics.py \
    --data_dir /nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_50k \
    --out_dir  /nfsd/tesi/zanettisim/MIDIT/outputs/thesis_analysis \
    --phase1_ckpt /nfsd/tesi/zanettisim/MIDIT/logs/checkpoints_phase1/phase1-epoch=14-val_loss=2.7254.ckpt \
    --infill_ckpt /nfsd/tesi/zanettisim/MIDIT/logs/checkpoints_infilling/infilling-epoch=05-val_loss=8.5836.ckpt
