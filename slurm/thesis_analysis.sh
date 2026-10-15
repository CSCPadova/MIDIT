#!/bin/bash
#SBATCH --job-name=midit-thesis-analysis
#SBATCH --output=logs/out/thesis_analysis_%j.txt
#SBATCH --error=logs/err/thesis_analysis_%j.txt
#SBATCH --partition=allgroups
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Three read-only thesis analyses. Runs on a COMPUTE node (login node forbids
# running Python). Reads the Phase-1 checkpoint hparams for the authoritative
# lambda values.
cd ~/MIDIT
source .venv/bin/activate

python scripts/analysis/thesis_mini_analyses.py \
    --data_dir /nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_50k \
    --out_dir  /nfsd/tesi/zanettisim/MIDIT/outputs/thesis_analysis \
    --n 100 --seed 42 \
    --phase1_ckpt /nfsd/tesi/zanettisim/MIDIT/logs/checkpoints_phase1/phase1-epoch=14-val_loss=2.7254.ckpt \
    --infill_ckpt /nfsd/tesi/zanettisim/MIDIT/logs/checkpoints_infilling/infilling-epoch=05-val_loss=8.5836.ckpt
