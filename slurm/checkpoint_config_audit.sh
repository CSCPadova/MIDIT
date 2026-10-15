#!/bin/bash
#SBATCH --job-name=midit-ckpt-audit
#SBATCH --output=logs/out/ckpt_audit_%j.txt
#SBATCH --error=logs/err/ckpt_audit_%j.txt
#SBATCH --partition=allgroups
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Read-only: dump checkpoint hparams + search logs for Trainer params.
# Runs on a COMPUTE node (login node forbids Python).
cd ~/MIDIT
source .venv/bin/activate

python scripts/verification/checkpoint_config_audit.py \
    --phase1_ckpt /nfsd/tesi/zanettisim/MIDIT/logs/checkpoints_phase1/phase1-epoch=14-val_loss=2.7254.ckpt \
    --infill_ckpt /nfsd/tesi/zanettisim/MIDIT/logs/checkpoints_infilling/infilling-epoch=05-val_loss=8.5836.ckpt \
    --out_dir  /nfsd/tesi/zanettisim/MIDIT/outputs/thesis_analysis
