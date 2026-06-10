#!/bin/bash
#SBATCH --job-name=midit_infill_v2a
#SBATCH --output=logs/slurm_infill_v2a_%j.out
#SBATCH --error=logs/slurm_infill_v2a_%j.err
#SBATCH --partition=allgroups
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:l40s:1

cd /home/zanettisim/MIDIT
source .venv/bin/activate
export PYTHONPATH=/home/zanettisim/MIDIT:$PYTHONPATH

# Fix CUDA memory fragmentation (OOM with reserved-but-unallocated)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Proposal A (Stability): accumulate=16, warmup=1000, grad_clip=0.5
# Uses conf/config_infilling_v2.yaml via Hydra --config-name override.
python midit/train_infilling.py \
    --config-name=config_infilling_v2 \
    logging.use_wandb=true \
    chunking.preprocessed_dir=/nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_50k
