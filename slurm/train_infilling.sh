#!/bin/bash
#SBATCH --job-name=midit_infilling
#SBATCH --output=logs/slurm_infilling_%j.out
#SBATCH --error=logs/slurm_infilling_%j.err
#SBATCH --partition=allgroups
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1

cd /home/zanettisim/MIDIT
source .venv/bin/activate
export PYTHONPATH=/home/zanettisim/MIDIT:$PYTHONPATH

# Fix CUDA memory fragmentation (OOM with 9.74 GB reserved-but-unallocated)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python midit/train_infilling.py \
    logging.use_wandb=true \
    chunking.preprocessed_dir=/nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_50k
