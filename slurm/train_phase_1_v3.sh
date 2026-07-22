#!/bin/bash
#SBATCH --job-name=midit-phase1-v3
#SBATCH --output=logs/out/%j.txt
#SBATCH --error=logs/err/%j.txt
#SBATCH --time=7-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --partition=allgroups
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
cd ~/MIDIT
source .venv/bin/activate
export PYTHONPATH=/home/zanettisim/MIDIT:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u midit/train_phase1.py \
    train.epochs=50 \
    logging.use_wandb=true \
    chunking.preprocessed_dir=/nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_50k \
    "$@"