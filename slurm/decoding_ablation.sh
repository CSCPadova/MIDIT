#!/bin/bash
#SBATCH --job-name=midit-decoding-ablation
#SBATCH --output=logs/out/%j.txt
#SBATCH --error=logs/err/%j.txt
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00

cd ~/MIDIT
source .venv/bin/activate
export PYTHONPATH=/home/zanettisim/MIDIT:$PYTHONPATH

python scripts/evaluation/decoding/decoding_ablation.py \
    --checkpoint /nfsd/tesi/zanettisim/MIDIT/logs/checkpoints_infilling/infilling-epoch=05-val_loss=8.5836.ckpt \
    --data_dir   /nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_50k \
    --output_dir /nfsd/tesi/zanettisim/MIDIT/outputs/decoding_ablation \
    --n_seeds    50 \
    --mask_bars  3 4 5
