#!/bin/bash
#SBATCH --job-name=midit-gen-zs
#SBATCH --output=logs/out/%j.txt
#SBATCH --error=logs/err/%j.txt
#SBATCH --time=3-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=allgroups
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1

cd ~/MIDIT
source .venv/bin/activate
export PYTHONPATH=/home/zanettisim/MIDIT:$PYTHONPATH

python -u scripts/inference/generate_samples_batch.py \
    --checkpoint /nfsd/tesi/zanettisim/MIDIT/logs/checkpoints_phase1/phase1-epoch=49-val_loss=2.3866.ckpt \
    --data_dir   /nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_50k \
    --output_dir /nfsd/tesi/zanettisim/MIDIT/outputs/samples_zs_1000 \
    --n_seeds    989 \
    --start_seed 1 \
    --mask_bars  3 4 5
