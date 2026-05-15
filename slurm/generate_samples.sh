#!/bin/bash
#SBATCH --job-name=midit-gen1k
#SBATCH --output=logs/out/%j.txt
#SBATCH --error=logs/err/%j.txt
#SBATCH --time=3-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=allgroups
#SBATCH --mem=10G
#SBATCH --gres=gpu:rtx:1

cd ~/MIDIT
source .venv/bin/activate
export PYTHONPATH=/home/zanettisim/MIDIT:$PYTHONPATH

python -u scripts/generate_samples_batch.py \
    --checkpoint /nfsd/tesi/zanettisim/MIDIT/logs/checkpoints_infilling/infilling-epoch=05-val_loss=8.5836.ckpt \
    --data_dir   /nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_50k \
    --output_dir /nfsd/tesi/zanettisim/MIDIT/outputs/samples_1000 \
    --n_seeds    1000 \
    --start_seed 1 \
    --mask_bars  3 4 5
