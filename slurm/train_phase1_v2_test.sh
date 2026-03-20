#!/bin/bash
#SBATCH --job-name=midit_v2_test
#SBATCH --output=logs/slurm_v2_test_%j.out
#SBATCH --error=logs/slurm_v2_test_%j.err
#SBATCH --partition=allgroups
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1

cd /home/zanettisim/MIDIT
source .venv/bin/activate

python midit/train_phase1.py \
    train.epochs=3 \
    train.batch_size=2
