#!/bin/bash
# Template launcher, not used for thesis runs (see slurm/train_phase_1_v3.sh).
#SBATCH --job-name midit-train
#SBATCH --output logs/out/%j.txt
#SBATCH --error logs/err/%j.txt
#SBATCH --mail-type ALL
#SBATCH --time 2-20:00:00
#SBATCH --ntasks-per-node 2
#SBATCH --partition allgroups
#SBATCH --mem 32G
#SBATCH --gres=gpu:rtx

source .env

srun --mail-user "$EMAIL" ~/.local/bin/uv run train
