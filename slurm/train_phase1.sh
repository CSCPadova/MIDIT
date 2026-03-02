#!/bin/bash
#SBATCH --job-name=midit-phase1
#SBATCH --output=logs/out/%j.txt
#SBATCH --error=logs/err/%j.txt
#SBATCH --time=2-20:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --partition=allgroups
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1

cd ~/MIDIT
source .venv/bin/activate
export PYTHONPATH=/home/zanettisim/MIDIT:$PYTHONPATH
python -u midit/train_phase1.py
