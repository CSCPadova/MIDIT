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

bash scripts/generate_1000_samples.sh
