#!/bin/bash
#SBATCH --job-name=midit-dataset-analysis
#SBATCH --output=logs/out/dataset_analysis_%j.txt
#SBATCH --error=logs/err/dataset_analysis_%j.txt
#SBATCH --partition=allgroups
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Chapter-3 preprocessed dataset analysis (read-only). Runs on a COMPUTE node:
# login.dei.unipd.it is a gateway and forbids running Python directly.
cd ~/MIDIT
source .venv/bin/activate

python scripts/Analyze_dataset/analyze_preprocessed_v2_dataset.py \
    --data_dir /nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_50k \
    --out_dir  /nfsd/tesi/zanettisim/MIDIT/outputs/dataset_analysis \
    --n 5000 --seed 42
