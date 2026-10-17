#!/bin/bash
#SBATCH --job-name=anticipatory-gen
#SBATCH --output=logs/anticipatory_gen_%j.out
#SBATCH --error=logs/anticipatory_gen_%j.err
#SBATCH --partition=allgroups
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:l40s:1

cd /home/zanettisim/MIDIT
source .venv_anticipation/bin/activate

# Redirect HuggingFace cache to /nfsd (home filesystem is shared and may be full)
export HF_HOME=/nfsd/tesi/zanettisim/huggingface_cache

python -u scripts/evaluation/baselines/anticipatory/generate_anticipatory_samples.py \
    --input_dir /nfsd/tesi/zanettisim/MIDIT/outputs/samples_1000 \
    --output_dir /nfsd/tesi/zanettisim/MIDIT/outputs/samples_anticipatory_1000 \
    --model_name stanford-crfm/music-medium-800k \
    --mask_bars 3 4 5 \
    --top_p 0.95
