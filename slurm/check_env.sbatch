#!/bin/bash
#SBATCH --job-name=bbb-check
#SBATCH --partition=main
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/%x-%j.out
# Setup check: GPU driver vs torch's CUDA build, then reproduce the Morgan+RF baseline.
# Submit from the repo root:   sbatch slurm/check_env.sbatch
set -euo pipefail

# Batch jobs don't read ~/.bashrc, so load conda's shell functions explicitly.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bbb

cd "$SLURM_SUBMIT_DIR"   # the directory sbatch was run from (the repo root)
echo "node: $(hostname) | env: $CONDA_DEFAULT_ENV"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c 'import torch; ok = torch.cuda.is_available(); print("torch CUDA build", torch.version.cuda, "| GPU usable:", ok, "|", torch.cuda.get_device_name(0) if ok else "-")'
python -W ignore scripts/run_fingerprint_rf.py | grep -v "Found local\|Loading\|Done!"
