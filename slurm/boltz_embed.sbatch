#!/bin/bash
#SBATCH --job-name=boltz-embed
#SBATCH --partition=main
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/%x-%j.out
# Step 4: frozen Boltz-2 embeddings for all 1966 unique molecules.
#
# Inputs (bbb env, first):   python scripts/boltz_export_inputs.py
# Submit from the repo root: sbatch slurm/boltz_embed.sbatch
# Output: data/boltz/full/out/boltz_results_inputs/predictions/<mol_id>/embeddings_<mol_id>.npz
#
# Fast setting validated by the pilot (job 404413): s/z come from the trunk,
# BEFORE diffusion, and were bit-identical with 10 vs 200 sampling steps.
# Expected failure: 1 molecule (m9d430bdbfc42, 229 atoms with H) exceeds
# Boltz's 4-character atom-name limit; it is skipped here and imputed later.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate boltz
cd "$SLURM_SUBMIT_DIR"
F=data/boltz/full
echo "node: $(hostname) | env: $CONDA_DEFAULT_ENV | $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "inputs: $(ls $F/inputs | wc -l)"

t0=$(date +%s)
boltz predict "$F/inputs" --out_dir "$F/out" --write_embeddings --seed 0 \
      --sampling_steps 10 --diffusion_samples 1 --override > "$F/boltz.log" 2>&1
rc=$?
echo "boltz exit $rc after $(( $(date +%s) - t0 ))s"
grep -E "Number of failed examples|Failed to process" "$F/boltz.log" | cut -c1-140
echo "embedding files: $(find $F/out -name 'embeddings_*.npz' | wc -l)"
du -sh "$F/out"
