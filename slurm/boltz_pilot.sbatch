#!/bin/bash
#SBATCH --job-name=boltz-pilot
#SBATCH --partition=main
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/%x-%j.out
# Step 4 pilot on 6 molecules (2..132 heavy atoms). Questions:
#   1. Does a ligand-only input (no protein) run?
#   2. s shape = [1, n_heavy_atoms, 384]?
#   3. Are embeddings identical with 10 vs 200 sampling steps? (they should be:
#      s/z come from the trunk, before diffusion) -> if so, use 10 for speed.
#   4. Seconds per molecule; do the cuEquivariance kernels work (CUDA 12 kernels
#      vs torch's CUDA 13.2 build)? If not, fall back to --no_kernels.
# Prepare inputs first (bbb env):  python scripts/boltz_export_inputs.py --limit 6
# Submit from the repo root:       sbatch slurm/boltz_pilot.sbatch
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate boltz
cd "$SLURM_SUBMIT_DIR"
P=data/boltz/pilot
echo "node: $(hostname) | env: $CONDA_DEFAULT_ENV | $(nvidia-smi --query-gpu=name --format=csv,noheader)"

run() {  # run <name> <extra boltz flags...>
  local name=$1; shift
  local t0=$(date +%s)
  boltz predict "$P/inputs" --out_dir "$P/$name" --write_embeddings --seed 0 \
        --override "$@" > "$P/$name.log" 2>&1
  local rc=$?
  echo "run $name: exit $rc, $(( $(date +%s) - t0 ))s  (flags: $*)"
  return $rc
}

KERNELS=""
run A_default || { echo "--- A failed with kernels; tail of log:"; tail -15 "$P/A_default.log";
                   KERNELS="--no_kernels"; run A_default $KERNELS || { tail -30 "$P/A_default.log"; exit 1; }; }
run B_fast $KERNELS --sampling_steps 10 --diffusion_samples 1 || { tail -30 "$P/B_fast.log"; exit 1; }

python - <<'EOF'
import glob, numpy as np, pandas as pd
from rdkit import Chem
mols = pd.read_csv("data/boltz/pilot/molecules.csv")
print(f"{'mol_id':<15} {'heavy':>5}  {'s shape':<16} {'z shape':<20} {'max |s_A - s_B|':>16}")
for mid, smi in zip(mols.mol_id, mols.boltz_smiles):
    f = {r: glob.glob(f"data/boltz/pilot/{r}/**/embeddings_{mid}.npz", recursive=True) for r in ("A_default", "B_fast")}
    if not f["A_default"] or not f["B_fast"]:
        print(f"{mid:<15} MISSING embeddings: {f}"); continue
    a, b = np.load(f["A_default"][0]), np.load(f["B_fast"][0])
    n = Chem.MolFromSmiles(smi).GetNumHeavyAtoms()
    print(f"{mid:<15} {n:>5}  {str(a['s'].shape):<16} {str(a['z'].shape):<20} {np.abs(a['s'] - b['s']).max():>16.2e}")
EOF
