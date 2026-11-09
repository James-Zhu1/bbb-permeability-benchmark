"""Step 4c (bbb env): pool Boltz-2's per-atom `s` embeddings into one vector per molecule.

    python scripts/boltz_pool_embeddings.py

Reads  data/boltz/full/out/boltz_results_inputs/predictions/<mol_id>/embeddings_<mol_id>.npz
Writes data/boltz/full/s_{mean,sum,max}.npz   {mol_ids: [n], X: [n, 384], missing: [k]}

Boltz's `s` has one 384-number row per heavy atom (shape [1, n_atoms, 384]).
Pooling turns that into one fixed-size vector per molecule:

  mean  average over atoms -- the default, matching frozen ChemBERTa.
  sum   add over atoms. A mean divides by atom count, so extensive ("how many
        N-H groups") properties survive only as fractions. On ChemBERTa, sum
        pooling lifted H-bond-donor recovery from R^2 0.768 to 0.896, so this
        tests whether mean pooling is what loses those properties here too.
  max   per-dimension maximum: "does any atom strongly activate this feature".

All three are computed in a single pass, because reading the ~700 MB of
per-atom files dominates the runtime. Molecules Boltz could not process (no
file) are listed in `missing` and imputed per split downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bbb.data import REPO_ROOT

FULL = REPO_ROOT / "data" / "boltz" / "full"
PRED = FULL / "out" / "boltz_results_inputs" / "predictions"

POOLERS = {
    "mean": lambda s: s.mean(axis=0),
    "sum": lambda s: s.sum(axis=0),
    "max": lambda s: s.max(axis=0),
}


def main() -> None:
    mols = pd.read_csv(FULL / "molecules.csv")
    ids: list[str] = []
    missing: list[str] = []
    vecs: dict[str, list[np.ndarray]] = {name: [] for name in POOLERS}
    n_atoms: list[int] = []

    for mid in mols["mol_id"]:
        f = PRED / mid / f"embeddings_{mid}.npz"
        if not f.exists():
            missing.append(mid)
            continue
        s = np.load(f)["s"][0]              # [n_atoms, 384]
        ids.append(mid)
        n_atoms.append(s.shape[0])
        for name, fn in POOLERS.items():
            vecs[name].append(fn(s))

    print(f"pooled {len(ids)} molecules ({min(n_atoms)}-{max(n_atoms)} atoms each); "
          f"missing {len(missing)}: {missing}")
    for name in POOLERS:
        X = np.stack(vecs[name]).astype(np.float32)
        out = FULL / f"s_{name}.npz"
        np.savez(out, mol_ids=np.array(ids), X=X, missing=np.array(missing))
        print(f"  {out.name:<12} X {X.shape}  mean {X.mean():+.3f}  std {X.std():.3f}  "
              f"min {X.min():+.1f}  max {X.max():+.1f}")


if __name__ == "__main__":
    main()
