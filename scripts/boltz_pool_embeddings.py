"""Step 4c (bbb env): pool Boltz-2's per-atom `s` embeddings into one vector per molecule.

    python scripts/boltz_pool_embeddings.py

Reads  data/boltz/full/out/boltz_results_inputs/predictions/<mol_id>/embeddings_<mol_id>.npz
Writes data/boltz/full/s_mean.npz   {mol_ids: [n], X: [n, 384], missing: [k]}

Boltz's `s` has one 384-number row per heavy atom (shape [1, n_atoms, 384]).
Averaging the rows gives one fixed-size vector per molecule -- the same
"mean pooling" used for frozen ChemBERTa. Molecules Boltz could not process
(no file) are listed in `missing` and imputed later, per split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bbb.data import REPO_ROOT

FULL = REPO_ROOT / "data" / "boltz" / "full"
PRED = FULL / "out" / "boltz_results_inputs" / "predictions"


def main() -> None:
    mols = pd.read_csv(FULL / "molecules.csv")
    ids, vecs, missing = [], [], []
    for mid in mols["mol_id"]:
        f = PRED / mid / f"embeddings_{mid}.npz"
        if not f.exists():
            missing.append(mid)
            continue
        s = np.load(f)["s"][0]              # [n_atoms, 384]
        ids.append(mid)
        vecs.append(s.mean(axis=0))         # average over atoms -> [384]

    X = np.stack(vecs).astype(np.float32)
    np.savez(FULL / "s_mean.npz", mol_ids=np.array(ids), X=X, missing=np.array(missing))
    print(f"pooled {len(ids)} molecules -> X {X.shape}; missing {len(missing)}: {missing}")
    print(f"value scale: mean {X.mean():+.3f}, std {X.std():.3f}, "
          f"min {X.min():+.2f}, max {X.max():+.2f}")


if __name__ == "__main__":
    main()
