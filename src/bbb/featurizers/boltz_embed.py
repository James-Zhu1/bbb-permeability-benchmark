"""Load frozen Boltz-2 embeddings, row-aligned to a protocol's dataframe.

The Boltz-2 pipeline runs in a separate conda env (different RDKit), so it
never loads our dataset. It writes one pooled vector per UNIQUE molecule plus
a row -> mol_id index per protocol; this module maps those back onto
load_protocol()'s exact rows and verifies the index still matches.

Pipeline: scripts/boltz_export_inputs.py -> slurm/boltz_embed.sbatch ->
scripts/boltz_pool_embeddings.py -> data/boltz/full/s_mean.npz
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bbb.data import REPO_ROOT

FULL = REPO_ROOT / "data" / "boltz" / "full"


def load_vectors(df: pd.DataFrame, protocol: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (n_rows, 384) vectors (NaN where Boltz failed) and a missing mask.

    Rows with no embedding are left as NaN here rather than filled, because the
    fill must come from each split's TRAINING molecules only -- the caller does
    that per split.
    """
    index = pd.read_csv(FULL / f"index_{protocol}.csv")
    # Guard: the exporter's rows must be exactly load_protocol()'s rows.
    if len(index) != len(df) or not (index["smiles"].values == df["smiles"].values).all():
        raise ValueError(f"index_{protocol}.csv does not match the dataset; re-run the exporter")

    pooled = np.load(FULL / "s_mean.npz")
    row_of = {m: i for i, m in enumerate(pooled["mol_ids"])}
    X = np.full((len(df), pooled["X"].shape[1]), np.nan, dtype=np.float32)
    missing = np.zeros(len(df), dtype=bool)
    for r, mid in enumerate(index["mol_id"]):
        if mid in row_of:
            X[r] = pooled["X"][row_of[mid]]
        else:
            missing[r] = True
    return X, missing


def impute_from_train(X_all: np.ndarray, missing: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Fill missing rows with the mean vector of this split's training molecules."""
    X = X_all.copy()
    if missing.any():
        X[missing] = X_all[train_idx][~missing[train_idx]].mean(axis=0)
    return X
