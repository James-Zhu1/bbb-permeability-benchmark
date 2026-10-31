"""Bemis-Murcko scaffold splits: generate, save, load.

The split files in `splits/` are the contract between all models: each holds
integer row indices into `bbb.data.load_dataset()`. Models only ever call
`load_split()`; they never re-split.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from bbb.data import REPO_ROOT, dataset_fingerprint

SPLITS_DIR = REPO_ROOT / "splits"
N_SPLITS = 5
FRACTIONS = (0.8, 0.1, 0.1)  # train, valid, test


# --------------------------------------------------------------------------
# 1. Scaffolds
# --------------------------------------------------------------------------
def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold (ring systems + linkers) as canonical SMILES.

    Example: propranolol  CC(C)NCC(O)COc1cccc2ccccc12  ->  c1ccc2ccccc2c1
    (the side chain is removed; the naphthalene ring system remains).

    Molecules with no rings have an empty scaffold "" and all end up in one
    group together.
    """
    mol = Chem.MolFromSmiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_groups(smiles: list[str]) -> dict[str, list[int]]:
    """Map scaffold -> list of row indices of the molecules that have it."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[murcko_scaffold(s)].append(i)
    return dict(groups)


# --------------------------------------------------------------------------
# 2. The split
# --------------------------------------------------------------------------
def scaffold_split(
    smiles: list[str], seed: int, fractions: tuple[float, float, float] = FRACTIONS
) -> dict[str, list[int]]:
    """Balanced, randomized scaffold split (chemprop-style).

    Every scaffold group is assigned WHOLE to exactly one of train/valid/test,
    so no scaffold in test was seen during training.

    Algorithm:
      - "Big" groups (larger than half the valid or test target size) can't
        fit in a small set without overshooting it badly, so they go first and
        in practice always land in train.
      - All remaining groups are shuffled with `seed`, so each seed holds out a
        different region of scaffold space.
      - Walk through the groups in that order and greedily fill train up to
        80%, then valid up to 10%; anything that fits in neither goes to test.
    """
    n = len(smiles)
    train_size = fractions[0] * n
    valid_size = fractions[1] * n
    test_size = fractions[2] * n

    rng = np.random.default_rng(seed)
    groups = list(scaffold_groups(smiles).values())

    big, small = [], []
    for g in groups:
        (big if len(g) > valid_size / 2 or len(g) > test_size / 2 else small).append(g)
    rng.shuffle(big)
    rng.shuffle(small)

    train: list[int] = []
    valid: list[int] = []
    test: list[int] = []
    for g in big + small:
        if len(train) + len(g) <= train_size:
            train += g
        elif len(valid) + len(g) <= valid_size:
            valid += g
        else:
            test += g

    return {"train": sorted(train), "valid": sorted(valid), "test": sorted(test)}


# --------------------------------------------------------------------------
# 3. Save / load
# --------------------------------------------------------------------------
def split_path(i: int, splits_dir: Path = SPLITS_DIR) -> Path:
    return splits_dir / f"split_{i}.json"


def save_split(
    split: dict[str, list[int]], i: int, seed: int, df: pd.DataFrame,
    splits_dir: Path = SPLITS_DIR,
) -> Path:
    """Write one split as JSON, tagged with the dataset fingerprint."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "split_id": i,
        "seed": seed,
        "method": "bemis_murcko_balanced_scaffold",
        "fractions": list(FRACTIONS),
        "n_molecules": len(df),
        "dataset_fingerprint": dataset_fingerprint(df),
        **split,
    }
    path = split_path(i, splits_dir)
    path.write_text(json.dumps(payload, indent=1))
    return path


def load_split(i: int, df: pd.DataFrame, splits_dir: Path = SPLITS_DIR) -> dict[str, np.ndarray]:
    """Load split i and check it belongs to this exact dataframe.

    Returns {"train": idx, "valid": idx, "test": idx} as integer arrays, ready
    for df.iloc[idx]. Raises if the dataset has changed since the split was
    written, because the indices would then point at the wrong molecules.
    """
    payload = json.loads(split_path(i, splits_dir).read_text())
    expected, actual = payload["dataset_fingerprint"], dataset_fingerprint(df)
    if expected != actual:
        raise ValueError(
            f"split_{i}.json was made for dataset {expected}, but the current "
            f"dataset is {actual}. The cleaning or the source data changed; "
            f"do not use these indices."
        )
    return {k: np.asarray(payload[k], dtype=int) for k in ("train", "valid", "test")}
