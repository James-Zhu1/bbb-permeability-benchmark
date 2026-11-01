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

from bbb.data import (
    REPO_ROOT,
    dataset_fingerprint,
    load_dataset,
    load_tdc_benchmark,
    tdc_group,
)

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
    method: str = "bemis_murcko_balanced_scaffold",
    fractions: tuple[float, float, float] = FRACTIONS,
) -> Path:
    """Write one split as JSON, tagged with the dataset fingerprint."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "split_id": i,
        "seed": seed,
        "method": method,
        "fractions": list(fractions),
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


# --------------------------------------------------------------------------
# 4. TDC benchmark-group protocol (leaderboard-comparable)
# --------------------------------------------------------------------------
TDC_SPLITS_DIR = SPLITS_DIR / "tdc"
TDC_SEEDS = (1, 2, 3, 4, 5)       # TDC's convention; split_id i uses seed i+1
TDC_FRACTIONS = (0.7, 0.1, 0.2)   # 20% fixed test; rest split 87.5/12.5


def tdc_split(df: pd.DataFrame, seed: int) -> dict[str, list[int]]:
    """TDC's own split for one seed, as row indices into load_tdc_benchmark().

    - test: TDC's fixed test set, identical for every seed.
    - train/valid: TDC's scaffold split of train_val with this seed.

    TDC returns dataframes with a reset index, so rows are matched back by
    (drug_id, smiles, label), which is unique per row in this dataset.
    """
    group = tdc_group()
    train_df, valid_df = group.get_train_valid_split(
        benchmark="BBB_Martins", split_type="default", seed=seed
    )
    row_of = {key: i for i, key in enumerate(zip(df["drug_id"], df["smiles"], df["label"]))}
    if len(row_of) != len(df):
        raise ValueError("(drug_id, smiles, label) is not unique; cannot map TDC rows")

    def to_idx(part: pd.DataFrame) -> list[int]:
        keys = zip(part["Drug_ID"].astype(str), part["Drug"], part["Y"].astype(int))
        return sorted(row_of[k] for k in keys)

    test = df.index[df["tdc_set"] == "test"].tolist()
    return {"train": to_idx(train_df), "valid": to_idx(valid_df), "test": test}


# --------------------------------------------------------------------------
# 5. Protocol registry: the one place that pairs a dataset with its splits
# --------------------------------------------------------------------------
PROTOCOLS = ("scaffold5", "tdc")


def load_protocol(name: str) -> tuple[pd.DataFrame, Path]:
    """Return (dataframe, splits_dir) for an evaluation protocol.

    scaffold5 - our cleaned data + 5 independent scaffold splits (primary,
                for comparing representations against each other)
    tdc       - TDC's raw data + fixed test set, 5 train/valid seeds
                (secondary, comparable to the TDC leaderboard)
    """
    if name == "scaffold5":
        return load_dataset(), SPLITS_DIR
    if name == "tdc":
        return load_tdc_benchmark(), TDC_SPLITS_DIR
    raise ValueError(f"unknown protocol {name!r}; choose from {PROTOCOLS}")
