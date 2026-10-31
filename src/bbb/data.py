"""Load BBB_Martins and turn it into the project's canonical dataframe.

Every model in this benchmark starts from `load_dataset()`. The split files in
`splits/` hold integer row indices into the dataframe returned here, so this
function MUST be deterministic: same input file -> same rows in the same order.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

# Repo root = three levels up from this file (src/bbb/data.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

RDLogger.DisableLog("rdApp.*")  # silence RDKit's per-molecule warnings


def download_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Fetch BBB_Martins via TDC (cached in data/raw after the first call).

    Returns TDC's dataframe with columns [Drug_ID, Drug, Y]:
    Drug = SMILES string, Y = 1 (BBB+) or 0 (BBB-).
    """
    from tdc.single_pred import ADME  # imported here: TDC is slow to import

    raw_dir.mkdir(parents=True, exist_ok=True)
    return ADME(name="BBB_Martins", path=str(raw_dir)).get_data()


_largest_fragment = rdMolStandardize.LargestFragmentChooser()


def standardize_smiles(smiles: str) -> str | None:
    """Parse, strip counter-ions, return RDKit canonical SMILES (None if invalid).

    - Salt stripping: "CC(C)NCC(O)COc1cccc2ccccc12.[Cl]" -> keep the biggest
      fragment (the drug), drop the chloride. The counter-ion doesn't change
      whether the drug crosses the BBB, but it would change the fingerprint.
    - Canonicalization: one molecule can be written as many SMILES strings;
      RDKit's canonical form gives each molecule exactly one string, so
      duplicate detection and scaffold computation are reliable.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = _largest_fragment.choose(mol)
    return Chem.MolToSmiles(mol)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw TDC frame -> canonical frame with columns [smiles, label].

    Steps:
      1. standardize every SMILES; drop any that fail to parse
      2. group identical molecules:
         - all copies agree on the label -> keep one row
         - copies disagree (BBB+ and BBB-) -> drop the molecule; there's no
           ground truth, and keeping it would put contradictory labels in
           the data
      3. sort by SMILES and reset the index, so row order doesn't depend on
         the order of rows in TDC's file
    """
    df = pd.DataFrame(
        {"smiles": raw["Drug"].map(standardize_smiles), "label": raw["Y"].astype(int)}
    )
    n_invalid = df["smiles"].isna().sum()
    df = df.dropna(subset=["smiles"])

    n_labels = df.groupby("smiles")["label"].nunique()
    conflicting = n_labels[n_labels > 1].index
    df = df[~df["smiles"].isin(conflicting)]

    n_before_dedup = len(df)
    df = df.drop_duplicates(subset="smiles")

    df = df.sort_values("smiles").reset_index(drop=True)

    print(
        f"[data] raw={len(raw)}  invalid={n_invalid}  "
        f"conflicting-label molecules dropped={len(conflicting)}  "
        f"duplicate rows merged={n_before_dedup - len(df)}  -> final={len(df)}"
    )
    return df


def load_dataset() -> pd.DataFrame:
    """The ONE entry point every model uses: canonical [smiles, label] frame."""
    return clean(download_raw())


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Short hash of the exact (smiles, label) rows, in order.

    Saved inside every split file. When a split is loaded, the hash is
    recomputed and compared, so if the cleaning logic or the TDC data ever
    changes, the stale indices fail loudly instead of silently pointing at
    the wrong molecules.
    """
    payload = "\n".join(f"{s}\t{y}" for s, y in zip(df["smiles"], df["label"]))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
    print(df["label"].value_counts(normalize=True).round(3).to_dict())
    print("fingerprint:", dataset_fingerprint(df))
