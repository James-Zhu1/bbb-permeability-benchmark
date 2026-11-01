"""SMILES -> Morgan (ECFP-style) fingerprint matrix."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def morgan_fingerprints(smiles: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Return an (n_molecules, n_bits) uint8 matrix of Morgan fingerprint bits.

    radius=2 is ECFP4, the standard choice: each atom's environment up to 2
    bonds away is hashed into one of `n_bits` positions. A 1 at position k
    means "some substructure that hashes to k is present".

    The input SMILES are already canonical and valid (bbb.data guarantees
    this), so a parse failure here means something upstream broke.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    X = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"unparseable SMILES at row {i}: {s}")
        X[i] = gen.GetFingerprintAsNumPy(mol)
    return X
