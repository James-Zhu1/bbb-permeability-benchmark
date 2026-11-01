"""SMILES -> molecular graph (PyTorch-Geometric `Data`) for GNNs.

Nodes = atoms, edges = bonds. Each atom gets a feature vector describing it
locally (element, charge, ...); each bond gets one too (single/double/...).
The GNN then learns how to combine these across the molecule; unlike a
fingerprint, nothing about "which substructures matter" is fixed in advance.
"""

from __future__ import annotations

import torch
from rdkit import Chem
from torch_geometric.data import Data

# --------------------------------------------------------------------------
# Feature vocabularies. Each feature is one-hot encoded; values not in the
# list fall into a final "other" slot, so unusual atoms never crash things.
# --------------------------------------------------------------------------
ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "P", "I"]
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-1, 0, 1]
NUM_HS = [0, 1, 2, 3]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot(value, choices: list) -> list[float]:
    """[0,..,1,..,0] with a 1 at value's position, or in the extra last slot."""
    vec = [0.0] * (len(choices) + 1)
    vec[choices.index(value) if value in choices else -1] = 1.0
    return vec


def atom_features(atom: Chem.Atom) -> list[float]:
    """What an atom 'looks like' locally. H-bond donors (N-H, O-H), charge
    and aromaticity are all relevant to crossing the BBB."""
    return (
        one_hot(atom.GetSymbol(), ELEMENTS)            # which element      (10)
        + one_hot(atom.GetDegree(), DEGREES)           # heavy-atom bonds    (7)
        + one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)  # charge          (4)
        + one_hot(atom.GetTotalNumHs(), NUM_HS)        # attached H's        (5)
        + one_hot(atom.GetHybridization(), HYBRIDIZATIONS)  # geometry       (4)
        + [float(atom.GetIsAromatic()), float(atom.IsInRing())]  #           (2)
    )


def bond_features(bond: Chem.Bond) -> list[float]:
    return (
        one_hot(bond.GetBondType(), BOND_TYPES)        # bond order          (5)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]  #         (2)
    )


N_ATOM_FEATURES = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))  # 32
N_BOND_FEATURES = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))  # 7


def smiles_to_graph(smiles: str, label: int | None = None) -> Data:
    """One molecule -> one PyG graph.

    x          [n_atoms, 32]      atom feature matrix
    edge_index [2, 2*n_bonds]     which atoms are connected. Each bond is
                                  stored twice (i->j and j->i) because PyG
                                  edges are directed and a bond goes both ways.
    edge_attr  [2*n_bonds, 7]     bond features, one row per directed edge
    y          [1]                label (if given)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles}")

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    src, dst, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        f = bond_features(bond)
        src += [i, j]
        dst += [j, i]
        edge_feats += [f, f]

    edge_index = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)
    edge_attr = torch.tensor(edge_feats, dtype=torch.float).reshape(-1, N_BOND_FEATURES)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    if label is not None:
        data.y = torch.tensor([float(label)])
    return data


def graphs_from_smiles(smiles: list[str], labels=None) -> list[Data]:
    """Featurize a whole dataset (call once; graphs are reused across splits)."""
    if labels is None:
        return [smiles_to_graph(s) for s in smiles]
    return [smiles_to_graph(s, int(y)) for s, y in zip(smiles, labels)]
