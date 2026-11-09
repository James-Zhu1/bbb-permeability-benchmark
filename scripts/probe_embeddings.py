"""Diagnostic: what physicochemical information is linearly recoverable from each representation?

    python scripts/probe_embeddings.py

Asks why frozen Boltz-2 underperforms. A "linear probe" fits a plain linear
model from a frozen representation to a known molecular property. If the
property cannot be read out linearly, the representation does not encode it in
a form a small head can use.

The properties are the ones that actually drive passive BBB permeability:
molecular weight and heavy-atom count (size), TPSA (polar surface area),
H-bond donors/acceptors, and cLogP (greasiness). Rotatable bonds are included
as a flexibility control.

Protocol mirrors the benchmark: the same 5 scaffold5 splits, fit on TRAIN
only (RidgeCV picks its own alpha by cross-validation inside the training
set), scored as R^2 on the held-out TEST molecules. R^2 = 1 is perfect,
0 means no better than predicting the training mean.

Writes results/probe_scaffold5.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bbb.data import REPO_ROOT
from bbb.featurizers.boltz_embed import impute_from_train, load_vectors
from bbb.featurizers.chemberta_embed import chemberta_embeddings
from bbb.featurizers.morgan import morgan_fingerprints
from bbb.splits import N_SPLITS, load_protocol, load_split

RDLogger.DisableLog("rdApp.*")

# Property name -> how to compute it from an RDKit molecule.
PROPERTIES = {
    "MolWt": Descriptors.MolWt,
    "HeavyAtoms": lambda m: float(m.GetNumHeavyAtoms()),
    "TPSA": Descriptors.TPSA,
    "HBondDonors": lambda m: float(Lipinski.NumHDonors(m)),
    "HBondAcceptors": lambda m: float(Lipinski.NumHAcceptors(m)),
    "cLogP": Crippen.MolLogP,
    "RotatableBonds": lambda m: float(Lipinski.NumRotatableBonds(m)),
}

ALPHAS = np.logspace(-2, 4, 13)  # RidgeCV searches these inside the training set


def properties_table(smiles: list[str]) -> pd.DataFrame:
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return pd.DataFrame({name: [fn(m) for m in mols] for name, fn in PROPERTIES.items()})


def main() -> None:
    df, splits_dir = load_protocol("scaffold5")
    smiles = df["smiles"].tolist()
    props = properties_table(smiles)

    # The three frozen representations, all row-aligned to this dataframe.
    boltz, missing = load_vectors(df, "scaffold5")
    reps = {
        "Morgan (2048 bits)": morgan_fingerprints(smiles).astype(np.float32),
        "ChemBERTa (mean)": chemberta_embeddings(smiles, pooling="mean"),
        # Pooling control: a mean divides by token count, so count-like
        # properties survive only as fractions; a sum keeps magnitude. If the
        # Boltz-2 H-bond-donor gap is caused by mean pooling rather than by the
        # representation, sum pooling should recover counts better here too.
        "ChemBERTa (sum)": chemberta_embeddings(smiles, pooling="sum"),
        "Boltz-2 s (mean)": boltz,
        # The same test applied to Boltz-2: if mean pooling is what loses the
        # H-bond donor counts, sum pooling should recover them here too.
        "Boltz-2 s (sum)": load_vectors(df, "scaffold5", pooling="sum")[0],
    }

    results: dict[str, dict[str, dict[str, float]]] = {}
    print(f"Linear probe: R^2 on held-out scaffold5 test molecules, mean ± std over {N_SPLITS} splits\n")
    header = f"{'property':<16}" + "".join(f"{name:>22}" for name in reps)
    print(header)
    print("-" * len(header))

    for prop in PROPERTIES:
        y = props[prop].to_numpy()
        row = f"{prop:<16}"
        for name, X_all in reps.items():
            scores = []
            for i in range(N_SPLITS):
                idx = load_split(i, df, splits_dir)
                tr, te = idx["train"], idx["test"]

                # Boltz has 2 rows it could not embed: fill from THIS split's
                # training mean, exactly as the benchmark runner does.
                X = impute_from_train(X_all, np.isnan(X_all).any(axis=1), tr)

                model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
                model.fit(X[tr], y[tr])
                scores.append(r2_score(y[te], model.predict(X[te])))

            results.setdefault(name, {})[prop] = {
                "r2_mean": float(np.mean(scores)),
                "r2_std": float(np.std(scores, ddof=1)),
            }
            row += f"{np.mean(scores):>15.3f} ± {np.std(scores, ddof=1):.3f}"
        print(row)

    out = REPO_ROOT / "results" / "probe_scaffold5.json"
    out.write_text(json.dumps({
        "description": "Linear probe (RidgeCV) from frozen representations to RDKit "
                       "physicochemical properties; fit on train, R^2 on test, per scaffold5 split",
        "protocol": "scaffold5",
        "alphas_searched": ALPHAS.tolist(),
        "boltz_missing_rows": np.flatnonzero(missing).tolist(),
        "results": results,
    }, indent=2))
    print(f"\nsaved {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
