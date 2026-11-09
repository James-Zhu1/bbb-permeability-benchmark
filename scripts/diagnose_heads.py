"""Diagnostic: is the MLP head the bottleneck, and how much is just molecular size?

    python scripts/diagnose_heads.py                   # scaffold5
    python scripts/diagnose_heads.py --protocol tdc

Two controls for the frozen-representation results:

1. LOGISTIC REGRESSION on the same vectors. The benchmark head has 148k
   parameters trained on ~1,500 molecules, so it could be overfitting. Plain
   L2 logistic regression has 385. If it scores HIGHER than the MLP head, the
   head was the bottleneck, not the representation.
2. SIZE-ONLY baseline: two features (heavy-atom count, molecular weight).
   Small molecules cross the BBB more often, so this is the floor any
   representation must clear to be doing more than measuring size.

Both use the same splits and the same shared metrics as every other model.
C (regularization strength) is picked by cross-validation inside the TRAINING
set only, so no valid/test information is used.

Writes results/diagnose_heads_{protocol}.json.
"""

from __future__ import annotations

import argparse
import json
import os

# LogisticRegressionCV(n_jobs=-1) forks after the tokenizer has run; silence the
# resulting "process just got forked" warnings from HuggingFace tokenizers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bbb.data import REPO_ROOT, dataset_fingerprint
from bbb.featurizers.boltz_embed import impute_from_train, load_vectors
from bbb.featurizers.chemberta_embed import chemberta_embeddings
from bbb.featurizers.morgan import morgan_fingerprints
from bbb.metrics import compute_metrics, summarize
from bbb.splits import N_SPLITS, PROTOCOLS, load_protocol, load_split

RDLogger.DisableLog("rdApp.*")
CS = np.logspace(-3, 3, 7)


def size_features(smiles: list[str]) -> np.ndarray:
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return np.array([[m.GetNumHeavyAtoms(), Descriptors.MolWt(m)] for m in mols], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=PROTOCOLS, default="scaffold5")
    args = parser.parse_args()

    df, splits_dir = load_protocol(args.protocol)
    smiles = df["smiles"].tolist()
    y = df["label"].to_numpy()

    boltz, missing = load_vectors(df, args.protocol)
    reps = {
        "Size only (2 features)": size_features(smiles),
        "Morgan (2048 bits)": morgan_fingerprints(smiles).astype(np.float32),
        "ChemBERTa frozen (mean)": chemberta_embeddings(smiles, pooling="mean"),
        "Boltz-2 frozen (s, mean)": boltz,
    }

    print(f"[{args.protocol}] logistic regression on frozen vectors, test AUROC\n")
    print(f"{'representation':<26} {'per split':<32} {'mean ± std':>14}")
    print("-" * 74)

    results = {}
    for name, X_all in reps.items():
        scores = []
        per_split = []
        for i in range(N_SPLITS):
            idx = load_split(i, df, splits_dir)
            tr, te = idx["train"], idx["test"]
            X = impute_from_train(X_all, np.isnan(X_all).any(axis=1), tr)

            model = make_pipeline(
                StandardScaler(),
                LogisticRegressionCV(Cs=CS, max_iter=5000, scoring="roc_auc", n_jobs=-1),
            )
            model.fit(X[tr], y[tr])
            m = compute_metrics(y[te], model.predict_proba(X[te])[:, 1])
            per_split.append({"split": i, "test": m})
            scores.append(m["auroc"])

        results[name] = {
            "per_split": per_split,
            "summary": summarize([p["test"] for p in per_split]),
        }
        line = " ".join(f"{s:.3f}" for s in scores)
        print(f"{name:<26} {line:<32} {np.mean(scores):>7.3f} ± {np.std(scores, ddof=1):.3f}")

    out = REPO_ROOT / "results" / f"diagnose_heads_{args.protocol}.json"
    out.write_text(json.dumps({
        "description": "Controls for the frozen-representation results: L2 logistic "
                       "regression (C by CV inside train) on the same vectors as the MLP head, "
                       "plus a size-only baseline (heavy atoms + MolWt).",
        "protocol": args.protocol,
        "Cs_searched": CS.tolist(),
        "dataset_fingerprint": dataset_fingerprint(df),
        "results": results,
    }, indent=2))
    print(f"\nsaved {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
