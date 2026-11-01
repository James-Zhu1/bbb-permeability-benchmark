"""Evaluation metrics, shared by ALL models so every one is measured identically."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_metrics(y_true, y_score) -> dict[str, float]:
    """Score one set of predictions.

    y_true:  0/1 labels (1 = BBB+)
    y_score: predicted probability (or any score) that the molecule is BBB+.
             Must be a continuous score, NOT hard 0/1 predictions: both metrics
             measure how well the model *ranks* molecules.

    auroc        - P(random BBB+ molecule is scored above a random BBB- one).
                   0.5 = random, 1.0 = perfect. Not distorted by imbalance.
    auprc        - average precision for the positive class.
    auprc_random - what AUPRC a random model would get (= BBB+ rate).
                   Compare auprc against this, not against 0.5.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "auprc_random": float(y_true.mean()),
    }


def summarize(per_split: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Combine one metrics dict per split into {metric: {mean, std}}.

    std uses ddof=1 (sample std): the 5 splits are a sample of possible
    held-out scaffold regions, not the whole population of them.
    """
    keys = per_split[0].keys()
    return {
        k: {
            "mean": float(np.mean([m[k] for m in per_split])),
            "std": float(np.std([m[k] for m in per_split], ddof=1)),
        }
        for k in keys
    }
