"""Baseline model: RandomForest on Morgan fingerprints."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Fixed, standard settings: this is a baseline, so no hyperparameter search.
RF_PARAMS = {
    "n_estimators": 500,   # more trees = more stable scores; 500 is plenty here
    "max_features": "sqrt",  # each split considers sqrt(2048) ~ 45 random bits
    "n_jobs": -1,          # use all CPU cores
}


def train_rf(X_train: np.ndarray, y_train: np.ndarray, seed: int = 0) -> RandomForestClassifier:
    """Fit a RandomForest. `seed` makes the tree randomness reproducible."""
    model = RandomForestClassifier(**RF_PARAMS, random_state=seed)
    model.fit(X_train, y_train)
    return model


def predict_proba(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """P(BBB+) for each row: the fraction of trees voting for class 1."""
    return model.predict_proba(X)[:, 1]
