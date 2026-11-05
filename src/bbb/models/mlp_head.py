"""A small classifier head trained on FIXED vectors (frozen embeddings).

Used for frozen ChemBERTa (Step 3b) and reusable for frozen Boltz-2 (Step 4):
any model that turns a molecule into a d-number vector can be plugged in.

Architecture = the same design as the fine-tuned ChemBERTa head
(RobertaClassificationHead), so the only difference vs fine-tuning is that
the body is frozen:
    standardize -> dropout -> Linear(d, d) -> tanh -> dropout -> Linear(d, 1) -> logit

Standardization (added in Step 4): different frozen models produce vectors on
very different scales (ChemBERTa per-dim std ~0.1; Boltz-2 ~7, some dims ~50,
values up to ~950). Unscaled, large inputs saturate tanh and the head mostly
reacts to a few huge dimensions. So every input dimension is rescaled to
mean 0 / std 1 using statistics from the TRAINING vectors of each split only,
stored inside the model and applied identically to train, valid and test.
Applied to every frozen model, so they all get the same preprocessing.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from bbb.metrics import compute_metrics


@dataclass
class HeadConfig:
    dropout: float = 0.144    # ChemBERTa-77M-MLM's own dropout rate (its config)
    lr: float = 1e-3          # usual rate for a fresh small network (the body is
                              # frozen, so there's nothing pre-trained to protect)
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 30        # same early-stopping rule as the GNN
    standardize: bool = True  # rescale inputs with training-set mean/std (see top)


class MLPHead(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        # Standardization statistics: buffers = saved with the model, not trained.
        # Defaults (0, 1) mean "no rescaling" until set by train_head.
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sigma", torch.ones(d))
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(d, d)        # re-mix the d inputs into d new features
        self.out_proj = nn.Linear(d, 1)     # combine them into one logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mu) / self.sigma
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x).squeeze(-1)


@torch.no_grad()
def predict_proba(model: MLPHead, X: np.ndarray) -> np.ndarray:
    model.eval()  # dropout off
    return torch.sigmoid(model(torch.from_numpy(X))).numpy()


def train_head(
    X_train: np.ndarray, y_train: np.ndarray,
    X_valid: np.ndarray, y_valid: np.ndarray,
    seed: int, cfg: HeadConfig = HeadConfig(),
) -> tuple[MLPHead, dict]:
    """Train the head with early stopping on validation AUROC (test never seen)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLPHead(X_train.shape[1], cfg.dropout)
    if cfg.standardize:
        # Fit on TRAINING vectors only; a constant dimension (std 0) is left unscaled.
        sd = X_train.std(axis=0)
        model.mu.copy_(torch.from_numpy(X_train.mean(axis=0)))
        model.sigma.copy_(torch.from_numpy(np.where(sd > 0, sd, 1.0).astype(np.float32)))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(X_train), torch.tensor(y_train, dtype=torch.float)

    best_auroc, best_state, best_epoch = -1.0, None, 0
    history = {"train_loss": [], "valid_auroc": []}
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = torch.from_numpy(rng.permutation(len(Xt)))
        total = 0.0
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        history["train_loss"].append(total / len(Xt))

        auroc = compute_metrics(y_valid, predict_proba(model, X_valid))["auroc"]
        history["valid_auroc"].append(auroc)
        if auroc > best_auroc:
            best_auroc, best_epoch = auroc, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            break

    model.load_state_dict(best_state)
    history.update(best_epoch=best_epoch, best_valid_auroc=best_auroc,
                   epochs_run=epoch, config=asdict(cfg))
    return model, history
