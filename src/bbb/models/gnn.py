"""GNN trained from scratch: model, training loop with early stopping, prediction.

Architecture (the 3 stages):
  1. ENCODE   each atom's 32 features -> 64 learned numbers
  2. MESSAGE  3 rounds of GINE message passing; each round, every atom
     PASSING  updates its vector from its neighbours (edge_index) with the
              bond features (edge_attr) shaping each message
  3. READOUT  average atom vectors per molecule -> small MLP -> 1 logit
Everything is trained together (end-to-end) from the BBB labels alone.
"""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

from bbb.featurizers.graph import N_ATOM_FEATURES, N_BOND_FEATURES
from bbb.metrics import compute_metrics


@dataclass
class GNNConfig:
    hidden: int = 64          # size of every atom vector
    n_layers: int = 3         # message-passing rounds = how many bonds each atom "sees"
    dropout: float = 0.1      # randomly zero 10% of numbers during training (anti-memorization)
    lr: float = 1e-3          # learning rate: how big each weight update is
    batch_size: int = 64      # molecules per training step
    max_epochs: int = 200     # hard upper limit
    patience: int = 30        # stop if valid AUROC hasn't improved for this many epochs


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------
class GNN(nn.Module):
    def __init__(self, cfg: GNNConfig):
        super().__init__()
        h = cfg.hidden

        # Stage 1: atom encoder (32 -> 64), shared by every atom
        self.encode = nn.Sequential(nn.Linear(N_ATOM_FEATURES, h), nn.ReLU())

        # Stage 2: message-passing layers. GINEConv computes, for each atom i:
        #   message from neighbour j = ReLU(h_j + bond_vector(i-j))
        #   new h_i = MLP( h_i + sum of messages )
        # `edge_dim` gives each layer its own bond encoder (7 -> 64).
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.n_layers):
            mlp = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
            self.convs.append(GINEConv(mlp, edge_dim=N_BOND_FEATURES))
            # BatchNorm rescales the numbers to a stable range each round,
            # which makes training faster and less sensitive to the lr.
            self.norms.append(nn.BatchNorm1d(h))
        self.dropout = nn.Dropout(cfg.dropout)

        # Stage 3: molecule vector (64) -> 1 number (a "logit")
        self.head = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(h, 1)
        )

    def forward(self, data) -> torch.Tensor:
        h = self.encode(data.x)                                   # [n_atoms, 64]
        for conv, norm in zip(self.convs, self.norms):
            update = conv(h, data.edge_index, data.edge_attr)
            update = self.dropout(torch.relu(norm(update)))
            # Residual connection: keep the old vector and ADD the update,
            # so each round refines the atom's description rather than
            # replacing it. Helps deeper GNNs train.
            h = h + update
        mol = global_mean_pool(h, data.batch)                     # [n_molecules, 64]
        return self.head(mol).squeeze(-1)                         # [n_molecules] logits


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Make weight init, dropout and batch shuffling reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_proba(model: GNN, graphs: list[Data], batch_size: int = 256) -> np.ndarray:
    """P(BBB+) for each graph, in input order."""
    model.eval()  # eval mode: dropout off, BatchNorm uses its learned statistics
    out = []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        out.append(torch.sigmoid(model(batch)))  # sigmoid: logit -> probability
    return torch.cat(out).numpy()


def train_gnn(
    train_graphs: list[Data], valid_graphs: list[Data], seed: int, cfg: GNNConfig = GNNConfig()
) -> tuple[GNN, dict]:
    """Train with early stopping on validation AUROC.

    Returns the model restored to its BEST validation epoch, and a history
    dict. The test set is never seen here.
    """
    set_seed(seed)
    model = GNN(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # BCEWithLogitsLoss = sigmoid + binary cross-entropy in one numerically
    # stable step. Loss is large when the model is confidently wrong.
    loss_fn = nn.BCEWithLogitsLoss()

    gen = torch.Generator().manual_seed(seed)  # reproducible shuffling
    loader = DataLoader(train_graphs, batch_size=cfg.batch_size, shuffle=True, generator=gen)
    y_valid = np.array([g.y.item() for g in valid_graphs])

    best_auroc, best_state, best_epoch = -1.0, None, 0
    history = {"train_loss": [], "valid_auroc": []}

    for epoch in range(1, cfg.max_epochs + 1):
        # ---- one epoch: every training molecule seen once ----
        model.train()  # train mode: dropout on, BatchNorm updates its statistics
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()                     # clear last step's gradients
            loss = loss_fn(model(batch), batch.y)     # forward pass + loss
            loss.backward()                           # backprop: gradient for every weight
            optimizer.step()                          # update every weight
            total += loss.item() * batch.num_graphs
        history["train_loss"].append(total / len(train_graphs))

        # ---- check validation; remember the best version ----
        auroc = compute_metrics(y_valid, predict_proba(model, valid_graphs))["auroc"]
        history["valid_auroc"].append(auroc)
        if auroc > best_auroc:
            best_auroc, best_epoch = auroc, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            break  # early stopping: no improvement for `patience` epochs

    model.load_state_dict(best_state)
    history.update(best_epoch=best_epoch, best_valid_auroc=best_auroc,
                   epochs_run=epoch, config=asdict(cfg))
    return model, history
