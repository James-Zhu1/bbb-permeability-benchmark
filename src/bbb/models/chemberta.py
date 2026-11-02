"""ChemBERTa-2 fine-tuning: a pre-trained SMILES language model + new classifier head.

How it differs from the GNN:
  - Input is the SMILES *text*, split into tokens (mostly one per character),
    not a graph. The model must infer bonds/rings from the text.
  - The body (3 transformer layers, 3.4M params) starts from weights
    PRE-TRAINED on 77M PubChem molecules (task: guess masked-out tokens).
    Only the small classifier head starts random.
  - Fine-tuning = train body + head together on BBB labels, with a SMALL
    learning rate so pre-trained knowledge is adjusted, not overwritten.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as hf_logging
from transformers import set_seed

from bbb.metrics import compute_metrics

hf_logging.set_verbosity_error()  # hide the expected "pooler newly initialized" notice

MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"
# Exact model version on Hugging Face, so a future upload can't silently change results.
MODEL_REVISION = "ed8a5374f2024ec8da53760af91a33fb8f6a15ff"


@dataclass
class ChemBERTaConfig:
    lr: float = 2e-5            # standard BERT fine-tuning rate (~50x smaller than the GNN's)
    weight_decay: float = 0.01  # standard AdamW default for transformers
    batch_size: int = 32
    max_epochs: int = 50
    patience: int = 10          # each epoch here is a big step; stop after 10 without gain
    max_length: int = 512       # model's limit; longest SMILES here is 236 tokens


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)


def load_model():
    """Pre-trained body + a NEW random classification head (1 output = logit).

    The standard RoBERTa head: take the final vector of the start token <s>
    (a summary slot the model learns to fill), then
    dropout -> dense(384->384) -> tanh -> dropout -> linear(384->1).

    The pinned revision ships weights only as pytorch_model.bin (a pickle).
    torch >= 2.6 loads it with weights_only=True, which reads plain tensors
    and refuses to execute code, so this is safe. (A model.safetensors exists
    only on an unmerged bot PR branch, refs/pr/2; we don't pin to that.)
    """
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, num_labels=1
    )


def _batches(smiles, labels, tokenizer, cfg: ChemBERTaConfig, order):
    """Yield (tokenized batch, labels) in the given row order."""
    for start in range(0, len(order), cfg.batch_size):
        idx = order[start:start + cfg.batch_size]
        enc = tokenizer(
            [smiles[i] for i in idx],
            padding=True,          # pad shorter SMILES so the batch is a rectangle
            truncation=True,
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        yield enc, (None if labels is None else torch.tensor(labels[idx], dtype=torch.float))


@torch.no_grad()
def predict_proba(model, tokenizer, smiles: list[str], cfg: ChemBERTaConfig = ChemBERTaConfig()) -> np.ndarray:
    """P(BBB+) for each SMILES, in input order."""
    model.eval()  # dropout off
    out = []
    for enc, _ in _batches(smiles, None, tokenizer, cfg, np.arange(len(smiles))):
        out.append(torch.sigmoid(model(**enc).logits.squeeze(-1)))
    return torch.cat(out).numpy()


def train_chemberta(
    train_smiles: list[str], train_y: np.ndarray,
    valid_smiles: list[str], valid_y: np.ndarray,
    seed: int, cfg: ChemBERTaConfig = ChemBERTaConfig(),
):
    """Fine-tune with early stopping on validation AUROC.

    Same recipe as the GNN: one pass over train per epoch, check valid,
    keep the best epoch's weights. The test set is never seen here.
    Returns (model restored to best epoch, tokenizer, history).
    """
    set_seed(seed)  # seeds python, numpy and torch: same head init + dropout each run
    tokenizer = load_tokenizer()
    model = load_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)

    best_auroc, best_state, best_epoch = -1.0, None, 0
    history = {"train_loss": [], "valid_auroc": []}

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()  # dropout on
        total = 0.0
        order = rng.permutation(len(train_smiles))  # new shuffle every epoch
        for enc, yb in _batches(train_smiles, train_y, tokenizer, cfg, order):
            optimizer.zero_grad()
            loss = loss_fn(model(**enc).logits.squeeze(-1), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
        history["train_loss"].append(total / len(train_smiles))

        auroc = compute_metrics(valid_y, predict_proba(model, tokenizer, valid_smiles, cfg))["auroc"]
        history["valid_auroc"].append(auroc)
        if auroc > best_auroc:
            best_auroc, best_epoch = auroc, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            break

    model.load_state_dict(best_state)
    history.update(best_epoch=best_epoch, best_valid_auroc=best_auroc,
                   epochs_run=epoch, config=asdict(cfg),
                   model_name=MODEL_NAME, model_revision=MODEL_REVISION)
    return model, tokenizer, history
