"""Frozen ChemBERTa-2: SMILES -> one fixed 384-number vector per molecule.

The body runs once per molecule with NO training (eval mode, no gradients),
so DeepChem's pre-trained weights are never changed. Only a head trained on
these vectors learns anything about BBB.

The body outputs a TABLE per molecule (tokens x 384). Two ways to turn that
into one row:
  cls   take row 0, the [CLS] slot -- exactly what the fine-tuned head read.
        Pre-training never taught this slot to summarize the molecule, so
        frozen, it may carry little information.
  mean  average every real token's row (padding excluded) -- the standard
        way to get a molecule vector from a frozen RoBERTa-style model.
  sum   add every real token's row instead of averaging. A mean divides by
        token count, so count-like properties ("how many N-H groups") survive
        only as fractions; a sum keeps magnitude. Used to test whether mean
        pooling is what loses H-bond donor counts (see scripts/probe_embeddings.py).
  max   per-dimension maximum over real tokens: "is there any token that
        strongly activates this feature", regardless of molecule size.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel

from bbb.models.chemberta import MODEL_NAME, MODEL_REVISION, load_tokenizer

POOLINGS = ("cls", "mean", "sum", "max")


@torch.no_grad()
def chemberta_embeddings(smiles: list[str], pooling: str, batch_size: int = 64) -> np.ndarray:
    """Return an (n_molecules, 384) float32 matrix of frozen ChemBERTa vectors."""
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")

    tokenizer = load_tokenizer()
    body = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, add_pooling_layer=False)
    body.eval()  # dropout off -> the same SMILES always gives the same vector

    out = []
    for start in range(0, len(smiles), batch_size):
        enc = tokenizer(smiles[start:start + batch_size], padding=True,
                        truncation=True, max_length=512, return_tensors="pt")
        table = body(**enc).last_hidden_state          # [batch, tokens, 384]

        # attention_mask is 1 for real tokens, 0 for padding.
        mask = enc["attention_mask"].unsqueeze(-1).float()       # [batch, tokens, 1]
        if pooling == "cls":
            vec = table[:, 0, :]                                 # row 0 = [CLS]
        elif pooling == "mean":
            vec = (table * mask).sum(dim=1) / mask.sum(dim=1)
        elif pooling == "sum":
            vec = (table * mask).sum(dim=1)
        else:  # max: push padding to -inf so it can never win the maximum
            vec = table.masked_fill(mask == 0, float("-inf")).max(dim=1).values

        out.append(vec.numpy().astype(np.float32))
    return np.concatenate(out)
