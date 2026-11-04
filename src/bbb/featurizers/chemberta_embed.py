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
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel

from bbb.models.chemberta import MODEL_NAME, MODEL_REVISION, load_tokenizer

POOLINGS = ("cls", "mean")


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

        if pooling == "cls":
            vec = table[:, 0, :]                        # row 0 = [CLS]
        else:
            # attention_mask is 1 for real tokens, 0 for padding. Zero out the
            # padding rows, then divide by the number of real tokens.
            mask = enc["attention_mask"].unsqueeze(-1).float()   # [batch, tokens, 1]
            vec = (table * mask).sum(dim=1) / mask.sum(dim=1)

        out.append(vec.numpy().astype(np.float32))
    return np.concatenate(out)
