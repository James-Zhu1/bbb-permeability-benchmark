"""Step 3b: FROZEN ChemBERTa-2 + trained head, across the 5 splits of a protocol.

Compare with run_chemberta.py (fine-tuned): same body, same head design,
but here the body never changes -- only the head learns.

    python scripts/run_chemberta_frozen.py --pooling cls     # [CLS] row (as fine-tuned head read)
    python scripts/run_chemberta_frozen.py --pooling mean    # average of token rows
    add --protocol tdc for the TDC leaderboard protocol

Writes results/chemberta_frozen_{pooling}_{protocol}.json.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from bbb.data import REPO_ROOT, dataset_fingerprint, tdc_group
from bbb.featurizers.chemberta_embed import POOLINGS, chemberta_embeddings
from bbb.metrics import compute_metrics, summarize
from bbb.models.chemberta import MODEL_NAME, MODEL_REVISION
from bbb.models.mlp_head import HeadConfig, predict_proba, train_head
from bbb.splits import N_SPLITS, PROTOCOLS, load_protocol, load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=PROTOCOLS, default="scaffold5")
    parser.add_argument("--pooling", choices=POOLINGS, required=True)
    args = parser.parse_args()

    torch.set_num_threads(4)
    cfg = HeadConfig()

    df, splits_dir = load_protocol(args.protocol)
    y = df["label"].to_numpy()

    # Frozen body: embed every molecule ONCE. Each vector depends only on its
    # own molecule and the fixed weights, so this can't leak between sets.
    X = chemberta_embeddings(df["smiles"].tolist(), pooling=args.pooling)

    per_split, test_preds = [], []
    print(f"[{args.protocol}] frozen ChemBERTa, pooling={args.pooling}, vectors {X.shape}")
    print(f"{'split':>5} | {'best epoch':>10} | {'valid AUROC':>11} | {'test AUROC':>10} | {'test AUPRC':>10}")
    for i in range(N_SPLITS):
        idx = load_split(i, df, splits_dir)
        tr, va, te = idx["train"], idx["valid"], idx["test"]

        model, history = train_head(X[tr], y[tr], X[va], y[va], seed=i, cfg=cfg)
        p_test = predict_proba(model, X[te])  # test set touched exactly once, here
        valid = compute_metrics(y[va], predict_proba(model, X[va]))
        test = compute_metrics(y[te], p_test)

        per_split.append({"split": i, "valid": valid, "test": test, "history": history})
        test_preds.append(p_test)
        print(f"{i:>5} | {history['best_epoch']:>4} / {history['epochs_run']:<3} | "
              f"{valid['auroc']:>11.3f} | {test['auroc']:>10.3f} | {test['auprc']:>10.3f}")

    summary = {part: summarize([r[part] for r in per_split]) for part in ("valid", "test")}
    t = summary["test"]
    print(f"\nTEST  AUROC {t['auroc']['mean']:.3f} ± {t['auroc']['std']:.3f}   "
          f"AUPRC {t['auprc']['mean']:.3f} ± {t['auprc']['std']:.3f}   "
          f"(random AUPRC {t['auprc_random']['mean']:.3f})")

    result = {
        "model": f"frozen ChemBERTa-2 ({MODEL_NAME}) + MLP head, pooling={args.pooling}",
        "model_revision": MODEL_REVISION,
        "protocol": args.protocol,
        "pooling": args.pooling,
        "head_config": asdict(cfg),
        "dataset_fingerprint": dataset_fingerprint(df),
        "per_split": per_split,
        "summary": summary,
    }
    if args.protocol == "tdc":
        tdc_mean, tdc_std = tdc_group().evaluate_many(
            [{"BBB_Martins": p} for p in test_preds]
        )["bbb_martins"]
        print(f"TDC official evaluator: AUROC {tdc_mean:.3f} ± {tdc_std:.3f}  (leaderboard format)")
        result["tdc_official"] = {"auroc_mean": tdc_mean, "auroc_std": tdc_std}

    out = REPO_ROOT / "results" / f"chemberta_frozen_{args.pooling}_{args.protocol}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"saved {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
