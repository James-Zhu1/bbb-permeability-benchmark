"""Step 4: FROZEN Boltz-2 embeddings + trained head, across the 5 splits of a protocol.

    python scripts/run_boltz_frozen.py                   # scaffold5
    python scripts/run_boltz_frozen.py --protocol tdc    # TDC leaderboard protocol

Prerequisites (see slurm/boltz_embed.sbatch): boltz_export_inputs.py ->
Boltz-2 GPU job -> boltz_pool_embeddings.py, producing data/boltz/full/s_mean.npz.

Same head and training as frozen ChemBERTa (models/mlp_head.py), so the only
difference between the two is where the 384-number vector comes from.

Molecules Boltz could not embed (1 of 1966: a 229-atom glycopeptide that
exceeds Boltz's 4-character atom-name limit) get, in each split, the MEAN
vector of that split's TRAINING molecules -- a "typical molecule" stand-in
that uses no valid/test information. Recorded in the results file.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np
import torch

from bbb.data import REPO_ROOT, dataset_fingerprint, tdc_group
from bbb.featurizers.boltz_embed import POOLINGS, impute_from_train, load_vectors
from bbb.metrics import compute_metrics, summarize
from bbb.models.mlp_head import HeadConfig, predict_proba, train_head
from bbb.splits import N_SPLITS, PROTOCOLS, load_protocol, load_split

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=PROTOCOLS, default="scaffold5")
    parser.add_argument("--pooling", choices=POOLINGS, default="mean")
    args = parser.parse_args()

    torch.set_num_threads(4)
    cfg = HeadConfig()

    df, splits_dir = load_protocol(args.protocol)
    y = df["label"].to_numpy()
    X_all, missing = load_vectors(df, args.protocol, pooling=args.pooling)

    per_split, test_preds = [], []
    print(f"[{args.protocol}] frozen Boltz-2 (s, {args.pooling} over atoms), vectors {X_all.shape}, "
          f"rows without embedding: {np.flatnonzero(missing).tolist()}")
    print(f"{'split':>5} | {'best epoch':>10} | {'valid AUROC':>11} | {'test AUROC':>10} | "
          f"{'test AUPRC':>10} | imputed")
    for i in range(N_SPLITS):
        idx = load_split(i, df, splits_dir)
        tr, va, te = idx["train"], idx["valid"], idx["test"]

        # Impute missing rows with THIS split's training mean (no valid/test info).
        X = impute_from_train(X_all, missing, tr)
        where = {name: np.flatnonzero(missing[s]).size for name, s in (("train", tr), ("valid", va), ("test", te))}

        model, history = train_head(X[tr], y[tr], X[va], y[va], seed=i, cfg=cfg)
        p_test = predict_proba(model, X[te])  # test set touched exactly once, here
        valid = compute_metrics(y[va], predict_proba(model, X[va]))
        test = compute_metrics(y[te], p_test)

        per_split.append({"split": i, "valid": valid, "test": test, "history": history,
                          "imputed_rows_by_set": where})
        test_preds.append(p_test)
        print(f"{i:>5} | {history['best_epoch']:>4} / {history['epochs_run']:<3} | "
              f"{valid['auroc']:>11.3f} | {test['auroc']:>10.3f} | {test['auprc']:>10.3f} | "
              f"{ {k: v for k, v in where.items() if v} or '-' }")

    summary = {part: summarize([r[part] for r in per_split]) for part in ("valid", "test")}
    t = summary["test"]
    print(f"\nTEST  AUROC {t['auroc']['mean']:.3f} ± {t['auroc']['std']:.3f}   "
          f"AUPRC {t['auprc']['mean']:.3f} ± {t['auprc']['std']:.3f}   "
          f"(random AUPRC {t['auprc_random']['mean']:.3f})")

    result = {
        "model": f"frozen Boltz-2 (boltz 2.2.1, boltz2_conf) trunk s, {args.pooling} over atoms + MLP head",
        "pooling": args.pooling,
        "protocol": args.protocol,
        "boltz_settings": "ligand-only input, largest fragment, --sampling_steps 10 "
                          "--diffusion_samples 1 --seed 0 (s is pre-diffusion; verified identical to 200 steps)",
        "head_config": asdict(cfg),
        "missing_rows": np.flatnonzero(missing).tolist(),
        "imputation": "missing rows set to the per-split training-set mean vector",
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

    out = REPO_ROOT / "results" / f"boltz_frozen_{args.pooling}_{args.protocol}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"saved {out.relative_to(REPO_ROOT)}")

if __name__ == "__main__":
    main()
