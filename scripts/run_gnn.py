"""Step 2: GNN from scratch across the 5 committed splits of a protocol.

    python scripts/run_gnn.py                   # scaffold5 (primary)
    python scripts/run_gnn.py --protocol tdc    # TDC leaderboard protocol

Writes results/gnn_{protocol}.json with per-split metrics and training history.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import torch

from bbb.data import REPO_ROOT, dataset_fingerprint, tdc_group
from bbb.featurizers.graph import graphs_from_smiles
from bbb.metrics import compute_metrics, summarize
from bbb.models.gnn import GNNConfig, predict_proba, train_gnn
from bbb.splits import N_SPLITS, PROTOCOLS, load_protocol, load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=PROTOCOLS, default="scaffold5")
    args = parser.parse_args()

    torch.set_num_threads(4)  # small graphs: a few CPU threads is fastest
    cfg = GNNConfig()

    df, splits_dir = load_protocol(args.protocol)
    y = df["label"].to_numpy()

    # Build every molecule's graph once; each depends only on its own molecule.
    graphs = graphs_from_smiles(df["smiles"].tolist(), y)
    pick = lambda idx: [graphs[i] for i in idx]

    per_split, test_preds = [], []
    print(f"[{args.protocol}]")
    print(f"{'split':>5} | {'best epoch':>10} | {'valid AUROC':>11} | {'test AUROC':>10} | "
          f"{'test AUPRC':>10} | {'time':>5}")
    for i in range(N_SPLITS):
        idx = load_split(i, df, splits_dir)
        tr, va, te = idx["train"], idx["valid"], idx["test"]

        t0 = time.time()
        model, history = train_gnn(pick(tr), pick(va), seed=i, cfg=cfg)
        p_test = predict_proba(model, pick(te))  # test set touched exactly once, here
        valid = compute_metrics(y[va], predict_proba(model, pick(va)))
        test = compute_metrics(y[te], p_test)

        per_split.append({"split": i, "valid": valid, "test": test, "history": history})
        test_preds.append(p_test)
        print(f"{i:>5} | {history['best_epoch']:>4} / {history['epochs_run']:<3} | "
              f"{valid['auroc']:>11.3f} | {test['auroc']:>10.3f} | {test['auprc']:>10.3f} | "
              f"{time.time() - t0:>4.0f}s")

    summary = {
        part: summarize([r[part] for r in per_split]) for part in ("valid", "test")
    }
    t = summary["test"]
    print(
        f"\nTEST  AUROC {t['auroc']['mean']:.3f} ± {t['auroc']['std']:.3f}   "
        f"AUPRC {t['auprc']['mean']:.3f} ± {t['auprc']['std']:.3f}   "
        f"(random AUPRC {t['auprc_random']['mean']:.3f})"
    )

    result = {
        "model": "GINE GNN from scratch",
        "protocol": args.protocol,
        "config": asdict(cfg),
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

    out = REPO_ROOT / "results" / f"gnn_{args.protocol}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"saved {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
