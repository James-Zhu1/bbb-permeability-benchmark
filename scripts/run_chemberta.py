"""Step 3: ChemBERTa-2 fine-tuning across the 5 committed splits of a protocol.

    python scripts/run_chemberta.py                   # scaffold5 (primary)
    python scripts/run_chemberta.py --protocol tdc    # TDC leaderboard protocol

~2 min per split on CPU. Writes results/chemberta_{protocol}.json.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import torch

from bbb.data import REPO_ROOT, dataset_fingerprint, tdc_group
from bbb.metrics import compute_metrics, summarize
from bbb.models.chemberta import (
    MODEL_NAME,
    MODEL_REVISION,
    ChemBERTaConfig,
    predict_proba,
    train_chemberta,
)
from bbb.splits import N_SPLITS, PROTOCOLS, load_protocol, load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=PROTOCOLS, default="scaffold5")
    args = parser.parse_args()

    torch.set_num_threads(4)  # CPU beat Apple MPS for this small model (4.0s vs 6.5s/epoch)
    cfg = ChemBERTaConfig()

    df, splits_dir = load_protocol(args.protocol)
    smiles = df["smiles"].tolist()
    y = df["label"].to_numpy()
    pick = lambda idx: [smiles[i] for i in idx]

    per_split, test_preds = [], []
    print(f"[{args.protocol}]  {MODEL_NAME}@{MODEL_REVISION[:8]}")
    print(f"{'split':>5} | {'best epoch':>10} | {'valid AUROC':>11} | {'test AUROC':>10} | "
          f"{'test AUPRC':>10} | {'time':>5}", flush=True)
    for i in range(N_SPLITS):
        idx = load_split(i, df, splits_dir)
        tr, va, te = idx["train"], idx["valid"], idx["test"]

        t0 = time.time()
        model, tok, history = train_chemberta(pick(tr), y[tr], pick(va), y[va], seed=i, cfg=cfg)
        p_test = predict_proba(model, tok, pick(te), cfg)  # test set touched exactly once, here
        valid = compute_metrics(y[va], predict_proba(model, tok, pick(va), cfg))
        test = compute_metrics(y[te], p_test)

        per_split.append({"split": i, "valid": valid, "test": test, "history": history})
        test_preds.append(p_test)
        print(f"{i:>5} | {history['best_epoch']:>4} / {history['epochs_run']:<3} | "
              f"{valid['auroc']:>11.3f} | {test['auroc']:>10.3f} | {test['auprc']:>10.3f} | "
              f"{time.time() - t0:>4.0f}s", flush=True)

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
        "model": f"ChemBERTa-2 fine-tuned ({MODEL_NAME})",
        "model_revision": MODEL_REVISION,
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

    out = REPO_ROOT / "results" / f"chemberta_{args.protocol}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"saved {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
