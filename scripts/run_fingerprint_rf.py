"""Morgan fingerprints + RandomForest across the 5 committed splits of a protocol.

    python scripts/run_fingerprint_rf.py                   # scaffold5 (primary)
    python scripts/run_fingerprint_rf.py --protocol tdc    # TDC leaderboard protocol

Writes results/fingerprint_rf_{protocol}.json with per-split and summary metrics.
"""

from __future__ import annotations

import argparse
import json

from bbb.data import REPO_ROOT, dataset_fingerprint, tdc_group
from bbb.featurizers.morgan import morgan_fingerprints
from bbb.metrics import compute_metrics, summarize
from bbb.models.fingerprint_rf import RF_PARAMS, predict_proba, train_rf
from bbb.splits import N_SPLITS, PROTOCOLS, load_protocol, load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=PROTOCOLS, default="scaffold5")
    args = parser.parse_args()

    df, splits_dir = load_protocol(args.protocol)
    y = df["label"].to_numpy()

    # Featurize once. Each fingerprint depends only on its own molecule,
    # so computing them all up front can't leak information between sets.
    X = morgan_fingerprints(df["smiles"].tolist(), radius=2, n_bits=2048)

    per_split, test_preds = [], []
    print(f"[{args.protocol}]")
    print(f"{'split':>5} | {'valid AUROC':>11} | {'test AUROC':>10} | {'test AUPRC':>10} (random={'':>4})")
    for i in range(N_SPLITS):
        idx = load_split(i, df, splits_dir)
        tr, va, te = idx["train"], idx["valid"], idx["test"]

        model = train_rf(X[tr], y[tr], seed=i)
        p_test = predict_proba(model, X[te])
        valid = compute_metrics(y[va], predict_proba(model, X[va]))
        test = compute_metrics(y[te], p_test)

        per_split.append({"split": i, "valid": valid, "test": test})
        test_preds.append(p_test)
        print(
            f"{i:>5} | {valid['auroc']:>11.3f} | {test['auroc']:>10.3f} | "
            f"{test['auprc']:>10.3f} (random={test['auprc_random']:.3f})"
        )

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
        "model": "morgan_r2_2048 + RandomForest",
        "protocol": args.protocol,
        "rf_params": RF_PARAMS,
        "dataset_fingerprint": dataset_fingerprint(df),
        "per_split": per_split,
        "summary": summary,
    }

    if args.protocol == "tdc":
        # Cross-check with TDC's official evaluator. Our test indices are in
        # TDC's test-file order, so predictions line up with its labels.
        # TDC reports population std (ddof=0); quote THIS for leaderboard use.
        tdc_mean, tdc_std = tdc_group().evaluate_many(
            [{"BBB_Martins": p} for p in test_preds]
        )["bbb_martins"]
        print(f"TDC official evaluator: AUROC {tdc_mean:.3f} ± {tdc_std:.3f}  (leaderboard format)")
        result["tdc_official"] = {"auroc_mean": tdc_mean, "auroc_std": tdc_std}

    out = REPO_ROOT / "results" / f"fingerprint_rf_{args.protocol}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"saved {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
