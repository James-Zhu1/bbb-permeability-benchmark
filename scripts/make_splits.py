"""Generate the 5 scaffold splits, save them to splits/, and verify them.

Run ONCE, then commit splits/. Every model loads these files; none re-split.

    python scripts/make_splits.py          # refuses to overwrite existing splits
    python scripts/make_splits.py --force  # regenerate (invalidates old results!)
"""

from __future__ import annotations

import argparse
import sys

from bbb.data import load_dataset
from bbb.splits import (
    N_SPLITS,
    load_split,
    murcko_scaffold,
    save_split,
    scaffold_split,
    split_path,
)


def verify(i: int, df, scaffolds: list[str]) -> dict:
    """Reload split i from disk and check every invariant. Raises on failure."""
    sp = load_split(i, df)  # also checks the dataset fingerprint
    tr, va, te = (set(sp[k].tolist()) for k in ("train", "valid", "test"))
    n = len(df)

    # 1. the three sets don't overlap, and together they cover every molecule
    assert not (tr & va or tr & te or va & te), f"split {i}: sets overlap"
    assert tr | va | te == set(range(n)), f"split {i}: molecules missing"

    # 2. leakage rule: no scaffold appears in more than one set
    scaf = lambda idx: {scaffolds[j] for j in idx}
    assert not (scaf(tr) & scaf(te)), f"split {i}: train/test share a scaffold"
    assert not (scaf(tr) & scaf(va)), f"split {i}: train/valid share a scaffold"
    assert not (scaf(va) & scaf(te)), f"split {i}: valid/test share a scaffold"

    # 3. every set contains both classes (otherwise AUROC is undefined)
    for name, idx in (("train", tr), ("valid", va), ("test", te)):
        assert df["label"].iloc[sorted(idx)].nunique() == 2, f"split {i}: {name} has one class"

    pos = lambda idx: df["label"].iloc[sorted(idx)].mean()
    return {
        "sizes": (len(tr), len(va), len(te)),
        "pos": (pos(tr), pos(va), pos(te)),
        "n_test_scaffolds": len(scaf(te)),
        "test": te,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite existing split files")
    args = parser.parse_args()

    existing = [split_path(i) for i in range(N_SPLITS) if split_path(i).exists()]
    if existing and not args.force:
        sys.exit(
            f"Refusing to overwrite {len(existing)} existing split file(s) in splits/.\n"
            "They are the shared contract; results from all models depend on them.\n"
            "Re-run with --force only if you really mean to regenerate."
        )

    df = load_dataset()
    smiles = df["smiles"].tolist()
    scaffolds = [murcko_scaffold(s) for s in smiles]

    for i in range(N_SPLITS):
        seed = i
        path = save_split(scaffold_split(smiles, seed=seed), i, seed, df)
        print(f"wrote {path.relative_to(path.parents[1])}")

    print(f"\nVerifying from disk  (n={len(df)}, overall BBB+ rate {df['label'].mean():.3f})")
    print(f"{'split':>5} | {'train/valid/test':>16} | {'BBB+ rate tr/va/te':>18} | test scaffolds")
    tests = []
    for i in range(N_SPLITS):
        r = verify(i, df, scaffolds)
        tests.append(r["test"])
        s, p = r["sizes"], r["pos"]
        print(f"{i:>5} | {s[0]:>5}/{s[1]:>4}/{s[2]:>4}  | {p[0]:.2f} / {p[1]:.2f} / {p[2]:.2f}  | {r['n_test_scaffolds']}")

    overlap = max(
        len(tests[a] & tests[b]) / len(tests[a] | tests[b])
        for a in range(N_SPLITS) for b in range(a + 1, N_SPLITS)
    )
    print(f"\nmax test-set overlap between any two splits (Jaccard): {overlap:.3f}")
    print("All checks passed: disjoint, complete, zero scaffold leakage, both classes everywhere.")


if __name__ == "__main__":
    main()
