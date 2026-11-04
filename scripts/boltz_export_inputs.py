"""Step 4a (bbb env): write Boltz-2 input files for every molecule.

    python scripts/boltz_export_inputs.py               # all molecules, both protocols
    python scripts/boltz_export_inputs.py --limit 5     # pilot: first 5 unique molecules

Boltz runs in its own env and must never re-derive our dataset (different
RDKit -> different canonical SMILES -> broken splits). So THIS script, in the
`bbb` env, decides exactly which SMILES Boltz sees and records which dataset
row each one belongs to:

  data/boltz/<run>/inputs/<mol_id>.yaml     one ligand-only Boltz input per UNIQUE molecule
  data/boltz/<run>/index_<protocol>.csv     row -> mol_id, in load_protocol() row order

Each molecule is reduced to its largest fragment (drops counter-ions such as
".[Cl]"; Boltz would otherwise treat the ion as a second ligand). scaffold5
SMILES are already stripped, so this only changes some raw TDC rows.
"""

from __future__ import annotations

import argparse
import hashlib

import pandas as pd
from rdkit import Chem

from bbb.data import REPO_ROOT, standardize_smiles
from bbb.splits import PROTOCOLS, load_protocol


def mol_id(smiles: str) -> str:
    """Stable, filename-safe ID derived from the SMILES itself."""
    return "m" + hashlib.sha1(smiles.encode()).hexdigest()[:12]


def boltz_yaml(smiles: str) -> str:
    """Ligand-only Boltz input: one ligand chain 'A', no protein."""
    return (
        "version: 1\n"
        "sequences:\n"
        "  - ligand:\n"
        "      id: A\n"
        f"      smiles: '{smiles}'\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="pilot run: only the first N unique molecules")
    args = parser.parse_args()

    run = "pilot" if args.limit else "full"
    out_dir = REPO_ROOT / "data" / "boltz" / run
    in_dir = out_dir / "inputs"
    in_dir.mkdir(parents=True, exist_ok=True)

    unique: dict[str, str] = {}  # mol_id -> boltz SMILES
    for protocol in PROTOCOLS:
        df, _ = load_protocol(protocol)
        boltz_smiles = df["smiles"].map(standardize_smiles)
        index = pd.DataFrame({
            "row": range(len(df)),
            "smiles": df["smiles"],
            "boltz_smiles": boltz_smiles,
            "mol_id": boltz_smiles.map(mol_id),
        })
        changed = (index["smiles"] != index["boltz_smiles"]).sum()
        for mid, s in zip(index["mol_id"], index["boltz_smiles"]):
            unique.setdefault(mid, s)
        if args.limit is None:
            index.to_csv(out_dir / f"index_{protocol}.csv", index=False)
        print(f"[{protocol}] {len(df)} rows, {index['mol_id'].nunique()} unique molecules, "
              f"{changed} SMILES changed by fragment-stripping/canonicalization")

    items = list(unique.items())
    print(f"unique molecules across both protocols: {len(items)}")
    if args.limit:
        # Pilot: spread over molecule SIZE (runtime/memory grow with atoms):
        # pick the molecules at evenly spaced size quantiles, smallest..largest.
        items.sort(key=lambda kv: (Chem.MolFromSmiles(kv[1]).GetNumHeavyAtoms(), kv[0]))
        picks = [round(q * (len(items) - 1) / (args.limit - 1)) for q in range(args.limit)]
        items = [items[i] for i in picks]
        for mid, s in items:
            print(f"  pilot {mid}  {Chem.MolFromSmiles(s).GetNumHeavyAtoms():>3} heavy atoms  {s[:60]}")
    for mid, s in items:
        (in_dir / f"{mid}.yaml").write_text(boltz_yaml(s))
    pd.DataFrame(items, columns=["mol_id", "boltz_smiles"]).to_csv(out_dir / "molecules.csv", index=False)
    print(f"wrote {len(items)} Boltz inputs to {in_dir.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
