# Blood-Brain Barrier (BBB) Permeability Representation Benchmark

A leak-free benchmark of four molecular representation paradigms on blood-brain barrier permeability, using the **Therapeutics Data Commons (TDC) `BBB_Martins`** dataset (~2,000 molecules, 76% BBB+).

All six models are evaluated on **identical, version-controlled splits** with a shared metrics harness and **no hyperparameter tuning**, so the differences reflect representation quality rather than search budget.

---

## Executive summary

A common assumption in AI for drug discovery is that graph neural networks and pre-trained foundation models beat classical descriptors on ADMET endpoints. This benchmark tests that assumption under strict out-of-distribution Bemis-Murcko scaffold splits.

1. **Morgan fingerprints + Random Forest are the best or tied-best model** (0.924 AUROC on held-out scaffolds). No learned representation beats them on more than 3 of 5 splits under either protocol.
2. **Frozen ChemBERTa-2 with mean pooling matches end-to-end fine-tuning on scaffold splits** (0.922 vs 0.891) while training 148k parameters instead of 3.4M — but it is 0.020 AUROC *behind* fine-tuning on the TDC protocol, so the two are best described as comparable rather than one dominating.
3. **How you pool a frozen transformer matters as much as which model you use.** Across `[CLS]`/mean/sum/max, test AUROC spans 0.821–0.882 on TDC — wider than the gap between model families. `[CLS]` is reliably last (MLM pre-training never trains that slot to summarize a sequence; fine-tuning is what teaches it to). Which of the other three is best is **not reliably selectable**: validation ranks them differently from test on both protocols, which is the empirical case for this project's no-tuning rule.
4. **Frozen Boltz-2 trunk features rank last on both protocols** (0.864 / 0.805), losing to the fingerprint baseline on 10 of 10 splits. Linear probes localize the cause: Boltz-2 encodes molecular size as well as ChemBERTa (R² 0.87) but **loses H-bond donor counts (R² 0.518 vs Morgan's 0.813)**, lipophilicity and flexibility — the descriptors passive BBB permeation actually depends on. It clears a two-number size-only baseline by just **+0.024 AUROC** on the TDC protocol. See [Diagnostics](#diagnostics-why-boltz-2-underperforms).

---

## Benchmark results

Two evaluation protocols, each with 5 committed splits that every model loads unchanged:

- **`scaffold5` (primary)** — 5 *independent* Bemis-Murcko scaffold splits (80/10/10) over the curated 1,955-molecule set, with whole scaffold groups assigned to exactly one of train/valid/test. Each split holds out a different region of chemical space, so the std reflects generalization across chemistry, not just seed noise. Std is the sample std (`ddof=1`) over the 5 test sets.
- **`tdc` (leaderboard-comparable)** — TDC's official benchmark-group split on the **raw, uncleaned** 2,030 rows: a fixed 406-molecule test set plus 5 train/valid scaffold reshuffles (seeds 1–5). AUROC is reported by TDC's own `evaluate_many` (`ddof=0`), so it is directly comparable to the public leaderboard.

| Model / representation | Paradigm | Trainable params | `scaffold5` AUROC | `scaffold5` AUPRC | `tdc` AUROC | `tdc` AUPRC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Morgan ECFP4 (r=2, 2048-bit) + RF** | Classical descriptors | 500 trees | **0.924 ± 0.024** | **0.971 ± 0.013** | 0.865 ± 0.005 | 0.954 ± 0.003 |
| **ChemBERTa-2, frozen + mean pool** | Pre-trained transformer | 148k head | 0.922 ± 0.036 | 0.968 ± 0.016 | 0.850 ± 0.011 | 0.947 ± 0.005 |
| **ChemBERTa-2, frozen + sum pool** † | Pre-trained transformer | 148k head | 0.912 ± 0.045 | 0.962 ± 0.021 | 0.879 ± 0.011 | 0.957 ± 0.004 |
| **ChemBERTa-2, frozen + max pool** † | Pre-trained transformer | 148k head | 0.908 ± 0.051 | 0.965 ± 0.019 | **0.882 ± 0.011** | **0.963 ± 0.004** |
| **GINE GNN, from scratch** | 2D molecular graph | 33k (all) | 0.900 ± 0.048 | 0.960 ± 0.021 | 0.847 ± 0.034 | 0.952 ± 0.013 |
| **ChemBERTa-2, fine-tuned** | Pre-trained transformer | 3.4M (all) | 0.891 ± 0.052 | 0.950 ± 0.040 | 0.870 ± 0.020 | 0.958 ± 0.008 |
| **ChemBERTa-2, frozen + `[CLS]`** | Pre-trained transformer | 148k head | 0.878 ± 0.027 | 0.951 ± 0.022 | 0.821 ± 0.012 | 0.942 ± 0.005 |
| **Boltz-2 trunk `s`, frozen + mean pool** | 3D co-folding foundation model | 148k head | 0.864 ± 0.041 | 0.944 ± 0.019 | 0.805 ± 0.008 | 0.930 ± 0.004 |
| *Random baseline* | — | — | *0.500* | *0.763* | *0.500* | *0.808* |

All models use default, untuned hyperparameters. AUPRC is reported against the positive-class rate (the *random* AUPRC column above), not against 0.5.

† **The sum/max rows are diagnostics, not a claimed win.** They were added after the fact to test a mechanism (see [Diagnostics §4](#4-pooling-choice-matters-but-validation-cannot-pick-it)), and **validation AUROC does not rank the pooling choices in the same order as test AUROC on either protocol** — an honest protocol that selected pooling on validation would have picked a *worse* option both times. Max pooling's 0.882 on `tdc` is reported for transparency, not as a result to cite.

**Head-to-head vs. the fingerprint baseline**, counting per-split wins on the same test sets:

| Model | Wins, `scaffold5` | Wins, `tdc` | Mean AUROC gap, `scaffold5` | Mean AUROC gap, `tdc` |
| :--- | :---: | :---: | :---: | :---: |
| ChemBERTa-2, fine-tuned | 0/5 | 3/5 | −0.033 | +0.005 |
| GINE GNN | 1/5 | 2/5 | −0.024 | −0.018 |
| ChemBERTa-2, frozen + mean | 2/5 | 0/5 | −0.002 | −0.015 |
| ChemBERTa-2, frozen + `[CLS]` | 0/5 | 0/5 | −0.046 | −0.044 |
| Boltz-2, frozen | 0/5 | 0/5 | −0.060 | −0.060 |

**Leaderboard context.** On the `tdc` protocol these numbers are comparable to the [public TDC leaderboard](https://tdcommons.ai/benchmark/admet_group/07bbb/), where the top entries reach ~0.92 (MiniMol 0.924, MapLight 0.916) and tuned classical models sit around 0.90 (Lantern RF 0.908). Our best, 0.870, lands mid-table — expected for untuned baselines with no feature engineering or ensembling, and the gap is a tuning/feature gap rather than a protocol difference. These are local runs, not official submissions.

**The split matters as much as the model.** On `scaffold5`, a single model's test AUROC moves by up to 0.13 between splits (the GNN spans 0.830–0.964). Any claim resting on one test set is unreliable at this dataset size, which is exactly why the primary protocol uses five independent held-out scaffold regions.

---

## Why these numbers are trustworthy

### Data curation
2,030 raw TDC molecules → **1,955 canonical molecules** for `scaffold5`: counter-ions/salts stripped to the largest fragment, SMILES canonicalized, exact duplicates merged, and **11 molecules with contradictory BBB+/BBB− labels dropped** (no ground truth exists for them). The `tdc` protocol deliberately skips this cleaning so its test set matches the leaderboard byte-for-byte.

### Split contract
Splits are generated **once**, saved as integer indices in `splits/`, and committed. No model ever re-splits. Each split file embeds a dataset fingerprint (truncated SHA-256 over the ordered `(smiles, label)` rows); `load_split()` recomputes it and **raises** on any mismatch, so a change to the cleaning logic or upstream data fails loudly instead of silently pointing at the wrong molecules. Verified per split: sets disjoint, every molecule used, **zero scaffold overlap** between train/valid/test, and both classes present everywhere.

### Negative controls
A **label-permutation test** (train on shuffled labels) yields ~0.56 AUROC rather than 0.50. Traced to the Random Forest ranking by molecular size, which is a genuine BBB signal (heavy-atom count alone gives 0.69 AUROC) and not label leakage — the training labels were random. That ~0.56 is the reference floor for "structure without labels" on these splits.

### Identical measurement
`src/bbb/metrics.py` is shared by every model. Every neural model uses the same early-stopping rule (best epoch by **validation** AUROC), and **the test set is scored exactly once**, after training.

---

## Representations evaluated

- **Morgan fingerprints** (`featurizers/morgan.py`) — 2048-bit radius-2 circular fingerprints (ECFP4-equivalent) + a 500-tree scikit-learn Random Forest.
- **GINE graph neural network** (`models/gnn.py`) — 3-layer Graph Isomorphism Network with edge features, hidden dim 64, BatchNorm, residual connections, global mean pooling; **33,217 parameters**. Graphs carry 32 atom features and 7 bond features (`featurizers/graph.py`).
- **ChemBERTa-2** (`models/chemberta.py`, `featurizers/chemberta_embed.py`) — `DeepChem/ChemBERTa-77M-MLM`, pinned to revision `ed8a5374`, evaluated three ways: full fine-tuning (AdamW, lr 2e-5), and frozen feature extraction with an MLP head comparing **padding-masked mean pooling** against **`[CLS]` extraction**.
- **Boltz-2** (`scripts/run_boltz_frozen.py`) — biomolecular co-folding foundation model v2.2.1, run **ligand-only** on an NVIDIA L40S cluster to dump the **pre-diffusion trunk representation `s`** (384 numbers per heavy atom), mean-pooled over atoms. A pilot confirmed `s` is **bit-identical** at 10 vs. 200 diffusion sampling steps, confirming it is computed before any 3D coordinates are generated — so the 45-minute full run used the cheap setting.

The frozen models share one head (`models/mlp_head.py`: standardize → dropout → `d→d` + tanh → dropout → `d→1`), so Boltz-2 and ChemBERTa differ only in where their 384 numbers come from.

---

## Key findings

1. **Simplicity is hard to beat at this scale.** With ~1,500 training molecules, Morgan fingerprints remain a top-tier representation. Learned representations have more capacity to fit, but also more capacity to fit the training scaffolds specifically.
2. **Pooling choice dominates the frozen/fine-tuned distinction.** Swapping `[CLS]` for mean pooling buys +0.044 AUROC — larger than the entire gap between frozen and fine-tuned ChemBERTa, and the full `[CLS]`→max span on TDC is +0.061. MLM pre-training gives the `[CLS]` slot no sequence-level training signal; fine-tuning is what makes it a usable summary vector. A cheap representation decision outweighed an expensive training decision here.
3. **A foundation model's pre-training objective matters more than its size.** Boltz-2 is orders of magnitude larger than everything else here and still ranks last. Its objective (placing atoms in 3D, in a protein's presence) is further from passive permeability than "predict masked SMILES tokens" is — and the probe shows this concretely, as a selective loss of H-bond donor and lipophilicity information rather than a general lack of chemical knowledge.
4. **Diagnose before concluding.** The intuitive explanations for Boltz-2's result — an overfitting head, or features that only encode size — are both testable in minutes and both turned out wrong (logistic regression scores no better; Boltz-2 does clear the size baseline on `scaffold5`). The probe found the actual gap, and a pooling control confirmed the mechanism behind it.
5. **At ~2,000 molecules, a held-out set of ~200 cannot adjudicate small differences.** Validation picked the wrong pooling on both protocols. Reporting a spread with its selection caveat is more useful than naming a winner that would not survive an honest protocol.

---

## Caveats

Stated plainly, because they bound what the results support:

- **Untuned by design.** Every model uses library defaults with no search. These are floors, not best-achievable scores. Any future tuning should use one shared budget per model, selected on validation AUROC only.
- **Boltz-2 could not embed 2 of 1,966 molecules** — one exceeds its 4-character atom-naming limit (229 atoms including hydrogens), and RDKit failed to generate a 3D conformer for the other. Each split imputes them with **that split's training-set mean vector** (no valid/test information). At most 1 appears in any test set.
- **Boltz-2 saw salt-stripped SMILES** (largest fragment), affecting 168 raw TDC rows; the other models saw TDC's raw SMILES. Recorded in the results JSON.
- **Frozen-model inputs are standardized** with training-set statistics, because Boltz-2's activations are ~70× ChemBERTa's scale (values to ~950) and would otherwise saturate `tanh`. Applied to *all* frozen models for parity; the frozen-ChemBERTa numbers here are from that re-run.
- **Hardware.** Boltz-2 embeddings were computed on an NVIDIA L40S (Slurm cluster); **all model training and scoring ran on one machine**, so the scores are mutually comparable. Re-running on different hardware shifts third-decimal digits (verified: identical test AUROCs, ±0.001 on some valid/AUPRC values).
- **The sum/max pooling runs are post-hoc diagnostics.** They were added while investigating the Boltz-2 result, not planned up front, and validation does not select among poolings reliably (see Diagnostics §4). They are reported in full rather than cherry-picked, and the headline comparison does not rest on them.
- **Trained weights are not checkpointed.** Runs are seeded and reproducible, but no model is saved for inference on new molecules.

---

## Diagnostics: why Boltz-2 underperforms

Rather than speculate, three controls were run on the same splits. Two candidate explanations are ruled out, and one is supported with a specific mechanism.

### 1. Linear probe — Boltz-2 encodes size, but loses the properties that govern permeation

A linear probe (`RidgeCV`, fit on train, R² on held-out test) asks which physicochemical properties are *linearly recoverable* from each frozen representation. If a property cannot be read out linearly, no small head can use it.

| Property | Morgan | ChemBERTa (mean) | Boltz-2 `s` |
| :--- | :---: | :---: | :---: |
| Molecular weight | 0.786 ± 0.061 | 0.879 ± 0.011 | 0.870 ± 0.034 |
| Heavy-atom count | 0.798 ± 0.057 | 0.882 ± 0.010 | 0.864 ± 0.037 |
| TPSA | 0.827 ± 0.061 | 0.892 ± 0.035 | 0.829 ± 0.006 |
| H-bond acceptors | 0.828 ± 0.053 | 0.878 ± 0.032 | 0.826 ± 0.026 |
| **H-bond donors** | **0.813 ± 0.086** | 0.768 ± 0.096 | **0.518 ± 0.128** |
| **cLogP** | 0.752 ± 0.079 | **0.842 ± 0.035** | **0.699 ± 0.075** |
| **Rotatable bonds** | 0.750 ± 0.044 | **0.792 ± 0.033** | **0.563 ± 0.076** |

Boltz-2 matches ChemBERTa on **size and polar surface area** but is far weaker on **H-bond donors (0.518 vs 0.813), lipophilicity and flexibility** — precisely the descriptors that classical CNS-penetration rules are built from. The representation is not uninformative; it is missing the specific information this endpoint needs.

A structural detail is consistent with the H-bond donor gap: the trunk emits **one token per heavy atom** (verified — 19 tokens for propranolol's 19 heavy atoms), so hydrogens are never tokens and donor counts must be inferred rather than read off. Suggestive, not proof.

### 2. The head is not the bottleneck

L2 logistic regression (385 parameters, `C` chosen by CV inside train) on the same vectors scores **at or below** the 148k-parameter MLP head everywhere, so head capacity is not what limits the frozen results:

| Representation | LogReg `scaffold5` | MLP head `scaffold5` | LogReg `tdc` | MLP head `tdc` |
| :--- | :---: | :---: | :---: | :---: |
| Morgan | 0.922 ± 0.020 | 0.924 (RF) | 0.863 ± 0.004 | 0.865 (RF) |
| ChemBERTa (mean) | 0.892 ± 0.047 | 0.922 | 0.834 ± 0.020 | 0.850 |
| Boltz-2 `s` | 0.843 ± 0.042 | 0.864 | 0.770 ± 0.004 | 0.805 |

### 3. Boltz-2 barely clears a two-number size baseline

A size-only baseline (heavy-atom count + molecular weight, two features) reaches **0.739 ± 0.073** on `scaffold5` and **0.781 ± 0.001** on `tdc`. Against that floor:

- **Morgan** clears it by **+0.183 / +0.082**.
- **Boltz-2** clears it by **+0.125 / +0.024** — and under logistic regression on `tdc`, Boltz-2's 384 numbers (0.770) score *below* the two-number baseline (0.781).

So on the leaderboard protocol, a 384-dimensional representation from a large structural foundation model carries little BBB-relevant signal beyond molecular size.

### 4. Pooling choice matters, but validation cannot pick it

The H-bond donor gap suggests a mechanism: a **mean** divides by token count, so extensive ("how many") properties survive only as fractions, while a **sum** keeps magnitude. Testing that on ChemBERTa, where per-token vectors are available locally, confirms it — sum pooling improves recovery of every count-like property:

| Property | ChemBERTa (mean) | ChemBERTa (sum) |
| :--- | :---: | :---: |
| H-bond donors | 0.768 ± 0.096 | **0.896 ± 0.046** |
| Heavy-atom count | 0.882 ± 0.010 | **0.996 ± 0.001** |
| Molecular weight | 0.879 ± 0.011 | **0.985 ± 0.001** |
| Rotatable bonds | 0.792 ± 0.033 | **0.893 ± 0.019** |

Part of this is near-trivial — the norm of a sum scales with token count, so size is almost readable off the magnitude — but the donor-count gain exceeds what size alone explains. **This predicts that sum-pooling Boltz-2's per-atom `s` would recover more of the missing properties**, which is the next experiment to run.

Downstream, pooling moves BBB AUROC as much as the model family does (`tdc`: `[CLS]` 0.821 → mean 0.850 → sum 0.879 → max 0.882). **But the choice cannot be made honestly here.** Validation and test disagree on the ranking:

| Protocol | Validation ranking (what you could legitimately select on) | Test ranking (ground truth) |
| :--- | :--- | :--- |
| `scaffold5` | **sum** 0.925 > max 0.920 > mean 0.917 > cls 0.888 | **mean** 0.922 > sum 0.912 > max 0.908 > cls 0.878 |
| `tdc` | **mean** 0.939 ≈ sum 0.939 > max 0.925 > cls 0.922 | **max** 0.882 > sum 0.879 > mean 0.850 > cls 0.821 |

Selecting on validation picks `sum` on `scaffold5` (test 0.912, worse than mean's 0.922) and `mean` on `tdc` (test 0.850, the worst of the three non-`[CLS]` options). With ~200 validation molecules, differences of 0.01–0.03 are not resolvable — which is the empirical justification for this project's no-tuning rule. The reportable finding is the **spread** (0.821–0.882 on `tdc`), not a winner.

The one robust conclusion survives: **`[CLS]` is last on both protocols, by both validation and test.**

### Still untested

| Hypothesis | Test | Status |
| :--- | :--- | :--- |
| Mean pooling destroys count-like information | Sum/max pooling | **Confirmed on ChemBERTa** (above); Boltz-2 re-pooling pending — needs the per-atom files on the cluster |
| No protein context — the trunk always trained with a macromolecular partner present | Re-embed alongside a fixed dummy protein and compare | Not run — needs GPU time |
| The pair representation `z` carries what `s` loses | Pool `z` and probe it | Not run — needs the per-atom files |

Reproduce the diagnostics:

```bash
python scripts/probe_embeddings.py                       # linear probe (~15 min)
python scripts/diagnose_heads.py --protocol tdc          # logistic regression + size baseline
python scripts/run_chemberta_frozen.py --pooling sum     # also: cls, mean, max
```

---

## Repository structure

```
bbb-permeability-benchmark/
├── splits/                    # Committed scaffold5 + tdc split indices — the contract
├── src/bbb/
│   ├── data.py                # Loading, salt stripping, dedup, dataset fingerprint
│   ├── splits.py              # Bemis-Murcko splitting, TDC splits, load_protocol()
│   ├── metrics.py             # Shared evaluation harness (AUROC, AUPRC)
│   ├── featurizers/
│   │   ├── morgan.py          # 2048-bit ECFP4
│   │   ├── graph.py           # PyG graphs (32 atom / 7 bond features)
│   │   └── chemberta_embed.py # Frozen extraction + pooling strategies
│   └── models/
│       ├── fingerprint_rf.py  # Random Forest baseline
│       ├── gnn.py             # 3-layer GINE + training loop
│       ├── chemberta.py       # End-to-end fine-tuning
│       └── mlp_head.py        # Standardized head for frozen embeddings
├── scripts/                   # One runner per model; all take --protocol
├── slurm/                     # HPC job scripts (L40S GPU cluster)
└── results/                   # Per-split metrics as {model}_{protocol}.json
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Quickstart

### Environment

```bash
git clone https://github.com/James-Zhu1/bbb-permeability-benchmark
cd bbb-permeability-benchmark

# venv (Python 3.11; 3.13 breaks PyTDC and the Boltz-2 deps)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# or conda, e.g. on a cluster
conda env create -f environment.yml && conda activate bbb
```

The dataset is not committed; TDC downloads it to `data/raw/` on first run. The committed splits' fingerprint check confirms you got identical data.

### Reproduce the splits (optional — they are committed)

```bash
python scripts/make_splits.py                  # scaffold5 -> splits/
python scripts/make_splits.py --protocol tdc   # TDC       -> splits/tdc/
```
Both refuse to overwrite existing split files without `--force`, and re-verify every invariant from disk.

### Run the benchmarks

Every runner takes `--protocol scaffold5` (default) or `--protocol tdc`:

```bash
python scripts/run_fingerprint_rf.py                      #  ~6 s
python scripts/run_gnn.py                                 # ~10 s/split
python scripts/run_chemberta.py                           # ~2 min/split
python scripts/run_chemberta_frozen.py --pooling mean     # ~20 s  (also: --pooling cls)
```

### Boltz-2 pipeline (needs a GPU)

```bash
python scripts/boltz_export_inputs.py          # bbb env: 1,966 ligand-only YAMLs + row index
sbatch slurm/boltz_embed.sbatch                # boltz env: ~45 min on 1x L40S
python scripts/boltz_pool_embeddings.py        # bbb env: s -> mean over atoms
python scripts/run_boltz_frozen.py             # bbb env: head + 5 splits
```

Boltz-2 lives in a **separate conda env** (`environment-boltz.yml`): it pins `scikit-learn==1.6.1` and `rdkit>=2024.3.2`, which conflict with the main env, and a different RDKit could change canonical SMILES and invalidate the split fingerprints. The `bbb` env therefore owns the dataset and exports the exact SMILES Boltz sees; the `boltz` env only turns inputs into embeddings.

### Summarize results

```bash
python -c "
import json, glob, pathlib
for p in sorted(glob.glob('results/*.json')):
    d = json.load(open(p)); t = d['summary']['test']
    print(f\"{pathlib.Path(p).stem:<38} AUROC {t['auroc']['mean']:.3f} ± {t['auroc']['std']:.3f}\"
          f\"   AUPRC {t['auprc']['mean']:.3f} ± {t['auprc']['std']:.3f}\")
"
```

Each JSON also carries the per-split metrics, the full training history, the exact config, and the dataset fingerprint the run was validated against.
