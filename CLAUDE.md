# BBB Permeability Benchmark

## Goal
Benchmark four molecular representations on blood-brain-barrier permeability
prediction, all evaluated on the SAME scaffold splits so the comparison is fair.

Representations:
1. Morgan fingerprints + RandomForest   (baseline — current step)
2. GNN from scratch
3. ChemBERTa fine-tune
4. Frozen Boltz-2 embeddings + simple head

Dataset: TDC `BBB_Martins` (binary BBB+/BBB-, ~2000 molecules, ~75% positive → imbalanced).

## Working style (important)
James is building this to learn. Write COMPLETE, working code for him, but ONE
piece at a time, and explain what each part does and why. Do not dump the whole
project at once. Go step by step; let him read and run each piece before the next.

## Key design decisions
- SPLITS: 5 independent Bemis-Murcko scaffold splits. Generate once with seeds
  0-4, save to `splits/split_{i}.json` as integer indices into the canonical
  dataframe, and COMMIT them. Every model loads the identical files; no model
  ever re-splits. Rationale: the scientific question is representation
  generalization to unseen scaffolds, so independent held-out scaffold regions
  give an honest mean +/- std. (TDC benchmark-group convention — fixed test set
  + 5 train/val reshuffles — was considered and rejected as the primary protocol
  because it only measures seed noise on a single test set.)
- SECONDARY PROTOCOL `tdc` (leaderboard-comparable): TDC admet_group data, RAW
  (2030 rows, no cleaning, so the test set matches the leaderboard exactly);
  fixed 406-mol test set + TDC scaffold train/valid with seeds 1-5 (70/10/20).
  Saved as `splits/tdc/split_{0..4}.json` (split i = seed i+1). Every runner
  takes `--protocol scaffold5|tdc`; `bbb.splits.load_protocol()` pairs data
  with splits. For tdc, quote TDC's official evaluator (`evaluate_many`, ddof=0 std).
- SANITY CHECK: on at least one split, Morgan+RF AUROC should land ~0.88-0.91
  (published ballpark). If it does, the harness is correct.
- SHARED INVARIANTS: `splits/` is committed and loaded by all models;
  `src/bbb/metrics.py` is shared so every model is measured identically.
- Split fractions 80/10/10. Leakage rule: a whole scaffold group goes ENTIRELY
  into one of train/valid/test — molecules sharing a scaffold are never split.
- Primary metric AUROC (data imbalanced); also report AUPRC.
- NO HYPERPARAMETER TUNING (decided after Step 2): every model uses sensible
  default settings, no search, so the comparison is about representations, not
  tuning effort. Write-up must state "default hyperparameters, untuned". If
  tuning is added later: one shared tune.py, equal budget for every model,
  selection on VALID AUROC only (mean over 5 splits), test touched once.

## Environment
- Python 3.11 venv at `.venv/` (built from Homebrew `/opt/homebrew/bin/python3.11`;
  3.13 causes friction with PyTDC / Torch / PyG / Boltz-2). Package installed
  editable: `pip install -r requirements.txt && pip install -e .`
- PyTDC needs `setuptools<81` (imports pkg_resources) — pinned in requirements.txt.
- Steps 1-3 run on the Mac (CPU). Results in results/ come from the Mac.

### Hopper HPC (GPU work, Step 4+)
- `ssh hopper` (jzii2024@hopper.mckenna.edu). Password login only; James opens a
  shared SSH connection (ControlMaster) in his own terminal; if it expires, ask
  him to log in again. Repo at /hopper/home/jzii2024/bbb-permeability-benchmark.
- RULES: modify files only inside /hopper/home/jzii2024. Slurm partition `main`
  for ALL jobs (never `debug`). Nodes: 4x NVIDIA L40S (46 GB) each.
- Standard conda: `conda env create -f environment.yml` then `conda activate bbb`
  (~/miniconda3/envs/bbb). conda is conda-forge only (~/.condarc; defaults removed
  from ~/miniconda3/.condarc) — do NOT `conda tos accept` (James's legal call).
- Sync: edit/commit/push on the Mac, `git pull` on Hopper (Hopper can't push).
- Verified: slurm/check_env.sbatch reproduces Morgan+RF 0.924 ± 0.024 on Hopper
  (identical test AUROCs; some valid/AUPRC differ in 3rd decimal: CPU/OS).
- Boltz-2 gets its OWN env (`boltz`): it pins scikit-learn==1.6.1, rdkit>=2024.3.2,
  which clash with `bbb`. A newer RDKit could change canonical SMILES and break
  the dataset fingerprint, so Boltz never re-derives the dataset: `bbb` exports
  SMILES -> `boltz` writes embeddings -> `bbb` pools + trains the head.

## Repo structure
```
bbb-permeability-benchmark/
├── requirements.txt / environment.yml / pyproject.toml
├── data/raw/              # downloaded dataset (gitignored)
├── splits/               # scaffold5 splits (+ splits/tdc/) — COMMITTED. the contract.
├── src/bbb/
│   ├── data.py           # dataset -> canonical dataframe [smiles, label]; TDC loader
│   ├── splits.py         # scaffold split, tdc split, load_protocol()
│   ├── metrics.py        # auroc/auprc — shared by ALL models
│   ├── featurizers/      # morgan.py, graph.py, chemberta_embed.py
│   └── models/           # fingerprint_rf.py, gnn.py, chemberta.py, mlp_head.py
├── scripts/              # make_splits.py, run_*.py (all take --protocol)
├── slurm/                # Hopper job scripts (logs/ gitignored)
└── results/              # {model}_{protocol}.json
```

## Progress
- [x] Repo + git + venv scaffolded
- [x] Step 1A: data.py + splits.py + scripts/make_splits.py. Verified: 1955 mols
      (2030 raw; salts stripped, 11 label-conflict mols dropped, dups merged),
      1564/195/196 per split, disjoint, zero scaffold leakage, both classes in all sets.
      Split files carry a dataset fingerprint; load_split() refuses on mismatch.
- [x] Step 1B: featurizers/morgan.py + metrics.py + models/fingerprint_rf.py +
      scripts/run_fingerprint_rf.py. Morgan r2/2048 + RF(500, untuned):
      test AUROC 0.924 ± 0.024 (split 0 = 0.900, in sanity range), AUPRC 0.971 ± 0.013
      (random 0.763). Shuffled-label control: AUROC ~0.56 (not 0.50) — traced to
      RF scores tracking molecule size (size alone AUROC 0.69), not label leakage.
- [x] TDC protocol added. Morgan+RF on tdc: AUROC 0.865 ± 0.005 (TDC evaluator),
      ~rank 18/26 on leaderboard; TDC's test set is harder than any of our 5.
- [ ] Step 2: GNN from scratch
  - [x] 2A: torch + torch_geometric installed (Apple M5; CPU fine for small graphs).
        featurizers/graph.py: smiles -> PyG Data, 32 atom feats, 7 bond feats.
  - [x] 2B: models/gnn.py (GINE, 3 layers, hidden 64, residual+BatchNorm, mean pool,
        33k params; Adam 1e-3, early stopping on valid AUROC, patience 30) +
        scripts/run_gnn.py. Reproducible (seeded, CPU). ~10s/split.
        scaffold5 test AUROC 0.900 ± 0.048; tdc 0.847 ± 0.034 (TDC evaluator)
        — below RF (0.924 / 0.865), in line with leaderboard from-scratch GNNs
        (GCN 0.842, AttentiveFP 0.855). Known issue: tdc split 2 picks epoch 3
        (valid AUROC erratic after) -> test 0.779. Untuned.
- [x] Step 3: ChemBERTa fine-tune. models/chemberta.py + scripts/run_chemberta.py.
      DeepChem/ChemBERTa-77M-MLM pinned @ed8a5374 (weights = pytorch_model.bin,
      loaded weights_only; safetensors exists only on unmerged refs/pr/2). 3.4M params,
      standard RoBERTa head, AdamW 2e-5, wd 0.01, batch 32, patience 10. CPU (beat MPS).
      Reproducible (split 0 identical across runs). ~2 min/split.
      scaffold5 test AUROC 0.891 ± 0.052 (loses to RF on 5/5 splits; split 0: valid
      0.939 but test 0.815). tdc 0.870 ± 0.020 (TDC evaluator) — best of 3 on tdc but
      beats RF on only 3/5 seeds; within noise. Untuned.
  - [x] 3b: FROZEN ChemBERTa + trained head (body never updated; run locally on Mac).
        featurizers/chemberta_embed.py (pooling cls|mean, padding-masked mean),
        models/mlp_head.py (same head design as fine-tuned: dropout .144 -> d->d tanh
        -> d->1; Adam 1e-3, patience 30, max 200 — reusable for Boltz-2 vectors),
        scripts/run_chemberta_frozen.py --pooling cls|mean.
        scaffold5: frozen-mean 0.906 ± 0.050, frozen-CLS 0.880 ± 0.027 (fine-tuned 0.891).
        tdc: frozen-mean 0.861 ± 0.010, frozen-CLS 0.809 ± 0.006 (fine-tuned 0.870).
        Takeaways: frozen [CLS] is clearly worst (loses to fine-tuned 5/5 on tdc) —
        pre-training never trained that slot. Frozen-mean ≈ fine-tuned (wins 3/5, 2/5):
        fine-tuning the body adds ~nothing over mean-pooled frozen features. Nothing
        beats Morgan+RF clearly. Note: tdc frozen-mean split 1 hit the 200-epoch cap.
- [ ] Step 4: frozen Boltz-2 embeddings
