# Blood-Brain Barrier (BBB) Permeability Representation Benchmark

[[https://img.shields.io/badge/python-3.11-blue.svg]](https://www.python.org/)
[[https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg]](https://pytorch.org/)
[[https://img.shields.io/badge/PyG-GNN-3C2179.svg]](https://pyg.org/)
[[https://img.shields.io/badge/License-MIT-yellow.svg]](https://opensource.org/licenses/MIT)

A rigorous, leak-free benchmarking framework evaluating four distinct molecular representation paradigms on Blood-Brain Barrier (BBB) permeability prediction using the **Therapeutics Data Commons (TDC) `BBB_Martins`** dataset (~2,000 molecules).

---

## Executive Summary

A pervasive assumption in modern AI for drug discovery is that deep graph neural networks and large pre-trained foundation models systematically outperform classical descriptors on tabular ADMET endpoints. 

This benchmark tests that assumption under **strict out-of-distribution (OOD) scaffold splits**. Under an untuned, fair-comparison protocol:
1. **Classical Morgan Fingerprints (ECFP4) + Random Forest** remain competitive with or outperform all deep learning approaches (**0.924 AUROC** on held-out scaffolds).
2. **ChemBERTa-2 frozen representations with mean pooling** match end-to-end fine-tuning while requiring a fraction of the compute, whereas **[CLS] token pooling degrades performance** because masked language modeling (MLM) does not explicitly adapt the [CLS] embedding.
3. **Boltz-2 3D structural embeddings** underperform 2D topological baselines (**0.864 AUROC**), indicating that general biomolecular folding representations without target protein context introduce conformational noise on small-molecule passive diffusion tasks.

---

## Benchmark Results

All models were evaluated across two distinct evaluation schemes:
- **`scaffold5` (Primary Protocol)**: 5 independent Bemis-Murcko scaffold splits (80/10/10), ensuring complete scaffold group isolation with zero leakage. Reports mean ± sample standard deviation across independent test partitions.
- **`tdc` (Leaderboard Protocol)**: The official Therapeutics Data Commons benchmark split (fixed 406-molecule test set + 5 train/val scaffold reshuffles), evaluated via TDC's official `evaluate_many` metric (`ddof=0`).

| Model / Representation | Paradigm | Parameters | `scaffold5` Test AUROC | `scaffold5` Test AUPRC | `tdc` Test AUROC | `tdc` Test AUPRC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Morgan ECFP4 (r=2, 2048-bit) + RF** | Classical Descriptors | 500 trees | **0.924 ± 0.024** | **0.971 ± 0.013** | **0.865 ± 0.005** | **0.954 ± 0.003** |
| **ChemBERTa-2 (Frozen Mean Pool + MLP)** | Pre-trained Transformer | 3.4M (frozen) | **0.922 ± 0.036** | **0.968 ± 0.016** | 0.850 ± 0.011 | 0.947 ± 0.005 |
| **GINE GNN (from scratch)** | 2D Molecular Graph | 33k | 0.900 ± 0.048 | 0.960 ± 0.021 | 0.847 ± 0.034 | 0.952 ± 0.013 |
| **ChemBERTa-2 (Fine-Tuned)** | Pre-trained Transformer | 3.4M (trained) | 0.891 ± 0.052 | 0.950 ± 0.040 | **0.870 ± 0.020** | **0.958 ± 0.008** |
| **ChemBERTa-2 (Frozen [CLS] + MLP)** | Pre-trained Transformer | 3.4M (frozen) | 0.878 ± 0.027 | 0.951 ± 0.022 | 0.821 ± 0.012 | 0.942 ± 0.005 |
| **Boltz-2 Trunk `s` (Frozen Mean + MLP)** | 3D Structural Foundation | ~400M (frozen) | 0.864 ± 0.041 | 0.944 ± 0.019 | 0.805 ± 0.008 | 0.930 ± 0.004 |
| *Random Baseline* | — | — | *0.500* | *0.763* | *0.500* | *0.808* |

*Note: All models use default, untuned hyperparameters so performance differences strictly reflect representation quality rather than search budget.*

---

## Architectural & Methodological Highlights

### 1. Data Integrity & Leakage Prevention
- **Dataset Curation**: Started with 2,030 raw molecules from TDC `BBB_Martins`. Stripped counterions/salts, merged duplicates, and dropped 11 contradictory label conflicts, yielding a 1,955 canonical molecule dataset.
- **Split Contract**: Splits are generated once via Bemis-Murcko scaffold decomposition, saved as immutable indices in `splits/`, and version-controlled. Each split file embeds a SHA-256 dataset fingerprint that halts execution upon any data mismatch.
- **Diagnostic Controls**: A label-shuffled permutation test yielded ~0.56 AUROC (rather than 0.50), traced via controls to the Random Forest capturing heavy-atom count / molecular weight as an orthogonal signal (size alone yielded 0.69 AUROC).

### 2. Evaluated Representations
- **Morgan Fingerprints (`morgan.py`)**: 2048-bit radius-2 circular fingerprints (ECFP4 equivalent) paired with a 500-estimator scikit-learn Random Forest.
- **GINE Graph Neural Network (`gnn.py`)**: 3-layer Graph Isomorphism Network with Edge features (GINE), hidden dimension 64, batch normalization, residual connections, and global mean pooling (33,409 parameters). Featurized with 32 atom features and 7 bond features in PyTorch Geometric.
- **ChemBERTa-2 (`chemberta.py`, `chemberta_embed.py`)**: `DeepChem/ChemBERTa-77M-MLM` evaluated under full fine-tuning (AdamW, lr=2e-5) vs. frozen feature extraction with an MLP classification head comparing padding-masked mean pooling against [CLS] token extraction.
- **Boltz-2 (`run_boltz_frozen.py`)**: State-of-the-art biomolecular structure foundation model (v2.2.1). Inferred on an HPC cluster with NVIDIA L40S GPUs to extract pre-diffusion trunk representations `s` (384-dimensional per-atom vectors). Handled non-embeddable edge cases (4-character PDB ligand naming limits and conformer generation failures) via split-isolated training mean imputation. Discovered and resolved a 70x activation scale shift via train-fit z-score standardization.

---

## Key Takeaways

1. **Simplicity Often Beats Scale for ADMET**: For datasets with ~2,000 compounds, classical Morgan fingerprints remain exceptionally hard to beat. Deep representations risk overfitting or capturing orthogonal features that fail to generalize across unfamiliar scaffolds.
2. **Mean Pooling vs. [CLS] in Chemistry Transformers**: Chemical language models pre-trained solely on Masked Language Modeling (MLM) do not update the `[CLS]` token with sequence-level information. Global mean pooling across token representations yields a +4.4% AUROC boost over `[CLS]`.
3. **Foundation Model Misalignment**: Biomolecular 3D foundation models trained for complex co-folding do not automatically transfer to small-molecule property prediction. Without target macromolecular binding partners, conformational embeddings act as noisy representations for passive membrane permeability.

---

## Repository Structure

```
bbb-permeability-benchmark/
├── splits/                   # Versioned scaffold5 and tdc split indices
├── src/bbb/
│   ├── data.py               # Data loading, salt stripping, and deduplication
│   ├── splits.py             # Bemis-Murcko splitting and protocol loaders
│   ├── metrics.py            # Unified evaluation harness (AUROC, AUPRC)
│   ├── featurizers/
│   │   ├── morgan.py         # 2048-bit ECFP4 generator
│   │   ├── graph.py          # PyG graph constructor (32 atom / 7 bond feats)
│   │   └── chemberta_embed.py# Token extraction & pooling strategies
│   └── models/
│       ├── fingerprint_rf.py # Random Forest baseline
│       ├── gnn.py            # 3-layer GINE architecture
│       ├── chemberta.py      # End-to-end transformer fine-tuner
│       └── mlp_head.py       # Standardized MLP probe for frozen embeddings
├── scripts/                  # Standalone execution runners per model & protocol
├── slurm/                    # HPC job submission scripts for L40S GPU cluster
└── results/                  # Raw benchmark metric JSONs
```

---

## Quickstart

### 1. Environment Setup
```bash
# Clone repository
git clone [https://github.com/](https://github.com/)<your-username>/bbb-permeability-benchmark.git
cd bbb-permeability-benchmark

# Create virtual environment (Python 3.11 recommended)
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

### 2. Running Benchmarks
All scripts accept `--protocol scaffold5` (default) or `--protocol tdc`:

```bash
# Morgan Fingerprints + Random Forest
python scripts/run_fingerprint_rf.py --protocol scaffold5

# GINE Graph Neural Network (from scratch)
python scripts/run_gnn.py --protocol scaffold5

# ChemBERTa-2 Fine-Tuning
python scripts/run_chemberta.py --protocol scaffold5

# ChemBERTa-2 Frozen Embeddings (mean pooling)
python scripts/run_chemberta_frozen.py --pooling mean --protocol scaffold5
```

### 3. Reproducing Results
All metrics output directly to `results/{model}_{protocol}.json`. To regenerate the comparative results table:
```bash
python -c "
import json, glob
for path in sorted(glob.glob('results/*_scaffold5.json')):
    data = json.load(open(path))
    print(f\"{data['model']}: AUROC = {data['summary']['test']['auroc']['mean']:.3f} ± {data['summary']['test']['auroc']['std']:.3f}\")
"
```
