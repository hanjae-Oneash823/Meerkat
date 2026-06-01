# Meerkat

# Meerkat — CYP3A4 Inhibition Predictor

A QSAR pipeline for predicting CYP3A4 enzyme inhibition (%) from molecular SMILES strings, built for the GISTomics AI-Driven Drug Discovery study group (Spring 2026).

CYP3A4 is responsible for metabolizing ~50% of all clinical drugs. Predicting its inhibition is a critical step in early-stage ADMET assessment, directly informing drug safety and candidate prioritization.

---

## Competition

| | |
|---|---|
| **Task** | Regression — predict % inhibition of CYP3A4 |
| **Input** | Canonical SMILES strings |
| **Training set** | ~1,681 molecules |
| **Test set** | 100 molecules |
| **Metric** | `Score = 0.5 × (1 − NRMSE) + 0.5 × Pearson r` |
| **Result** | 🥉 38th place |

The scoring metric penalizes both absolute error (NRMSE) and failure to capture inhibition rank order (Pearson r), requiring the model to be accurate and well-calibrated simultaneously.

---

## Approach

### 1. Molecular Featurization

Each SMILES string is converted into a ~3,200-dimensional feature vector combining four complementary representations:

| Feature | Dim | Description |
|---|---|---|
| Morgan Fingerprints (ECFP4) | 2,048 | Circular fingerprints encoding local atomic environments (radius=2) |
| MACCS Keys | 167 | Binary pharmacophore-level structural patterns |
| RDKit 2D Descriptors | ~200 | Physicochemical properties: MW, logP, TPSA, H-bond donors/acceptors, aromaticity, etc. |
| ChemBERTa Embeddings | 768 | `[CLS]` token from a BERT-style transformer pretrained on 77M SMILES (PubChem) |

```python
def featurize(smiles: str) -> np.ndarray:
    morgan  = get_morgan_fp(smiles)       # 2048-dim
    maccs   = get_maccs_fp(smiles)        # 167-dim
    rdkit   = get_rdkit_descriptors(smiles) # ~200-dim
    bert    = get_chemberta_embedding(smiles) # 768-dim
    return np.concatenate([morgan, maccs, rdkit, bert])
```

### 2. Cross-Validation

Random K-fold CV is inappropriate for molecular data — structurally similar molecules can appear in both train and validation sets, producing inflated estimates that don't generalize.

**Meerkat uses scaffold-based GroupKFold** (Murcko scaffolds, 5 folds), the industry standard for QSAR evaluation. Each fold splits by molecular scaffold, ensuring the model is evaluated on structurally novel compounds.

```python
df["scaffold"] = df["Canonical_Smiles"].apply(get_murcko_scaffold)
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X, y, groups=df["scaffold"]))
```

### 3. Models & Ensemble

Five models are trained on the same scaffold CV folds and ensembled:

| # | Model | Feature Set |
|---|---|---|
| 1 | XGBoost | Morgan + MACCS + RDKit |
| 2 | XGBoost | ChemBERTa only |
| 3 | XGBoost | Full 3,200-dim |
| 4 | LightGBM | Morgan + MACCS + RDKit |
| 5 | CatBoost | Full 3,200-dim |

Final predictions are OOF score-weighted averages across models and folds. A Ridge meta-learner stacking variant is also evaluated.

### 4. Hyperparameter Tuning

XGBoost hyperparameters are tuned with **Optuna** (100 trials) optimizing scaffold CV OOF score directly — not a proxy metric.

### 5. Handling Small Dataset Size (~1,600 samples)

- PCA on Morgan fingerprints (2,048 → 200 components, ~95% variance retained) to reduce noise from sparse bits
- Variance threshold feature selection to remove near-zero-variance features
- Optional pseudo-labeling from external ChEMBL CYP3A4 data, filtered by ensemble prediction variance

---

## File Structure

```
Meerkat_v01/
├── data/
│   ├── train.csv               # ID, Canonical_Smiles, Inhibition
│   ├── test.csv                # ID, Canonical_Smiles
│   └── sample_submission.csv
│
├── features/                   # Precomputed feature arrays (.npy)
│   ├── tr_morgan.npy
│   ├── tr_maccs.npy
│   ├── tr_rdkit.npy
│   ├── te_morgan.npy
│   ├── te_maccs.npy
│   └── te_rdkit.npy
│
├── models/                     # Saved model weights per fold
├── submissions/                # Competition submission CSVs
│
├── cyp3a4_predictor.ipynb      # Main training notebook
├── cyp3a4_predictor_plan.md    # Full model design document
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/<your-username>/meerkat
cd meerkat/Meerkat_v01
pip install -r requirements.txt
jupyter notebook cyp3a4_predictor.ipynb
```

---

## Dependencies

```
rdkit>=2023.9.1
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
optuna>=3.4.0
transformers>=4.35.0
torch>=2.1.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
```

---

## License & Compliance

All pretrained models and external data use licenses permitting non-commercial use (MIT, Apache 2.0, CC BY-SA 4.0) and were released prior to the competition cutoff date (2025-06-22).

| Resource | License |
|---|---|
| RDKit | BSD |
| XGBoost / LightGBM / CatBoost | Apache 2.0 / MIT / Apache 2.0 |
| ChemBERTa (`seyonec/ChemBERTa-zinc-base-v1`) | MIT |
| ChEMBL external data | CC BY-SA 4.0 |

> API-based inference (OpenAI, Gemini, etc.) is not used. All inference runs locally.

---

*GISTomics AI-Driven Drug Discovery Study Group — Spring 2026*
