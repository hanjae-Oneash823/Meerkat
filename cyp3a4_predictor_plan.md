# CYP3A4 Inhibition Predictor — Model Design Plan

**Competition**: Predict % inhibition of CYP3A4 from canonical SMILES strings  
**Training set**: ~1,600 samples (SMILES + Inhibition %)  
**Test set**: 100 SMILES (predict inhibition %, submit to leaderboard)  
**Metric**: `Score = 0.5 × (1 - min(NRMSE, 1)) + 0.5 × Pearson r`

---

## Table of Contents

1. [Overview & Rationale](#1-overview--rationale)
2. [Stage 1 — Molecular Featurization](#2-stage-1--molecular-featurization)
3. [Stage 2 — XGBoost Model](#3-stage-2--xgboost-model)
4. [Stage 3 — Cross-Validation Strategy](#4-stage-3--cross-validation-strategy)
5. [Stage 4 — Ensembling](#5-stage-4--ensembling)
6. [Stage 5 — Handling Small Dataset Size](#6-stage-5--handling-small-dataset-size)
7. [Execution Roadmap](#7-execution-roadmap)
8. [What Differentiates This Approach](#8-what-differentiates-this-approach)
9. [File Structure](#9-file-structure)
10. [Dependencies & Compliance](#10-dependencies--compliance)

---

## 1. Overview & Rationale

### Scoring Metric Breakdown

```
A = Normalized RMSE = RMSE / (max(y) - min(y))      # lower is better
B = Pearson r = clip(Cov(y, ŷ) / (σ_y · σ_ŷ), 0, 1) # higher is better

Score = 0.5 × (1 - min(A, 1)) + 0.5 × B
```

The dual metric means the model must:
- Be accurate in absolute terms (minimize RMSE)
- Capture the **trend** of inhibition values correctly (maximize correlation)

A model that outputs the mean of all training labels would score 0.5 on correlation and near-0 on NRMSE — so both components require serious modeling effort.

### Core Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Base model | XGBoost | Fast, robust, strong baseline for tabular/fingerprint data |
| Feature type | Morgan FP + RDKit descriptors + ChemBERTa | Best of classical + deep representations |
| CV strategy | Scaffold-based 5-fold | Prevents structural data leakage |
| Ensembling | XGBoost + LightGBM weighted average | Reduces variance on small dataset |
| Data augmentation | Optional pseudo-labeling | Addresses ~1600 sample limitation |

---

## 2. Stage 1 — Molecular Featurization

This is the **most critical stage**. The quality of molecular features directly determines the ceiling of model performance.

### 2.1 Classical Fingerprints

#### Morgan Fingerprints (ECFP4)
- Circular fingerprints that encode local atomic environments up to radius 2
- Standard for QSAR (Quantitative Structure-Activity Relationship) models
- Produces a binary bit vector of length 2048

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def get_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp)
```

#### MACCS Keys
- 166 binary features encoding predefined pharmacophore-level structural patterns
- Complementary to Morgan FP — captures "structural vocabulary" rather than local environments

```python
from rdkit.Chem import MACCSkeys

def get_maccs_fp(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(167)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp)
```

#### RDKit 2D Descriptors
- ~200 physicochemical descriptors: molecular weight, logP, TPSA, H-bond donors/acceptors, rotatable bonds, ring count, aromaticity, etc.
- Interpretable features directly relevant to CYP3A4 interactions (hydrophobicity, size, planarity)

```python
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

def get_rdkit_descriptors(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(200)
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(
        [d[0] for d in Descriptors.descList]
    )
    return np.array(calc.CalcDescriptors(mol))
```

### 2.2 Pretrained Molecular Embeddings

#### ChemBERTa
- A BERT-style transformer pretrained on ~77 million SMILES strings from PubChem
- Extract the `[CLS]` token embedding (768-dim) as a molecular representation
- License: MIT-compatible (check exact release date compliance with contest rules: must be before 2025.06.22)

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model.eval()

def get_chemberta_embedding(smiles: str) -> np.ndarray:
    inputs = tokenizer(smiles, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding  # shape: (768,)
```

### 2.3 Feature Concatenation

Combine all representations into a single feature vector per molecule:

```
Final feature vector = [Morgan FP (2048) | MACCS (167) | RDKit descriptors (200) | ChemBERTa (768)]
                     = ~3183 dimensions total
```

```python
def featurize(smiles: str) -> np.ndarray:
    morgan = get_morgan_fp(smiles)
    maccs = get_maccs_fp(smiles)
    rdkit = get_rdkit_descriptors(smiles)
    chemberta = get_chemberta_embedding(smiles)
    return np.concatenate([morgan, maccs, rdkit, chemberta])
```

> **Note**: Handle NaN values in RDKit descriptors with imputation (median fill) before feeding into XGBoost.

---

## 3. Stage 2 — XGBoost Model

### 3.1 Model Architecture

```
Feature vector (3183-dim)
        │
        ▼
  [XGBoost Regressor]
   - Gradient boosted trees
   - Optimized for RMSE
        │
        ▼
  Predicted Inhibition (%)
```

### 3.2 Key Hyperparameters

| Parameter | Search Range | Notes |
|---|---|---|
| `n_estimators` | 500 – 3000 | Use early stopping on validation fold |
| `max_depth` | 4 – 8 | Keep shallow to avoid overfitting on 1600 samples |
| `learning_rate` | 0.01 – 0.05 | Lower = better generalization, needs more trees |
| `subsample` | 0.6 – 0.9 | Row subsampling per tree |
| `colsample_bytree` | 0.5 – 0.8 | Column subsampling per tree |
| `min_child_weight` | 3 – 20 | Key regularizer for small datasets |
| `gamma` | 0 – 5 | Minimum loss reduction for split |
| `reg_alpha` | 0 – 1 | L1 regularization |
| `reg_lambda` | 1 – 5 | L2 regularization |

### 3.3 Hyperparameter Tuning with Optuna

```python
import optuna
import xgboost as xgb

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "objective": "reg:squarederror",
        "tree_method": "hist",  # fast histogram-based
        "random_state": 42,
    }
    # evaluate on scaffold CV OOF score
    return scaffold_cv_score(params)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

### 3.4 Target Transformation (Optional but Recommended)

Inhibition % is bounded `[0, 100]`. Tree models don't natively respect this constraint. Consider:

```python
# Option A: log1p transform (softens high-end outliers)
y_train_transformed = np.log1p(y_train)
# ... train model ...
y_pred = np.expm1(model.predict(X_test))

# Option B: Logit transform (maps [0,100] to (-inf, inf))
y_train_norm = y_train / 100.0
y_train_transformed = np.log(y_train_norm / (1 - y_train_norm + 1e-6))
# ... train model ...
y_pred = 100.0 * (1 / (1 + np.exp(-model.predict(X_test))))
```

Evaluate both on scaffold CV — choose whichever improves OOF score.

---

## 4. Stage 3 — Cross-Validation Strategy

### 4.1 Why Scaffold-Based CV?

| CV Type | Validation Leak Risk | Realistic Estimate |
|---|---|---|
| Random K-Fold | HIGH — structurally similar molecules may appear in train and val | Overly optimistic |
| Scaffold K-Fold | LOW — splits by Murcko scaffold, preventing structural leakage | Realistic generalization |

The test set contains compounds that may be structurally novel. Random split gives inflated CV scores that won't translate to private leaderboard performance. Scaffold split is the **industry standard** for QSAR evaluation.

### 4.2 Scaffold Split Implementation

```python
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd

def get_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles  # fallback: use original SMILES as group
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold

# Assign scaffold group to each molecule
df["scaffold"] = df["Canonical_Smiles"].apply(get_scaffold)

# Use scaffold as group in GroupKFold
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X, y, groups=df["scaffold"]))
```

### 4.3 OOF (Out-of-Fold) Predictions

For each fold, save predictions on the held-out set:

```python
oof_preds = np.zeros(len(y_train))

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = xgb.XGBRegressor(**best_params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=50,
              verbose=False)

    oof_preds[val_idx] = model.predict(X_val)

# Evaluate OOF score (this is your true estimate of test performance)
oof_score = compute_competition_score(y_train, oof_preds)
print(f"OOF Score: {oof_score:.4f}")
```

### 4.4 Competition Score Function

```python
from scipy.stats import pearsonr

def compute_competition_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    A = min(nrmse, 1.0)

    r, _ = pearsonr(y_true, y_pred)
    B = max(0.0, min(r, 1.0))  # clip to [0, 1]

    score = 0.5 * (1 - A) + 0.5 * B
    return score
```

---

## 5. Stage 4 — Ensembling

### 5.1 Model Pool

Train multiple models on the same scaffold CV folds:

| # | Model | Feature Set | Rationale |
|---|---|---|---|
| 1 | XGBoost | Morgan + MACCS + RDKit | Classical QSAR pipeline |
| 2 | XGBoost | ChemBERTa embeddings only | Deep representation |
| 3 | XGBoost | Full concatenated features (3183-dim) | Combined |
| 4 | LightGBM | Morgan + MACCS + RDKit | Different boosting algorithm |
| 5 | CatBoost | Full concatenated features | Symmetric trees, different inductive bias |

### 5.2 Ensemble Weighting

Use OOF scores to determine weights. Two options:

**Option A — Score-proportional weighting:**
```python
oof_scores = [score_model1, score_model2, ..., score_model5]
weights = np.array(oof_scores) / np.sum(oof_scores)
final_pred = np.average(test_preds, axis=0, weights=weights)
```

**Option B — Stacking (Ridge meta-learner on OOF):**
```python
from sklearn.linear_model import Ridge

# Stack OOF predictions from all models as features
oof_stack = np.column_stack([oof_model1, oof_model2, ..., oof_model5])
test_stack = np.column_stack([test_model1, test_model2, ..., test_model5])

meta = Ridge(alpha=1.0)
meta.fit(oof_stack, y_train)
final_pred = meta.predict(test_stack)
```

> Start with Option A (simpler). Move to Option B only if it improves OOF score.

### 5.3 Test Prediction Aggregation

For each model, average predictions across all 5 folds (trained on different fold combinations):

```python
test_preds_per_fold = []
for fold_model in fold_models:
    test_preds_per_fold.append(fold_model.predict(X_test))

model_test_pred = np.mean(test_preds_per_fold, axis=0)
```

---

## 6. Stage 5 — Handling Small Dataset Size

With only ~1,600 training samples, the main risks are:
- **Overfitting** to training distribution
- **Poor generalization** to structurally novel test compounds

### 6.1 Dimensionality Reduction

High-dimensional sparse features (Morgan FP: 2048 bits, ~90% zeros) can introduce noise.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Apply PCA to Morgan fingerprints before concatenation
scaler = StandardScaler()
pca = PCA(n_components=200, random_state=42)

morgan_scaled = scaler.fit_transform(morgan_features)
morgan_reduced = pca.fit_transform(morgan_scaled)  # 2048 → 200 dims
```

This reduces Morgan FP from 2048 to 200 PCA components while retaining ~95% variance.

### 6.2 Pseudo-Labeling (Virtual Data Augmentation)

Inspired by the GNN-augmentation approach used by other teams, but without requiring a GNN:

**Step 1**: Train a base model on all training data.  
**Step 2**: Collect an external CYP3A4 inhibition dataset (e.g., from ChEMBL, licensed CC BY-SA 4.0 — verify date compliance).  
**Step 3**: Predict labels for external molecules using the base model.  
**Step 4**: Filter to high-confidence predictions (e.g., where ensemble variance is low).  
**Step 5**: Retrain with augmented dataset.

```python
# Filter pseudo-labeled samples by prediction variance across ensemble
pred_variance = np.var(np.column_stack(all_model_preds), axis=1)
confident_mask = pred_variance < np.percentile(pred_variance, 25)  # bottom 25% variance

X_augmented = np.vstack([X_train, X_external[confident_mask]])
y_augmented = np.concatenate([y_train, ensemble_pred[confident_mask]])
```

> **Risk**: Noisy pseudo-labels can hurt performance. Use conservatively. Only add if OOF score improves.

### 6.3 Feature Selection

Remove features with near-zero variance or very high pairwise correlation:

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X_train)
```

---

## 7. Execution Roadmap

### Phase 1 — Baseline (Day 1–2)

- [ ] Implement `featurize()` for all training and test molecules
- [ ] Morgan FP + RDKit descriptors → XGBoost with random 5-fold CV
- [ ] Implement `compute_competition_score()`
- [ ] Submit baseline prediction

### Phase 2 — Scaffold CV (Day 2–3)

- [ ] Implement `get_scaffold()` function
- [ ] Switch to scaffold-based GroupKFold
- [ ] Compare OOF scores: random split vs scaffold split
- [ ] Observe score gap — this is your "true generalization estimate"

### Phase 3 — Feature Upgrade (Day 3–4)

- [ ] Add ChemBERTa embeddings
- [ ] Concatenate all features (3183-dim)
- [ ] Evaluate each feature set individually and combined on scaffold CV
- [ ] Apply PCA to Morgan FP if dimensionality causes slow training

### Phase 4 — Hyperparameter Tuning (Day 4–5)

- [ ] Run Optuna study (100 trials) on scaffold CV OOF score
- [ ] Apply best hyperparameters
- [ ] Evaluate target transformation (log1p vs logit vs no transform)

### Phase 5 — Ensemble (Day 5–6)

- [ ] Train LightGBM with same features and scaffold CV
- [ ] Train CatBoost variant
- [ ] Compute OOF-weighted ensemble
- [ ] Try Ridge meta-learner stacking
- [ ] Compare ensemble OOF score vs single best model

### Phase 6 — Augmentation (Day 6–7, Optional)

- [ ] Source external CYP3A4 data from ChEMBL (verify license compliance)
- [ ] Generate pseudo-labels with ensemble
- [ ] Filter by prediction variance
- [ ] Retrain with augmented data; compare OOF score

### Phase 7 — Final Submission

- [ ] Select best configuration based on OOF score
- [ ] Train final model on **all** training data (no held-out fold)
- [ ] Generate predictions for 100 test molecules
- [ ] Format as `sample_submission.csv` and submit

---

## 8. What Differentiates This Approach

### vs. Team using Five-Fold CV with CatBoost/LightGBM
- Our scaffold split CV gives a more realistic performance estimate
- Random split teams may suffer significant shake-up on the private leaderboard
- We also ensemble gradient boosters → comparable final output, more honest evaluation

### vs. Team using GNN for virtual data generation
- GNN training requires significant compute and expertise
- Our ChemBERTa embeddings provide comparable deep molecular representations without custom GNN training
- Pseudo-labeling from an external dataset is a lighter-weight augmentation strategy

### vs. Team using clustering for test set analysis
- Clustering can be used complementarily: analyze test molecule scaffolds to understand which training compounds are most similar
- Can inform which training samples to up-weight, or whether augmentation is truly necessary

---

## 9. File Structure

```
cyp3a4_predictor/
├── data/
│   ├── train.csv                  # ID, Canonical_Smiles, Inhibition
│   ├── test.csv                   # ID, Canonical_Smiles
│   └── sample_submission.csv      # ID, Inhibition (template)
│
├── features/
│   ├── morgan_fp.npy              # precomputed (train)
│   ├── maccs_fp.npy
│   ├── rdkit_desc.npy
│   ├── chemberta_emb.npy
│   ├── morgan_fp_test.npy         # precomputed (test)
│   ├── maccs_fp_test.npy
│   ├── rdkit_desc_test.npy
│   └── chemberta_emb_test.npy
│
├── models/
│   ├── xgb_fold1.json             # saved XGBoost models per fold
│   ├── xgb_fold2.json
│   └── ...
│
├── notebooks/
│   ├── 01_eda.ipynb               # exploratory data analysis
│   ├── 02_featurization.ipynb     # test all feature types
│   ├── 03_baseline.ipynb          # random CV baseline
│   ├── 04_scaffold_cv.ipynb       # scaffold split evaluation
│   ├── 05_hyperparameter_tuning.ipynb
│   └── 06_ensemble.ipynb
│
├── src/
│   ├── featurize.py               # featurization functions
│   ├── scaffold_cv.py             # scaffold split utilities
│   ├── train.py                   # training loop
│   ├── evaluate.py                # score computation
│   └── ensemble.py                # ensembling logic
│
├── submissions/
│   └── submission_v1.csv
│
└── requirements.txt
```

---

## 10. Dependencies & Compliance

### Python Dependencies

```txt
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

### License Compliance

Per contest rules, all pretrained models and external data must:
- Have been officially released **before 2025.06.22**
- Be distributed under a license permitting at minimum non-commercial use (MIT, Apache 2.0, CC BY-NC, CC0, etc.)
- Have source, dataset name, and license information documented in code submission

| Resource | License | Compliant |
|---|---|---|
| RDKit | BSD | ✅ |
| XGBoost | Apache 2.0 | ✅ |
| LightGBM | MIT | ✅ |
| CatBoost | Apache 2.0 | ✅ |
| ChemBERTa (`seyonec/ChemBERTa-zinc-base-v1`) | MIT | ✅ (verify release date) |
| ChEMBL external data | CC BY-SA 4.0 | ✅ (verify date) |

> **API-based models (OpenAI, Gemini) are explicitly prohibited** per contest rule 4.2. All inference must run locally.

---

*Last updated: 2026-05-26*
