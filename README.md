# MiniMol Predictive Modeling Toolkit

A lightweight Python library for precomputing molecular features and training single-task (STL) or multi-task (MTL) feedforward neural networks to predict molecular properties using precomputed fingerprints or learned representations.

---

## 📦 Installation

Install the package and its dependencies via `pip`:

```bash
pip install torch pytorch-lightning scikit-learn rdkit-pypi minimol ray[tune]
```

Alternatively, clone this repository and install with:

```bash
git clone https://github.com/leifsieben/MiniMol-STL-MTL-Unified
cd MiniMol-STL-MTL-Unified
pip install -r requirements.txt
```

---

## 🛠️ Functionality

1. **Precompute Features**: Standardize SMILES, featurize molecules, and save tensors.
2. **Train Models**:
   - **STL_FFN**: Single-task feedforward network.
   - **MTL_FFN**: Multi-task feedforward network.
   - Optional early stopping and model checkpointing.
3. **Hyperparameter Optimization**: Use Ray Tune with ASHA scheduler.
4. **Prediction**:
   - Predict on SMILES strings or pandas DataFrames.
   - Supports both STL and MTL models.

---

## ⚙️ Precomputing Features

```python
from minimol_dataset import precompute_features
import pandas as pd

df = pd.DataFrame({
    "SMILES":   ["CCO", "CCC", "CCN"],
    "Activity": [0,      1,     0]
})

precompute_features(
    input_data=df,
    output_dir="data/train",      # will be created if missing
    smiles_col="SMILES",          # defaults to "SMILES"
    target_cols=["Activity"],     # optional – pass `None` for inference-only
    metadata_cols=None,           # keep *all* original columns
    standardize_smiles=True,      # canonicalises & de-duplicates
    batch_size=1024               # default 1 000
)

```

Output directory structure:
```
data/train/
├── features.pt     # torch.FloatTensor [N, D]
├── targets.pt      # torch.FloatTensor [N, T]  (only if target_cols given)
├── metadata.csv    # original rows that were successfully featurised
└── meta.json       # stores smiles_col / target_cols / metadata_cols
```

---

## 🚀 Training Models

You can use the flag  --use_residual (CLI) or use_residual=True (Python) – to move from the original “plain” feed-forward network to a residual (skip-connected) variant.

### Single-Task Model (STL)

```bash
python minimol_train.py \
  --mode stl \
  --train_dir data/train \
  --val_dir data/val \
  --test_dir data/test \
  --max_epochs 20 \
  --batch_size 64 \
  --learning_rate 1e-3 \
  --monitor_metric auroc \
  --use_residual        # <-- add this flag for the residual net
```
```python
from minimol_train import train_model

cfg = dict(
    dim_size=128,
    shrinking_scale=0.5,
    num_layers=3,
    dropout_rate=0.1,
    activation_function="relu",
    use_batch_norm=True,
    learning_rate=1e-3,
    batch_size=64,
    use_residual=True,        # <-- toggle here
)

train_model(
    config       = cfg,
    train_dir    = "data/train",
    val_dir      = "data/val",
    test_dir     = "data/test",
    ckpt_root    = "checkpoints/stl",   # required positional arg now :contentReference[oaicite:1]{index=1}
    mode         = "stl",
    max_epochs   = 20,
    monitor_metric="auroc"
)
```

### Multi-Task Model (MTL)

```bash
python minimol_train.py \
  --mode mtl \
  --train_dir data/train \
  --val_dir data/val \
  --max_epochs 30 \
  --batch_size 128 \
  --learning_rate 5e-4 \
  --monitor_metric auprc \
  --monitor_task 0     # <-- by default doesn't uses standard FFN architecture
  ```

```python
# MTL:
train_model(
    config          = cfg,
    train_dir       = "data/train",
    val_dir         = "data/val",
    ckpt_root       = "checkpoints/mtl",
    mode            = "mtl",
    max_epochs      = 30,
    monitor_metric  = "auprc",
    monitor_task    = 0,          # which task to watch for “best model”
    use_residual    = True
)
```

Checkpoints are saved automatically in the working directory with the best monitored metric. Models land in
```
<ckpt_root>/
  single_run/          # or ensemble_{i}/
    epoch=...-val_auroc_task_0=0.9339.ckpt
    last.ckpt
```

Ensembles: --ensemble 5 trains ensemble_0 … ensemble_4 with independent seeds; predictions can later be averaged via minimol_predict.py.

---

## 🔍 Hyperparameter Optimization

```bash
python minimol_train.py \
  --mode stl \
  --train_dir data/train \
  --val_dir   data/val \
  --hyperopt               \        # activate Optuna search
  --n_trials 50            \        # <-- was --hyperopt_num_samples
  --max_epochs 10          \
  --num_workers 4          \        # dataloader workers, not CPU cores
  --monitor_metric auroc   \        # optimise AUROC instead of loss
  --use_residual                    # let Optuna try the residual net

```

```python
from minimol_train import run_hyperopt

best_params = run_hyperopt(
    train_dir   = "data/train",
    val_dir     = "data/val",
    mode        = "stl",
    max_epochs  = 10,
    n_trials    = 20,
    num_workers = 4,
    monitor_metric = "auroc",
    use_residual   = True      # search the residual variant too
)
print("🏆 Best hyper-parameters:", best_params)
```

Results and best configuration are logged via Optuna and lightning. 

---

## 🔮 Prediction API

### Predict for a single SMILES

```python
from minimol_predict import predict_smiles

ckpt = "best_model.ckpt"                # or a *list* of ckpts for an ensemble
score = predict_smiles(
    "CCO",
    checkpoints = ckpt,                 # str | list[str]
    mode        = "stl",                # "stl" | "mtl"
    aggregated  = True,                 # default → mean across models
    architecture= "standard",           # "standard" | "residual"
)
print(score)   # scalar (STL) or 1-D array (MTL)
```

### Batch prediction with DataFrame

```python
from minimol_predict import predict_df
import pandas as pd

df = pd.DataFrame({"SMILES": ["CCC", "CCN", "COC"]})

out = predict_df(
    df,
    smiles_col       = "SMILES",
    checkpoints      = ["ens0.ckpt", "ens1.ckpt"],   # ⇦ ensemble
    mode             = "mtl",
    aggregated       = True,         # mean / std columns
    architecture     = "residual",   # load residual checkpoints
    task_of_interest = 0             # (optional) slice a single task
)
print(out.head())
```

### Predicting on pre-computed features tensor

```python
from minimol_predict import predict_on_precomputed

X = torch.load("data/train/features.pt")        # [N, D] tensor
preds = predict_on_precomputed(
    X,
    checkpoints   = ["ens0.ckpt", "ens1.ckpt"],
    mode          = "mtl",
    species_indices = [0, 3],    # pick tasks 0 & 3
    include_individual_models = False
)

```

Output DataFrame contains original SMILES and `pred_model_{i}_task_{j}` columns.

---

## 🎯 End-to-End Example

This is an example for a STL model with residual-skip connections, monitoring the AUPRC metric. It performs the train/val split internally with the default split of 80-20. 

```bash
########################################
# 1) Pre-compute MiniMol features
########################################
# Will create data/stl_split/ with features.pt, targets.pt, metadata.csv
python minimol_train.py --precompute \
  --input_tsv  data.csv            \
  --target_cols Hit                \
  --output_dir data/stl_split

########################################
# 2) Hyper-parameter search (Optuna)
########################################
python minimol_train.py \
  --mode stl \
  --train_dir data/stl_split        \
  --hyperopt                       \
  --n_trials 30                    \   # was --hyperopt_num_samples
  --max_epochs 5                   \
  --monitor_metric auprc           \   # optimise validation AUPRC
  --use_residual                       # ← search the residual architecture
# ⤷ Best trial parameters & checkpoint are logged under ./optuna_checkpoints/

########################################
# 3) Pick best checkpoint & predict
########################################
BEST=$(ls -t optuna_checkpoints/single_run/*.ckpt | head -n1)

python minimol_predict.py predict-csv \
  --input  data.csv                 \
  --smiles-col SMILES               \
  --checkpoint "$BEST"              \
  --mode stl                        \
  --output preds_stl.csv

```
No explicit val_dir given → Lightning’s PrecomputedDataModule automatically withholds 20 % for validation at run-time. AUPRC on that split drives checkpointing & Optuna pruning.

Here is a python example for a standard MTL model (3 tasks) where the 2nd task is monitored via loss. An ensemble of 3 models are trained, each with a different seed (handled internally) We use a user-defined split here. 

```python
"""
End-to-end: 3-task MTL, task-2 loss monitored, *ensemble of 3* models.
Assumes data.csv with columns: SMILES, Task0, Task1, Task2
"""
import os, glob, pandas as pd
from minimol_dataset import precompute_features
from minimol_train   import run_hyperopt, train_model
from minimol_predict import predict_df

# ──────────────────────────────────────────────────────────────
# 1) Split CSV → train / val / test
# ──────────────────────────────────────────────────────────────
df = pd.read_csv("data.csv")
n  = len(df)
train_df = df.iloc[:int(0.7*n)]
val_df   = df.iloc[int(0.7*n):int(0.85*n)]
test_df  = df.iloc[int(0.85*n):]

# ──────────────────────────────────────────────────────────────
# 2) Pre-compute MiniMol tensors for every split
# ──────────────────────────────────────────────────────────────
for name, split in {"train":train_df, "val":val_df, "test":test_df}.items():
    precompute_features(
        input_data = split,
        output_dir = f"data/mtl_{name}",
        smiles_col = "SMILES",
        target_cols= ["Task0", "Task1", "Task2"],
        standardize_smiles = True
    )

# ──────────────────────────────────────────────────────────────
# 3) Hyper-opt (single model) to get best hyper-params
# ──────────────────────────────────────────────────────────────
best_params = run_hyperopt(
    train_dir   = "data/mtl_train",
    val_dir     = "data/mtl_val",
    mode        = "mtl",
    max_epochs  = 8,
    n_trials    = 25,
    monitor_metric = "loss",   # minimise BCE loss
    monitor_task   = 2         # watch Task-2
)
print("🏆 Optuna best:", best_params)

# ──────────────────────────────────────────────────────────────
# 4) Train an *ensemble* of 3 models with those params
# ──────────────────────────────────────────────────────────────
ckpt_dir   = "checkpoints/mtl_ensemble"
ckpt_paths = []

for idx in range(3):                       # ensemble of 3
    ckpt = train_model(
        config         = {**best_params, "use_residual": False},
        train_dir      = "data/mtl_train",
        val_dir        = "data/mtl_val",
        test_dir       = "data/mtl_test",
        ckpt_root      = ckpt_dir,         # each member goes to ckpt_dir/ensemble_{idx}
        mode           = "mtl",
        max_epochs     = 15,
        monitor_metric = "loss",
        monitor_task   = 2,
        early_stop     = True,
        ensemble_size  = 3,                # inform Lightning callbacks
        ensemble_idx   = idx               # seed offset
    )
    ckpt_paths.append(ckpt)

print("✓ Ensemble checkpoints:", *ckpt_paths, sep="\n  ")

# ──────────────────────────────────────────────────────────────
# 5) Predict on the held-out test split, averaging the ensemble
# ──────────────────────────────────────────────────────────────
preds = predict_df(
    test_df[["SMILES"]],
    smiles_col   = "SMILES",
    checkpoints  = ckpt_paths,   # list → automatic mean / std
    mode         = "mtl",
    aggregated   = True          # adds pred_mean_task_k / pred_std_task_k
)

preds.to_csv("preds_mtl_ensemble.csv", index=False)
print(preds.head())
```
This pattern scales to any ensemble size—just change the range() and ensemble_size arguments.