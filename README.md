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
git clone https://github.com/yourusername/minimol-toolkit.git](https://github.com/leifsieben/MiniMol-STL-MTL-Unified
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
  "Activity": [0,    1,     0]
})

precompute_features(
  input_data=df,
  output_dir="data/train",
  smiles_col="SMILES",
  target_cols=["Activity"],
  save_smiles=True,
  standardize_smiles=True,
  batch_size=100
)
```
```bash
python minimol_train.py --precompute \
  --input_tsv data.csv \
  --fingerprint_col SMILES \
  --target_cols Activity \
  --output_dir data/train
```

Output directory structure:
```
data/train/
├── features.pt        # FloatTensor [N, D]
├── targets.pt         # FloatTensor [N, T]
├── metadata.csv       # SMILES strings
└── meta.json          # Configuration
```

---

## 🚀 Training Models

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
  --monitor_metric loss
```
```python
from minimol_train import train_model

cfg = {
  "batch_size":     64,
  "learning_rate":  1e-3,
  "dim_size":       128,
  "shrinking_scale":0.5,
  "num_layers":     3,
  "dropout_rate":   0.1,
  "activation_function":"relu",
  "use_batch_norm": True,
  "use_residual":   False,
}

# STL:
train_model(
  config=cfg,
  train_dir="data/train",
  val_dir="data/val",
  test_dir="data/test",
  mode="stl",
  max_epochs=20
)
```

### Multi-Task Model (MTL)

```bash
python minimol_train.py \
  --mode mtl \
  --train_dir data/train \
  --val_dir data/val \
  --test_dir data/test \
  --max_epochs 30 \
  --batch_size 128 \
  --learning_rate 5e-4 \
  --monitor_metric auroc \
  --monitor_task 0
```

```python
# MTL:
train_model(
  config=cfg,
  train_dir="data/train",
  val_dir="data/val",
  test_dir="data/test",
  mode="mtl",
  max_epochs=30,
  monitor_metric="auroc",
  monitor_task=0
)
```

Checkpoints are saved automatically in the working directory with the best monitored metric.

---

## 🔍 Hyperparameter Optimization

```bash
python minimol_train.py \
  --mode stl \
  --train_dir data/train \
  --val_dir data/val \
  --test_dir data/test \
  --hyperopt \
  --hyperopt_num_samples 50 \
  --max_epochs 10 \
  --cpus_per_trial 2 \
  --gpus_per_trial 1 \
  --tune_dir ray_results
```

```python
from minimol_train import run_hyperopt

best_config = run_hyperopt(
  train_dir="data/train",
  val_dir="data/val",
  test_dir="data/test",
  mode="stl",
  max_epochs=5,
  hyperopt_num_samples=20,
  cpus_per_trial=1,
  gpus_per_trial=0,
  tune_dir="./ray_results"
)
print("🏆 Best hyperparameters:", best_config)
```

Results and best configuration are logged via Ray Tune.

---

## 🔮 Prediction API

### Predict for a single SMILES

```python
from minimol_predict import predict_smiles

ckpt_path = "best_model.ckpt"
predictions = predict_smiles("CCO", ckpt_path, mode='stl')
print(predictions)  # numpy array of shape (1,)
```

### Batch prediction with DataFrame

```python
from minimol_predict import predict_smiles, predict_df
import pandas as pd

# Single SMILES
pred = predict_smiles("CCO", "best_model.ckpt", mode="stl")
print(pred)  # numpy array

# Batch via DataFrame
df_new = pd.DataFrame({"SMILES": ["CCC","CCN","COC"]})
out = predict_df(
  df_new,
  smiles_col="SMILES",
  checkpoint_path="best_model.ckpt",
  mode="mtl",
  aggregated=False
)
print(out)
```

Output DataFrame contains original SMILES and `pred_model_{i}_task_{j}` columns.

---

## 🎯 End-to-End Example

```bash
# 1) Precompute
python minimol_train.py --precompute \
  --input_tsv data.csv \
  --fingerprint_col SMILES \
  --target_cols Hit \
  --output_dir train

python minimol_train.py --precompute \
  --input_tsv data.csv \
  --fingerprint_col SMILES \
  --target_cols Hit \
  --output_dir val \
  --rows 2:3

python minimol_train.py --precompute \
  --input_tsv data.csv \
  --fingerprint_col SMILES \
  --target_cols Hit \
  --output_dir test \
  --rows 3:

# 2) Train + Hyperopt
python minimol_train.py \
  --mode stl \
  --train_dir train \
  --val_dir val \
  --test_dir test \
  --hyperopt \
  --hyperopt_num_samples 20 \
  --max_epochs 5 \
  --cpus_per_trial 1

# 3) Pick best & predict
BEST=$(ls -t ray_results/*/checkpoint_*/*.ckpt | head -n1)
python minimol_predict.py predict-csv \
  --input test.csv \
  --smiles-col SMILES \
  --checkpoint $BEST \
  --mode stl \
  --output preds.csv
```
Here a python example with hyperparameter optimization: 

```python
import pandas as pd, glob, os
from minimol_dataset import precompute_features
from minimol_train   import train_model, run_hyperopt
from minimol_predict import predict_df

# Split & precompute
df = pd.read_csv("data.csv")
splits = {
  "train": df.iloc[:2],
  "val":   df.iloc[2:3],
  "test":  df.iloc[3:]
}
for name, split in splits.items():
    precompute_features(split, name, smiles_col="SMILES", target_cols=["Hit"], clean_data=True)

# Hyperopt
best_cfg = run_hyperopt("train","val","test", mode="stl", max_epochs=5, hyperopt_num_samples=20)
print("Best:", best_cfg)

# Find best checkpoint
ckpts = glob.glob(os.path.join("ray_results","*","checkpoint_*","*.ckpt"))
best_ckpt = max(ckpts, key=os.path.getmtime)

# Predict
df_test = splits["test"][["SMILES"]]
out = predict_df(df_test, "SMILES", best_ckpt, mode="stl", aggregated=False)
print(out)
```
