# dataset.py
import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from minimol import Minimol
from hydra.core.global_hydra import GlobalHydra

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")
# Suppress tqdm warnings
tqdm.disable_warnings = True
# Suppress torch FutureWarnings for load
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"You are using `torch.load` with `weights_only=False`.*"
)

def standardize(smiles: str) -> str:
    """
    Standardize a SMILES string using RDKit MolStandardize.
    Returns canonical SMILES or None if invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or len(smiles) == 0:
            return None
        mol = Chem.RemoveHs(mol)
        clean = rdMolStandardize.Cleanup(mol)
        parent = rdMolStandardize.FragmentParent(clean)
        uncharged = rdMolStandardize.Uncharger().uncharge(parent)
        taut = rdMolStandardize.TautomerEnumerator().Canonicalize(uncharged)
        return Chem.MolToSmiles(taut, canonical=True) if taut else None
    except Exception:
        return None

def precompute_features(
    input_data,
    output_dir: str,
    smiles_col: str = "SMILES",
    target_cols: list = None,
    save_smiles: bool = True,
    standardize_smiles: bool = True,
    batch_size: int = 10000,
    sep: str = ",",
):
    """
    Reads a CSV file or DataFrame, standardizes SMILES, filters invalid/missing,
    featurizes with Minimol, and saves under output_dir:
      - features.pt    (FloatTensor [N, D])
      - targets.pt     (FloatTensor [N, T]) if targets
      - metadata.csv   (raw SMILES) if save_smiles
      - meta.json      (smiles_col, target_cols)
    """
    # Load DataFrame
    if isinstance(input_data, str):
        df = pd.read_csv(input_data, sep=sep)
    else:
        df = input_data.copy()

    # Check columns
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found.")
    if target_cols:
        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Target columns {missing} not in DataFrame.")

    raw_smiles, targets_list = [], []

    # Standardize and filter
    for _, row in tqdm(df.iterrows(), total=len(df)):
        smi = row[smiles_col]
        std = standardize(smi) if standardize_smiles else smi
        if std is None:
            continue
        if target_cols:
            t = row[target_cols].to_numpy(dtype=np.float32, na_value=np.nan)
            if np.isnan(t).any():
                continue
            targets_list.append(t)
        raw_smiles.append(std)

    # if Hydra was already initialized by a previous run, clear it
    gh = GlobalHydra.instance()
    if gh.is_initialized():
        gh.clear()

    # Featurize
    featurizer = Minimol(batch_size=int(batch_size))
    try:
        feats = featurizer(raw_smiles)
    except Exception as e:
        raise RuntimeError(f"Featurization error: {e}")

    features_tensor = torch.stack([f.clone().detach().float() for f in feats])
    targets_tensor = None
    if target_cols and len(targets_list) > 0:
        arr = np.stack(targets_list, axis=0)
        targets_tensor = torch.tensor(arr, dtype=torch.float32)

    # Write to disk
    os.makedirs(output_dir, exist_ok=True)
    torch.save(features_tensor, os.path.join(output_dir, "features.pt"))
    if targets_tensor is not None:
        torch.save(targets_tensor, os.path.join(output_dir, "targets.pt"))
    if save_smiles:
        pd.DataFrame({smiles_col: raw_smiles}).to_csv(
            os.path.join(output_dir, "metadata.csv"), index=False
        )
    # Save config
    meta = {"smiles_col": smiles_col, "target_cols": target_cols}
    import json
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved features.pt{', targets.pt' if targets_tensor is not None else ''}{', metadata.csv' if save_smiles else ''}, meta.json in {output_dir}")

class PrecomputedDataset(Dataset):
    """
    Loads pre-saved features.pt, targets.pt, and optional metadata.csv.
    """
    def __init__(
        self,
        features_path: str,
        targets_path: str = None,
        metadata_path: str = None,
    ):
        self.features = torch.load(features_path)
        self.targets = torch.load(targets_path) if targets_path and os.path.exists(targets_path) else None
        self.metadata = pd.read_csv(metadata_path) if metadata_path and os.path.exists(metadata_path) else None

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        sample = {'x': self.features[idx]}
        if self.targets is not None:
            sample['y'] = self.targets[idx]
        if self.metadata is not None:
            sample.update(self.metadata.iloc[idx].to_dict())
        return sample

class PrecomputedDataModule:
    """
    Simple loader for precomputed train/val/test directories.
    """
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        test_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = PrecomputedDataset(
            os.path.join(self.train_dir, "features.pt"),
            os.path.join(self.train_dir, "targets.pt"),
            os.path.join(self.train_dir, "metadata.csv"),
        )
        self.val_ds = PrecomputedDataset(
            os.path.join(self.val_dir, "features.pt"),
            os.path.join(self.val_dir, "targets.pt"),
            os.path.join(self.val_dir, "metadata.csv"),
        )
        self.test_ds = PrecomputedDataset(
            os.path.join(self.test_dir, "features.pt"),
            os.path.join(self.test_dir, "targets.pt"),
            os.path.join(self.test_dir, "metadata.csv"),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
