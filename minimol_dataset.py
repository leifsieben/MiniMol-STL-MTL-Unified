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

import os, json
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from rdkit import Chem
from minimol import Minimol
from hydra.core.global_hydra import GlobalHydra

def precompute_features(
    input_data,
    output_dir: str,
    smiles_col: str = "SMILES",
    target_cols: list = None,
    metadata_cols: list = None,
    standardize_smiles: bool = True,
    batch_size: int = 1_000,
    skip_if_exists: bool = True
):
    """
    1) Load CSV/Parquet or DataFrame from `input_data`
    2) Standardize SMILES (if requested), drop invalid / NaN targets
    3) Featurize via MiniMol → FloatTensor [N, D] in large batches,
       skipping only the individual SMILES that truly fail
    4) Save under `output_dir`:
       - features.pt      (Tensor [N, D])
       - targets.pt       (Tensor [N, T]) if target_cols
       - metadata.csv     (subset of original columns for kept rows)
       - meta.json        (smiles_col, target_cols, metadata_cols)
    """
    os.makedirs(output_dir, exist_ok=True)
    feat_path = os.path.join(output_dir, "features.pt")
    if skip_if_exists and os.path.exists(feat_path):
        print(f"[precompute] skipping, found {feat_path}")
        return feat_path, os.path.join(output_dir, "metadata.csv")

    # ——— Load DataFrame ———
    if isinstance(input_data, (str, os.PathLike)):
        if str(input_data).endswith(".csv"):
            df = pd.read_csv(input_data)
        else:
            df = pd.read_parquet(input_data)
    else:
        df = input_data.copy()

    # ——— Column checks ———
    if smiles_col not in df.columns:
        raise KeyError(f"SMILES column '{smiles_col}' not found")
    if target_cols:
        missing = set(target_cols) - set(df.columns)
        if missing:
            raise KeyError(f"Target columns {missing} not in DataFrame")

    # default: save *all* columns
    if metadata_cols is None:
        metadata_cols = list(df.columns)

    # ——— Standardize & initial filter ———
    raw_smiles = []
    initial_targets = []
    initial_indices = []
    for label, row in df.iterrows():
        smi = row[smiles_col]
        std = smi
        if standardize_smiles:
            mol = Chem.MolFromSmiles(smi)
            std = Chem.MolToSmiles(mol) if mol else None
        if std is None:
            continue

        if target_cols:
            t = row[target_cols].to_numpy(dtype=np.float32, na_value=np.nan)
            if np.isnan(t).any():
                continue
            initial_targets.append(t)

        raw_smiles.append(std)
        initial_indices.append(label)

    # clear Hydra if needed
    gh = GlobalHydra.instance()
    if gh.is_initialized():
        gh.clear()

    # ——— Batched featurization with divide‐and‐conquer skip ———
    def _featurize_with_skips(featurizer, smiles_list, positions):
        try:
            feats = featurizer(smiles_list)
            return feats, smiles_list, positions
        except Exception:
            if len(smiles_list) == 1:
                print(f"[precompute] dropping '{smiles_list[0]}'")
                return [], [], []
            mid = len(smiles_list) // 2
            lf, ls, lp = _featurize_with_skips(
                featurizer, smiles_list[:mid], positions[:mid]
            )
            rf, rs, rp = _featurize_with_skips(
                featurizer, smiles_list[mid:], positions[mid:]
            )
            return lf + rf, ls + rs, lp + rp

    featurizer = Minimol(batch_size=batch_size)
    feats, kept_smiles, kept_positions = _featurize_with_skips(
        featurizer, raw_smiles, initial_indices
    )
    if not feats:
        raise RuntimeError("No SMILES could be featurized.")

    features_tensor = torch.stack([f.float().detach() for f in feats])

    # ——— Align & stack targets ———
    targets_tensor = None
    if target_cols and initial_targets:
        # map original df‐indices to their target arrays
        target_map = {idx: t for idx, t in zip(initial_indices, initial_targets)}
        final_targets = [target_map[pos] for pos in kept_positions]
        targets_tensor = torch.tensor(np.stack(final_targets), dtype=torch.float32)

    # ——— Write out ———
    torch.save(features_tensor.cpu(), feat_path)
    if targets_tensor is not None:
        torch.save(targets_tensor.cpu(), os.path.join(output_dir, "targets.pt"))

    # subset metadata to only successfully featurized rows and save CSV
    meta_df = df.loc[kept_positions, metadata_cols]
    meta_csv = os.path.join(output_dir, "metadata.csv")
    meta_df.to_csv(meta_csv, index=False)

    # write the small JSON with config
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump({
            "smiles_col": smiles_col,
            "target_cols": target_cols,
            "metadata_cols": metadata_cols
        }, f, indent=2)

    print(
        f"[precompute] wrote features.pt, "
        f"{'targets.pt, ' if targets_tensor is not None else ''}"
        f"metadata.csv, meta.json → {output_dir}"
    )
    return feat_path, meta_csv


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
        self.features = torch.load(features_path, map_location='cpu')
        self.targets = torch.load(targets_path, map_location='cpu') if targets_path and os.path.exists(targets_path) else None
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
            persistent_workers=self.num_workers > 0,
            pin_memory=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.num_workers > 0,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.num_workers > 0,
            drop_last=True,

        )