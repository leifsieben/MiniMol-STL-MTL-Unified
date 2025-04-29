# minimol_predict.py
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from minimol import Minimol
from minimol_dataset import standardize
from minimol_models import STL_FFN, MTL_FFN
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from hydra.core.global_hydra import GlobalHydra

def load_model(
    checkpoint_path: str,
    mode: str = 'mtl',
    device: str = None
) -> torch.nn.Module:
    """
    Load a trained STL or MTL model from either a Lightning checkpoint (.ckpt)
    or a raw state_dict (.pt). Falls back smoothly if the file lacks PL metadata.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ModelCls = MTL_FFN if mode == 'mtl' else STL_FFN

    try:
        # 1) Try full Lightning checkpoint
        model = ModelCls.load_from_checkpoint(checkpoint_path, map_location=device)

    except (KeyError, AttributeError):
        # 2) Fallback: load raw state_dict (*.pt)
        state = torch.load(checkpoint_path, map_location=device)

        # Infer input_dim from the first 2D weight tensor we find
        first_w = next(v for k, v in state.items() if v.ndim == 2 and 'weight' in k)
        input_dim = first_w.shape[1]

        # Collect all 1D tensors (bias vectors), ignoring batchnorm stats
        bias_cands = [
            v for k, v in state.items()
            if v.ndim == 1 and 'running' not in k and 'num_batches_tracked' not in k
        ]

        # Build the model with the right signature
        if mode == 'mtl':
            # last bias vector’s length = number of tasks
            output_dim = bias_cands[-1].shape[0]
            model = ModelCls(input_dim=input_dim, output_dim=output_dim)
        elif mode == 'stl':
            model = ModelCls(input_dim=input_dim)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # allow missing/unexpected keys when loading old checkpoints
        model.load_state_dict(state, strict=False)

    # Final touches
    model.to(device)
    model.eval()
    return model


def predict_smiles(
    smiles: str,
    checkpoint_path: str,
    mode: str = 'mtl',
    device: str = None,
    batch_size: int = 1
) -> np.ndarray:
    """
    Predict score(s) for a single SMILES string.

    Args:
      smiles: raw SMILES string
      checkpoint_path: path to trained model checkpoint
      mode: 'stl' or 'mtl'
      device: torch device
      batch_size: featurizer batch size (ignored for single)

    Returns:
      numpy array of probabilities (shape () for STL, (T,) for MTL)
    """
    # Standardize
    std = standardize(smiles)
    if std is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
        # Clear any existing Hydra initialization to avoid errors
    # clear any prior Hydra initialization so we can safely re-init
    gh = GlobalHydra.instance()
    if gh.is_initialized():
        gh.clear()
    featurizer = Minimol(batch_size=batch_size)    
    feats = featurizer([std])
    x = torch.stack([f.clone().detach().float() for f in feats])

    # Load model
    model = load_model(checkpoint_path, mode=mode, device=device)
    # Predict
    with torch.no_grad():
        x = x.to(next(model.parameters()).device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs[0] if probs.ndim > 0 else probs


def predict_df(
    df: pd.DataFrame,
    smiles_col: str,
    checkpoint_paths,
    mode: str = 'mtl',
    aggregated: bool = False,
    batch_size: int = 10000,
    device: str = None
) -> pd.DataFrame:
    """
    Add prediction columns for each model in an ensemble (or aggregated).

    Args:
      df: input DataFrame containing SMILES
      smiles_col: name of column in df
      checkpoint_paths: str or list of str paths to checkpoints
      mode: 'stl' or 'mtl'
      aggregated: if True, return mean/std instead of per-model
      batch_size: featurizer batch size
      device: torch device

    Returns:
      new DataFrame with prediction columns appended
    """
    # Ensure list of checkpoints
    if isinstance(checkpoint_paths, str):
        paths = [checkpoint_paths]
    else:
        paths = list(checkpoint_paths)

    # Standardize SMILES
    smiles = df[smiles_col].tolist()
    std_list = [standardize(s) for s in smiles]
    smiles = df[smiles_col].tolist()
    with ProcessPoolExecutor() as pool:
        std_list = list(pool.map(standardize, smiles)) 

    valid_idx = [i for i, s in enumerate(std_list) if s is not None]
    std_valid = [std_list[i] for i in valid_idx]

    # Featurize
    gh = GlobalHydra.instance()
    if gh.is_initialized():
        gh.clear()
    featurizer = Minimol(batch_size=batch_size)    
    feats_list = featurizer(std_valid)
    X = torch.stack([f.clone().detach().float() for f in feats_list])

    # Predict for each model
    all_preds = []
    for ckpt in paths:
        model = load_model(ckpt, mode=mode, device=device)
        with torch.no_grad():
            Xd = X.to(next(model.parameters()).device)
            logits = model(Xd)
            probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(probs)

    # helper to do one-model inference
    def _pred_for_ckpt(ckpt):
        m = load_model(ckpt, mode=mode, device=device)
        with torch.no_grad():
            Xd = X.to(next(m.parameters()).device)
            logits = m(Xd)
            return torch.sigmoid(logits).cpu().numpy()

        # Parallelize across the ensemble
    with ThreadPoolExecutor(max_workers=len(paths)) as pool:
            all_preds = list(pool.map(_pred_for_ckpt, paths))
 
    preds = np.stack(all_preds, axis=0)  # shape (M, N, T) or (M, N)
    n_models, n_valid = preds.shape[0], preds.shape[1]
    # tasks dimension
    T = preds.shape[2] if preds.ndim == 3 else 1

    out = df.copy()
    # allocate columns
    if aggregated:
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        # for each task
        for t in range(T):
            col_mean = f"pred_mean_task_{t}"
            col_std = f"pred_std_task_{t}"
            out[col_mean] = np.nan
            out[col_std] = np.nan
            for j, idx in enumerate(valid_idx):
                out.at[idx, col_mean] = mean[j, t] if T>1 else mean[j]
                out.at[idx, col_std] = std[j, t] if T>1 else std[j]
    else:
        # per model per task
        for m in range(n_models):
            for t in range(T):
                col = f"pred_model_{m}_task_{t}"
                out[col] = np.nan
                for j, idx in enumerate(valid_idx):
                    val = preds[m, j, t] if T>1 else preds[m, j]
                    out.at[idx, col] = val
    return out


def predict_batch(
    smiles_list,
    checkpoint_paths,
    mode: str = 'mtl',
    aggregated: bool = False,
    batch_size: int = 10000,
    device: str = None,
    output_csv: str = None
):
    """
    Fast inference on a list of SMILES. Returns DataFrame or saves to CSV.

    Args:
      smiles_list: list of SMILES strings
      checkpoint_paths: str or list of checkpoint paths
      mode: 'stl' or 'mtl'
      aggregated: whether to compute mean/std
      batch_size: featurizer batch size
      device: torch device
      output_csv: if given, path to save CSV

    Returns:
      DataFrame with SMILES and predictions
    """
    df = pd.DataFrame({ 'SMILES': smiles_list })
    result = predict_df(
        df, 'SMILES', checkpoint_paths,
        mode=mode, aggregated=aggregated,
        batch_size=batch_size, device=device
    )
    if output_csv:
        result.to_csv(output_csv, index=False)
    return result
