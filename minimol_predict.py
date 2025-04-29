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
from typing import Sequence, Optional, Union, Tuple

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

def _ensemble_predict(
    X: torch.Tensor,
    checkpoints: Sequence[str],
    mode: str,
    device: Optional[str]
) -> np.ndarray:
    """
    Given a feature tensor X of shape (N, D), load each checkpoint,
    run model(X), and return an array of shape (M, N, T) or (M, N)
    where M = len(checkpoints), T = #tasks for MTL (1 for STL).
    """
    def single_pred(ckpt):
        m = load_model(ckpt, mode=mode, device=device)
        with torch.no_grad():
            out = m(X.to(next(m.parameters()).device))
            probs = torch.sigmoid(out).cpu().numpy()
        # ensure shape (N,) or (N, T)
        return probs

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(checkpoints)) as pool:
        all_preds = pool.map(single_pred, checkpoints)
        preds = np.stack(list(all_preds), axis=0)
    return preds



def predict_smiles(
    smiles: str,
    checkpoints,                 # str or list
    mode: str = 'mtl',
    device: str = None,
    batch_size: int = 1,
    aggregated: bool = True      # new!
) -> Union[np.ndarray, Tuple[np.ndarray,np.ndarray]]:
    """
    Predict score(s) for a single SMILES string using one or more checkpoints.
    If aggregated=True, returns the mean (and optionally std if you change return
    signature). If aggregated=False, returns the full per-model array.
    """
    # normalize checkpoints to list
    paths = checkpoints if isinstance(checkpoints, (list,tuple)) else [checkpoints]

    std = standardize(smiles)
    if std is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    gh = GlobalHydra.instance()
    if gh.is_initialized(): gh.clear()

    feats = Minimol(batch_size=batch_size)([std])
    X = torch.stack([f.float().clone().detach() for f in feats])

    preds = _ensemble_predict(X, paths, mode, device)  # shape (M, 1) or (M,1,T)

    # collapse model axis
    if aggregated:
        mean = preds.mean(axis=0)  # shape (1,) or (1,T)
        return mean.squeeze(0)     # scalar or (T,)
    else:
        return preds.squeeze(1)    # shape (M,) or (M,T)




def predict_df(
    df: pd.DataFrame,
    smiles_col: str,
    checkpoints,                # str or list
    mode: str = 'mtl',
    aggregated: bool = False,
    batch_size: int = 10000,
    device: str = None
) -> pd.DataFrame:
    paths = checkpoints if isinstance(checkpoints, (list,tuple)) else [checkpoints]

    smiles = df[smiles_col].tolist()
    with ProcessPoolExecutor() as pool:
        std_list = list(pool.map(standardize, smiles))
    valid_idx = [i for i,s in enumerate(std_list) if s is not None]
    std_valid = [std_list[i] for i in valid_idx]

    gh = GlobalHydra.instance()
    if gh.is_initialized(): gh.clear()

    feats = Minimol(batch_size=batch_size)(std_valid)
    X = torch.stack([f.float().clone().detach() for f in feats])

    preds = _ensemble_predict(X, paths, mode, device)
    M, N = preds.shape[0], preds.shape[1]
    T = preds.shape[2] if preds.ndim==3 else 1

    out = df.copy()
    if aggregated:
        mean = preds.mean(axis=0)   # (N,T) or (N,)
        std  = preds.std(axis=0)
        for t in range(T):
            cm, cs = f"pred_mean_task_{t}", f"pred_std_task_{t}"
            out[cm], out[cs] = np.nan, np.nan
            for j, idx in enumerate(valid_idx):
                out.at[idx, cm] = mean[j,t] if T>1 else mean[j]
                out.at[idx, cs] = std[j,t]  if T>1 else std[j]
    else:
        for m in range(M):
            for t in range(T):
                col = f"pred_model_{m}_task_{t}"
                out[col] = np.nan
                for j, idx in enumerate(valid_idx):
                    val = preds[m,j,t] if T>1 else preds[m,j]
                    out.at[idx, col] = val

    return out


def predict_batch(
    smiles_list,
    checkpoints,
    mode: str = 'mtl',
    aggregated: bool = False,
    batch_size: int = 10000,
    device: str = None,
    output_csv: str = None
):
    df = pd.DataFrame({'SMILES': smiles_list})
    res = predict_df(
        df, 'SMILES',
        checkpoints      = checkpoints,
        mode             = mode,
        aggregated       = aggregated,
        batch_size       = batch_size,
        device           = device
    )
    if output_csv:
        res.to_csv(output_csv, index=False)
    return res
