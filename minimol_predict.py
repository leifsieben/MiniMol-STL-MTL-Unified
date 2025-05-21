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
from minimol_models import TaskHeadMTL, TaskHeadSTL

def load_model(
    checkpoint_path: str,
    mode: str = 'mtl',
    device: str = None,
    architecture: str = 'standard'  # New parameter to specify architecture type
) -> torch.nn.Module:
    """
    Load a trained model from a checkpoint.
    
    :param checkpoint_path: Path to the checkpoint file
    :param mode: 'mtl' for multi-task or 'stl' for single-task model
    :param device: Device to load the model on ('cuda' or 'cpu')
    :param architecture: 'standard' for original architecture, 'residual' for colleague's architecture
    :return: Loaded model
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Select model class based on mode and architecture
    if architecture == 'standard':
        ModelCls = MTL_FFN if mode == 'mtl' else STL_FFN
    elif architecture == 'residual':
        ModelCls = TaskHeadMTL if mode == 'mtl' else TaskHeadSTL
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    try:
        # 1) Try full Lightning checkpoint
        model = ModelCls.load_from_checkpoint(checkpoint_path, map_location=device)

    except (KeyError, AttributeError):
        # 2) Fallback: load raw state_dict (*.pt)
        state = torch.load(checkpoint_path, map_location=device)

        # Infer input_dim from the first 2D weight tensor we find
        first_w = next(v for k, v in state.items() if v.ndim == 2 and 'weight' in k)
        input_dim = first_w.shape[1]

        # Use an appropriate way to determine output_dim based on architecture
        if architecture == 'standard':
            # Original approach for standard models
            bias_cands = [
                v for k, v in state.items()
                if v.ndim == 1 and 'running' not in k and 'num_batches_tracked' not in k
            ]
            output_dim = bias_cands[-1].shape[0] if mode == 'mtl' else 1
            
            # Build the model with the right signature
            if mode == 'mtl':
                model = ModelCls(input_dim=input_dim, output_dim=output_dim)
            else:
                model = ModelCls(input_dim=input_dim)
                
        elif architecture == 'residual':
            # For residual architecture, find the final_dense layer
            if 'final_dense.weight' in state:
                final_dense_weight = state['final_dense.weight']
                output_dim = final_dense_weight.shape[0]
                hidden_dim = state['layers.0.weight'].shape[0]  # First layer's output dim
                
                # Count number of layers
                num_layers = len([k for k in state.keys() if k.startswith('layers.') and k.endswith('.weight')])
                
                # Create model with inferred parameters
                if mode == 'mtl':
                    model = ModelCls(input_dim=input_dim, output_dim=output_dim, 
                                    hidden_dim=hidden_dim, num_layers=num_layers)
                else:
                    model = ModelCls(input_dim=input_dim, hidden_dim=hidden_dim, 
                                    num_layers=num_layers)
            else:
                raise ValueError("Cannot infer model structure from checkpoint - no final_dense.weight found")
                
        # Allow missing/unexpected keys when loading old checkpoints
        model.load_state_dict(state, strict=False)

    # Final touches
    model.to(device)
    model.eval()
    return model

def _ensemble_predict(
    X: torch.Tensor,
    checkpoints: Sequence[str],
    mode: str,
    device: Optional[str] = None,
    architecture: str = 'standard',
    task_of_interest: Optional[int] = None
) -> np.ndarray:
    """
    Given a feature tensor X, load each checkpoint, run model(X), and return predictions.
    
    :param architecture: 'standard' for original architecture, 'residual' for colleague's
    """
    def single_pred(ckpt):
        m = load_model(ckpt, mode=mode, device=device, architecture=architecture)
        with torch.no_grad():
            out = m(X.to(next(m.parameters()).device))
            probs = torch.sigmoid(out).cpu().numpy()
        return probs

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(checkpoints)) as pool:
        all_preds = pool.map(single_pred, checkpoints)
        preds = np.stack(list(all_preds), axis=0)  # (M, N) or (M, N, T)
        if mode == 'mtl' and preds.ndim == 3 and task_of_interest is not None:
            if task_of_interest >= preds.shape[2]:
                raise IndexError(f"task_of_interest={task_of_interest} out of range for {preds.shape[2]} tasks.")
            preds = preds[:, :, task_of_interest:task_of_interest+1]  # keep dim for consistent downstream

    return preds


def predict_smiles(
    smiles: str,
    checkpoints,                 # str or list
    mode: str = 'mtl',
    device: str = None,
    batch_size: int = 1,
    aggregated: bool = True,     
    architecture: str = 'standard', 
    task_of_interest: Optional[int] = None
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

    preds = _ensemble_predict(X, paths, mode, device, architecture=architecture, task_of_interest=task_of_interest)  # shape (M, 1) or (M,1,T)

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
    device: str = None,
    architecture: str = 'standard',
    task_of_interest: Optional[int] = None
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

    preds = _ensemble_predict(X, paths, mode, device, architecture=architecture, task_of_interest=task_of_interest)
    M, N = preds.shape[0], preds.shape[1]
    T = preds.shape[2] if preds.ndim==3 else 1

    out = df.copy()
    if aggregated:
        mean = preds.mean(axis=0)   # (N,T) or (N,)
        std  = preds.std(axis=0)
        task_indices = [task_of_interest] if task_of_interest is not None else range(T)
        for t in task_indices:
            cm, cs = f"pred_mean_task_{t}", f"pred_std_task_{t}"
            out[cm], out[cs] = np.nan, np.nan
            for j, idx in enumerate(valid_idx):
                out.at[idx, cm] = mean[j,t] if T>1 else mean[j]
                out.at[idx, cs] = std[j,t]  if T>1 else std[j]
    else:
        for m in range(M):
            task_indices = [task_of_interest] if task_of_interest is not None else range(T)
            for t in task_indices:
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
    output_csv: str = None,
    architecture: str = 'standard', 
    task_of_interest: Optional[int] = None
):
    df = pd.DataFrame({'SMILES': smiles_list})
    res = predict_df(
        df, 'SMILES',
        checkpoints      = checkpoints,
        mode             = mode,
        aggregated       = aggregated,
        batch_size       = batch_size,
        device           = device,
        architecture     = architecture, 
        task_of_interest = task_of_interest
    )
    if output_csv:
        res.to_csv(output_csv, index=False)
    return res

def predict_on_precomputed(
    X_feat: torch.Tensor,
    checkpoints,  # str or sequence of str
    mode: str = 'mtl',
    species_indices: Optional[Sequence[int]] = None,
    device: Optional[str] = None,
    include_individual_models: bool = False,
    architecture: str = 'standard',
    task_of_interest: Optional[int] = None,  
) -> pd.DataFrame:
    """
    Make predictions using precomputed features with ensemble or single model.
    
    Parameters:
    -----------
    X_feat : torch.Tensor
        Precomputed features tensor of shape (N, D)
    checkpoints : str or sequence of str
        Path(s) to model checkpoint(s)
    mode : str
        'mtl' for multi-task or 'stl' for single-task model
    species_indices : sequence of int, optional
        Indices of specific tasks/species to extract from MTL predictions
        If None, include all available tasks
    device : str, optional
        Device to use for predictions ('cuda' or 'cpu')
        If None, will use CUDA if available
    include_individual_models : bool
        Whether to include predictions from each individual model
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns for mean and std_dev predictions
        If MTL mode, includes columns for each task
        If include_individual_models=True, includes columns for each model
    """
    # Normalize checkpoints to a list
    ckpts = checkpoints if isinstance(checkpoints, (list, tuple)) else [checkpoints]
    
    # Ensure all checkpoints exist
    missing = [c for c in ckpts if not os.path.isfile(c)]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint files: {', '.join(missing)}")

    # If task_of_interest is set, override species_indices
    if task_of_interest is not None:
        if mode != 'mtl':
            raise ValueError("task_of_interest is only supported in 'mtl' mode.")
        species_indices = [task_of_interest]

    # Get predictions using _ensemble_predict
    preds = _ensemble_predict(X_feat, ckpts, mode=mode, device=device, architecture=architecture)
    
    result_df = pd.DataFrame()
    
    # Handle different prediction shapes based on mode
    if mode == 'mtl':
        if len(preds.shape) == 3:
            if task_of_interest is not None:
                max_task_idx = preds.shape[2] - 1
                if task_of_interest > max_task_idx:
                    raise IndexError(f"task_of_interest={task_of_interest} is out of bounds for model with {max_task_idx + 1} tasks.")

            # Shape [M, N, T]
            num_models, num_samples, num_tasks = preds.shape
            
            # Determine which tasks/species to include
            if species_indices is None:
                species_indices = list(range(num_tasks))
            
            # Map common species indices to names
            species_map = {0: "AB", 1: "EC", 2: "KP", 3: "PA"}
            
            # Process each requested task/species
            for idx in species_indices:
                if idx < num_tasks:
                    # Get name (if in map) or use task_X notation
                    name = species_map.get(idx, f"task_{idx}")
                    
                    # Extract predictions for this task
                    task_preds = preds[:, :, idx]  # Shape [M, N]
                    
                    # Calculate stats across ensemble members
                    result_df[f"{name}_mean"] = np.mean(task_preds, axis=0)
                    result_df[f"{name}_std_dev"] = np.std(task_preds, axis=0)
                    
                    # Add individual model predictions if requested
                    if include_individual_models:
                        for m in range(num_models):
                            result_df[f"{name}_model_{m}"] = task_preds[m, :]
                else:
                    raise IndexError(f"Species index {idx} out of range (max {num_tasks-1})")
            
            # Calculate cross-species metrics if multiple species
            if len(species_indices) > 1:
                # Collect mean predictions across specified tasks
                species_means = np.stack([
                    result_df[f"{species_map.get(idx, f'task_{idx}')}_mean"].values
                    for idx in species_indices
                ], axis=0)
                
                # Calculate aggregate metrics
                result_df["max_across_species"] = np.max(species_means, axis=0)
                result_df["mean_across_species"] = np.mean(species_means, axis=0)
                
        elif len(preds.shape) == 2:
            # Handle single-task MTL model, shape [M, N]
            num_models, num_samples = preds.shape
            
            result_df["mean"] = np.mean(preds, axis=0)
            result_df["std_dev"] = np.std(preds, axis=0)
            
            # Add individual model predictions if requested
            if include_individual_models:
                for m in range(num_models):
                    result_df[f"model_{m}"] = preds[m, :]
        
    elif mode == 'stl':
        # For STL, shape should be [M, N]
        if len(preds.shape) == 2:
            num_models, num_samples = preds.shape
            
            # Calculate stats across ensemble members
            result_df["mean"] = np.mean(preds, axis=0)
            result_df["std_dev"] = np.std(preds, axis=0)
            
            # Add individual model predictions if requested
            if include_individual_models:
                for m in range(num_models):
                    result_df[f"model_{m}"] = preds[m, :]
        else:
            raise ValueError(f"Unexpected prediction shape for STL mode: {preds.shape}")
    
    return result_df


def predict_precomputed_file(
    precomputed_features_path: str,
    metadata_csv_path: str,
    checkpoints,  # str or sequence of str
    output_path: Optional[str] = None,
    mode: str = 'mtl',
    species_indices: Optional[Sequence[int]] = None,
    device: Optional[str] = None,
    include_individual_models: bool = False,
    smiles_col: str = "SMILES",
    architecture: str = 'standard',
    task_of_interest: Optional[int] = None 
) -> pd.DataFrame:
    """
    Make predictions on precomputed features stored in a file, and optionally save results.
    
    A convenience wrapper around predict_on_precomputed that:
    1. Loads precomputed features from a file
    2. Loads metadata from a CSV
    3. Makes predictions
    4. Combines predictions with SMILES from metadata
    5. Optionally saves to a CSV file
    
    Parameters:
    -----------
    precomputed_features_path : str
        Path to the precomputed features .pt file
    metadata_csv_path : str
        Path to the metadata CSV containing SMILES strings
    checkpoints : str or sequence of str
        Path(s) to model checkpoint(s)
    output_path : str, optional
        Path to save the predictions CSV
        If None, results will not be saved to a file
    mode : str
        'mtl' for multi-task or 'stl' for single-task model
    species_indices : sequence of int, optional
        Indices of specific tasks/species to extract from MTL predictions
    device : str, optional
        Device to use for predictions ('cuda' or 'cpu')
    include_individual_models : bool
        Whether to include predictions from each individual model
    smiles_col : str
        Name of the SMILES column in the metadata CSV
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with SMILES and predictions
    """
    # Load precomputed features
    X_feat = torch.load(precomputed_features_path, map_location="cpu")
    
    # Load metadata
    meta_df = pd.read_csv(metadata_csv_path)
    
    # Check dimensions
    if len(meta_df) != X_feat.shape[0]:
        raise ValueError(
            f"Number of samples in metadata ({len(meta_df)}) does not match "
            f"number of samples in features ({X_feat.shape[0]})"
        )
    
    # Get predictions
    result_df = predict_on_precomputed(
        X_feat, 
        checkpoints, 
        mode=mode,
        species_indices=species_indices,
        device=device,
        include_individual_models=include_individual_models,
        architecture=architecture, 
        task_of_interest=task_of_interest
    )
    
    # Combine with SMILES
    result_df.insert(0, smiles_col, meta_df[smiles_col].values)
    
    # Save if output path provided
    if output_path:
        result_df.to_csv(output_path, index=False)
    
    return result_df