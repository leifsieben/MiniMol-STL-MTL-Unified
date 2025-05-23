import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import optuna
from sklearn.metrics import roc_auc_score, average_precision_score
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback as _OptunaPruningCallback
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional, Dict, Any

# --- Lightning pruning wrapper ------------------------------------------------
class PyTorchLightningPruningCallback(_OptunaPruningCallback, pl.Callback):
    """Wrap Optuna's pruning callback so Lightning recognises it."""

    def __init__(self, trial, monitor):
        super().__init__(trial, monitor)


# --- MiniMol utils ------------------------------------------------------------
from minimol_dataset import PrecomputedDataModule, precompute_features
from minimol_models import (
    STL_FFN,
    MTL_FFN,
    TaskHeadSTL,
    TaskHeadMTL,
)

# -----------------------------------------------------------------------------
#                              METRIC HELPERS
# -----------------------------------------------------------------------------

def compute_val_metrics(model, dataloader, metric_type: str):
    """Compute per-task AUROC/AUPRC on an arbitrary DataLoader."""
    model.eval()
    all_preds, all_targs = [], []
    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(model.device)
            logits = model(x)
            preds = torch.sigmoid(logits).cpu()
            all_preds.append(preds)
            all_targs.append(batch["y"])

    preds = torch.cat(all_preds).numpy()
    targs = torch.cat(all_targs).numpy()
    num_tasks = targs.shape[1] if targs.ndim > 1 else 1
    results = []
    for t in range(num_tasks):
        mask = targs[:, t] >= 0
        if mask.sum() > 0:
            y_true = targs[mask, t]
            y_pred = preds[mask, t]
            if metric_type == "auroc":
                results.append(roc_auc_score(y_true, y_pred))
            else:
                results.append(average_precision_score(y_true, y_pred))
        else:
            results.append(float("nan"))
    return results


class MultiTaskMetricCallback(pl.Callback):
    """Log *per‑task* AUROC/AUPRC so we can monitor & checkpoint on them."""

    def __init__(self, metric_type: str, monitor_tasks: list[int], val_dataloader):
        self.metric_type = metric_type
        self.monitor_tasks = monitor_tasks
        self.val_dataloader = val_dataloader

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = compute_val_metrics(pl_module, self.val_dataloader, self.metric_type)
        for t in self.monitor_tasks:
            name = f"val_{self.metric_type}_task_{t}"
            pl_module.log(name, metrics[t], on_epoch=True, prog_bar=True, logger=False)


class SingleTaskMetricCallback(pl.Callback):
    """Same as above, but for STL models."""

    def __init__(self, metric_type: str, val_dataloader):
        self.metric_type = metric_type
        self.val_dataloader = val_dataloader

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        all_preds, all_targs = [], []
        with torch.no_grad():
            for batch in self.val_dataloader:
                x = batch["x"].to(pl_module.device)
                y = batch["y"].to(pl_module.device)
                logits = pl_module(x)
                preds = torch.sigmoid(logits).cpu()
                all_preds.append(preds)
                all_targs.append(y.cpu())

        preds = torch.cat(all_preds).numpy().flatten()
        targs = torch.cat(all_targs).numpy().flatten()

        score = (
            roc_auc_score(targs, preds)
            if self.metric_type == "auroc"
            else average_precision_score(targs, preds)
        )
        name = f"val_{self.metric_type}"
        trainer.logger.log_metrics({name: score}, step=trainer.global_step)
        pl_module.log(name, score, on_epoch=True, prog_bar=True, logger=True)


# -----------------------------------------------------------------------------
#                                TRAINING
# -----------------------------------------------------------------------------

def _select_model_class(mode: str, architecture: str):
    """Return the correct (STL|MTL) × (standard|residual) class."""
    if architecture == "standard":
        return STL_FFN if mode == "stl" else MTL_FFN
    elif architecture == "residual":
        return TaskHeadSTL if mode == "stl" else TaskHeadMTL
    else:
        raise ValueError(f"Unknown architecture '{architecture}'.")


def _prepare_model_kwargs(model_cls, mode: str, config: Dict[str, Any], *, input_dim: int, output_dim: int):
    """Filter + map `config` so it matches the signature of `model_cls`."""
    sig_params = model_cls.__init__.__code__.co_varnames

    # Provide common key translations across architectures
    cfg = config.copy()
    if "dim_size" in cfg and "hidden_dim" not in cfg and "hidden_dim" in sig_params:
        # residual TaskHead uses `hidden_dim`
        cfg["hidden_dim"] = cfg.pop("dim_size")

    # Finally pick the intersection
    kwargs = {k: v for k, v in cfg.items() if k in sig_params}

    # Mandatory dims
    kwargs["input_dim"] = input_dim
    if mode == "mtl":
        kwargs["output_dim"] = output_dim
    return kwargs


def train_model(
    config: Dict[str, Any],
    train_dir: str,
    val_dir: str,
    ckpt_root: str,
    *,
    architecture: str = "standard",  # NEW ▹ "standard" | "residual"
    mode: str = "mtl",  # "stl" | "mtl"
    max_epochs: int = 10,
    num_workers: int = 4,
    checkpoint_path: str | None = None,
    return_metric: bool = False,
    early_stop: bool = False,
    early_stop_patience: int = 3,
    monitor_metric: str = "loss",
    val_split_ratio: float = 0.2,
    split_seed: int = 777,
    monitor_task: int = 0,
    ensemble_size: int = 1,
    ensemble_idx: int = 0,
    seed: int = 777,
    task_weights: list[float] | None = None,
    test_dir: Optional[str] = None,
):
    """Train a single model (STL/MTL, standard/residual)."""

    # -------------------------- sanity checks & housekeeping -----------------
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be ≥ 1")
    if num_workers < 0:
        raise ValueError("num_workers must be ≥ 0")
    if early_stop_patience < 1:
        early_stop_patience = 1
    # Warn + ignore task_weights if in STL mode
    if task_weights is not None and mode != "mtl":
        print("Warning: task_weights provided but mode is not 'mtl'. Ignoring weights.")
        task_weights = None

    # Seed (unique per ensemble member)
    pl.seed_everything(seed + ensemble_idx)

    # -------------------------- data ----------------------------------------
    dm = PrecomputedDataModule(
        train_dir=train_dir,
        val_dir=val_dir if val_dir else None,
        test_dir=test_dir if test_dir else None,
        batch_size=config["batch_size"],
        num_workers=num_workers,
        val_split_ratio=val_split_ratio if val_dir is None else None,
        split_seed=split_seed,
    )
    dm.setup()

    feats = torch.load(os.path.join(train_dir, "features.pt"))
    targs = torch.load(os.path.join(train_dir, "targets.pt"))
    input_dim = feats.shape[1]
    output_dim = targs.shape[1] if mode == "mtl" else 1

    # -------------------------- model ---------------------------------------
    ModelCls = _select_model_class(mode, architecture)
    model_kwargs = _prepare_model_kwargs(ModelCls, mode, config, input_dim=input_dim, output_dim=output_dim)

    # Task weights are only valid for MTL
    if mode == "mtl" and task_weights is not None:
        model_kwargs["task_weights"] = task_weights

    model = ModelCls(**model_kwargs)

    if checkpoint_path:
        model = model.load_from_checkpoint(checkpoint_path, strict=False)

    # -------------------------- callbacks -----------------------------------
    callbacks = []

    # Metric callbacks – only compute expensive metrics if we monitor them
    val_loader_for_metrics = dm.val_dataloader()
    if monitor_metric in ["auroc", "auprc"]:
        if mode == "mtl":
            callbacks.append(
                MultiTaskMetricCallback(monitor_metric, [monitor_task], val_loader_for_metrics)
            )
        else:
            callbacks.append(SingleTaskMetricCallback(monitor_metric, val_loader_for_metrics))

    # Checkpointing
    monitor_name = "val_loss" if monitor_metric == "loss" else f"val_{monitor_metric}_task_{monitor_task}"
    mode_minmax = "min" if monitor_metric == "loss" else "max"

    run_id = f"ensemble_{ensemble_idx}" if ensemble_size > 1 else "single_run"
    ckpt_dir = Path(ckpt_root) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"{run_id}-" + "{epoch}-{" + monitor_name + ":.4f}",
        monitor=monitor_name,
        save_top_k=1,
        save_last=True,
        mode=mode_minmax,
    )
    callbacks.append(ckpt_cb)

    # Early stopping
    if early_stop:
        callbacks.append(EarlyStopping(monitor=monitor_name, patience=early_stop_patience, mode=mode_minmax))

    # -------------------------- TRAIN ---------------------------------------
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=False,
        default_root_dir=str(ckpt_dir),
    )

    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader(), ckpt_path=None)

    # -------------------------- return --------------------------------------
    if return_metric:
        if ckpt_cb.best_model_score is None:
            fallback = float("inf") if monitor_metric == "loss" else float("-inf")
            return fallback
        return ckpt_cb.best_model_score.item()
    return ckpt_cb.best_model_path


# -----------------------------------------------------------------------------
#                               HYPER‑OPT
# -----------------------------------------------------------------------------

def run_hyperopt(
    train_dir: str,
    *,
    architecture: str = "standard",
    val_dir: Optional[str] = None,
    mode: str = "mtl",
    max_epochs: int = 10,
    num_workers: int = 4,
    monitor_metric: str = "loss",
    monitor_task: int = 0,
    early_stop_patience: int = 5,
    n_trials: int = 50,
    n_jobs: int = 1,
    storage_url: str = "sqlite:///optuna.db",
    study_name: str = "minimol_study",
):
    """Optuna hyper‑parameter search (for *either* architecture)."""
    direction = "minimize" if monitor_metric == "loss" else "maximize"
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )

    def _objective(trial):
        cfg = {
            # common to both architectures (extra keys are filtered later)
            "dim_size": trial.suggest_int("dim_size", 128, 4096, log=True),
            "shrinking_scale": trial.suggest_float("shrinking_scale", 0.5, 1.0),
            "num_layers": trial.suggest_int("num_layers", 1, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.9),
            "activation_function": trial.suggest_categorical(
                "activation_function", ["relu", "tanh", "leaky_relu", "gelu"]
            ),
            "use_batch_norm": trial.suggest_categorical("use_batch_norm", [True, False]),
            "use_residual": trial.suggest_categorical("use_residual", [True, False]),
            "L1_weight_norm": trial.suggest_float("L1_weight_norm", 1e-12, 1e-2, log=True),
            "L2_weight_norm": trial.suggest_float("L2_weight_norm", 1e-8, 1e-2, log=True),
            "scheduler_step_size": trial.suggest_int("scheduler_step_size", 1, 20),
            "scheduler_gamma": trial.suggest_float("scheduler_gamma", 0.1, 0.9),
            "loss_type": "bce_with_logits",
        }

        best_metric = train_model(
            config=cfg,
            train_dir=train_dir,
            val_dir=val_dir,
            ckpt_root="./optuna_checkpoints",
            architecture=architecture,
            mode=mode,
            max_epochs=max_epochs,
            num_workers=num_workers,
            early_stop=True,
            early_stop_patience=early_stop_patience,
            monitor_metric=monitor_metric,
            monitor_task=monitor_task,
            split_seed=trial.number + 42,
            val_split_ratio=0.2,
            ensemble_size=1,
            ensemble_idx=0,
            return_metric=True,
        )
        return best_metric

    study.optimize(_objective, n_trials=n_trials, n_jobs=n_jobs)
    print("Best params:", study.best_trial.params)
    return study.best_trial.params


# -----------------------------------------------------------------------------
#                                   CLI
# -----------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser("Precompute, train, or hyper‑opt a MiniMol model")

    # Core directories & mode -------------------------------------------------
    parser.add_argument("--mode", choices=["stl", "mtl"], required=True)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir")
    parser.add_argument("--test_dir")

    # Architecture -----------------------------------------------------------
    parser.add_argument(
        "--architecture",
        choices=["standard", "residual"],
        default="standard",
        help="Backbone: 'standard'=FFN, 'residual'=TaskHead residual concat.",
    )

    # Pre‑compute features ----------------------------------------------------
    parser.add_argument("--precompute", action="store_true")
    parser.add_argument("--input_tsv")
    parser.add_argument("--target_cols", nargs="+")
    parser.add_argument("--metadata_cols", nargs="+")
    parser.add_argument("--output_dir")

    # Resume / fine‑tune ------------------------------------------------------
    parser.add_argument("--checkpoint_path")

    # Training settings -------------------------------------------------------
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--monitor_metric", choices=["loss", "auroc", "auprc"], default="loss")
    parser.add_argument("--monitor_task", type=int, default=0)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--task_weights", type=float, nargs="+", help="Weights for each task (MTL only)")

    # Hyper‑opt ---------------------------------------------------------------
    parser.add_argument("--hyperopt", action="store_true", help="Run Optuna search")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--optuna_storage", type=str, default="sqlite:///optuna.db")

    # Manual hyper‑params (shared between architectures) ----------------------
    parser.add_argument("--dim_size", type=int, default=128)
    parser.add_argument("--shrinking_scale", type=float, default=0.5)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--use_batch_norm", action="store_true")
    parser.add_argument("--use_residual", action="store_true")
    parser.add_argument("--L1_weight_norm", type=float, default=0.0)
    parser.add_argument("--L2_weight_norm", type=float, default=0.0)
    parser.add_argument("--scheduler_step_size", type=int, default=5)
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--loss_type", type=str, default="bce_with_logits")

    args = parser.parse_args()

    # PRE‑COMPUTE -------------------------------------------------------------
    if args.precompute:
        precompute_features(
            args.input_tsv,
            args.target_cols,
            args.output_dir,
            metadata_cols=args.metadata_cols,
        )
        return

    # Assemble config shared across architectures ----------------------------
    cfg: Dict[str, Any] = {
        "dim_size": args.dim_size,
        "shrinking_scale": args.shrinking_scale,
        "num_layers": args.num_layers,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "dropout_rate": args.dropout_rate,
        "activation_function": args.activation_function,
        "use_batch_norm": args.use_batch_norm,
        "use_residual": args.use_residual,
        "L1_weight_norm": args.L1_weight_norm,
        "L2_weight_norm": args.L2_weight_norm,
        "scheduler_step_size": args.scheduler_step_size,
        "scheduler_gamma": args.scheduler_gamma,
        "loss_type": args.loss_type,
    }

    # HYPER‑OPT ---------------------------------------------------------------
    if args.hyperopt:
        best_params = run_hyperopt(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            architecture=args.architecture,
            mode=args.mode,
            max_epochs=args.max_epochs,
            num_workers=args.num_workers,
            monitor_metric=args.monitor_metric,
            monitor_task=args.monitor_task,
            early_stop_patience=args.early_stop_patience,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            storage_url=args.optuna_storage,
        )
        print("Optuna best hyper‑parameters:", best_params)
        return

    # PLAIN TRAIN -------------------------------------------------------------
    if args.ensemble > 1:
        for idx in range(args.ensemble):
            train_model(
                config=cfg,
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                test_dir=args.test_dir,
                ckpt_root=args.train_dir,  # save alongside data by default
                architecture=args.architecture,
                mode=args.mode,
                max_epochs=args.max_epochs,
                num_workers=args.num_workers,
                checkpoint_path=args.checkpoint_path,
                early_stop=args.early_stop,
                early_stop_patience=args.early_stop_patience,
                monitor_metric=args.monitor_metric,
                monitor_task=args.monitor_task,
                ensemble_size=args.ensemble,
                ensemble_idx=idx,
                task_weights=args.task_weights,
            )
    else:
        train_model(
            config=cfg,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            test_dir=args.test_dir,
            ckpt_root=args.train_dir,
            architecture=args.architecture,
            mode=args.mode,
            max_epochs=args.max_epochs,
            num_workers=args.num_workers,
            checkpoint_path=args.checkpoint_path,
            early_stop=args.early_stop,
            early_stop_patience=args.early_stop_patience,
            monitor_metric=args.monitor_metric,
            monitor_task=args.monitor_task,
            task_weights=args.task_weights,
        )


if __name__ == "__main__":
    main()
