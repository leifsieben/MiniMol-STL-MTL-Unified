import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import optuna
from sklearn.metrics import roc_auc_score, average_precision_score
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback as _OptunaPruningCallback
import pytorch_lightning as pl
from pathlib import Path

# wrap so Lightning truly sees it as a pl.Callback subclass
class PyTorchLightningPruningCallback(_OptunaPruningCallback, pl.Callback):
    def __init__(self, trial, monitor):
        super().__init__(trial, monitor)
from minimol_dataset import PrecomputedDataModule, precompute_features
from minimol_models import STL_FFN, MTL_FFN


def compute_val_metrics(model, dataloader, metric_type: str):
    model.eval()
    all_preds, all_targs = [], []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to(model.device)
            logits = model(x)
            preds = torch.sigmoid(logits).cpu()
            all_preds.append(preds)
            all_targs.append(batch['y'])
    preds = torch.cat(all_preds).numpy()
    targs = torch.cat(all_targs).numpy()
    num_tasks = targs.shape[1] if targs.ndim > 1 else 1
    results = []
    for t in range(num_tasks):
        mask = targs[:,t] >= 0
        if mask.sum() > 0:
            y_true = targs[mask,t]
            y_pred = preds[mask,t]
            if metric_type == 'auroc':
                results.append(roc_auc_score(y_true, y_pred))
            else:
                results.append(average_precision_score(y_true, y_pred))
        else:
            results.append(float('nan'))
    return results


class MultiTaskMetricCallback(pl.Callback):
    def __init__(self, metric_type: str, monitor_tasks: list, val_dataloader):
        """
        val_dataloader: the DataLoader to use for per-task metric computation
        """
        self.metric_type   = metric_type
        self.monitor_tasks = monitor_tasks
        self.val_dataloader = val_dataloader

    def on_validation_epoch_end(self, trainer, pl_module):
        # compute per-task AUROC or AUPRC
        metrics = compute_val_metrics(pl_module, self.val_dataloader, self.metric_type)
        for t in self.monitor_tasks:
            name = f"val_{self.metric_type}_task_{t}"
            # log to Lightning
            pl_module.log(
                name,
                metrics[t],
                on_epoch=True,      # aggregate across the epoch
                prog_bar=True,      # show in progress bar
                logger=False         # log metric for Lightning (and Optuna if using pruning)
            )


def train_model(
    config,
    train_dir: str,
    val_dir: str,
    test_dir: str,
    ckpt_root: str,  
    mode: str = 'mtl',
    max_epochs: int = 10,
    num_workers: int = 4,
    checkpoint_path: str = None,
    return_metric: bool = False,
    early_stop: bool = False,
    early_stop_patience: int = 3,
    monitor_metric: str = 'loss',
    monitor_task: int = 0,
    ensemble_size: int = 1,
    ensemble_idx: int = 0,
    seed: int = 777, 
    task_weights: list = None,
):
    # seed differently for each ensemble member
    if ensemble_size > 1:
        pl.seed_everything(seed + ensemble_idx)
    if ensemble_size < 1:
        print("Warning: ensemble_size can't be smaller than 1, setting to 1")
        ensemble_size =1
    if num_workers < 0:
        raise ValueError("num_workers must be ≥ 0")
    if early_stop_patience < 1:
        print("Warning: early_stop_patience can't be smaller than 1, setting to 1")
        early_stop_patience = 1
    if task_weights is not None:
        if mode != 'mtl':
            print("Warning: task_weights provided but mode is not 'mtl'. Ignoring weights.")
            task_weights = None
        else:
            targs = torch.load(os.path.join(train_dir, 'targets.pt'))
            num_tasks = targs.shape[1]
            if len(task_weights) != num_tasks:
                raise ValueError(f"Number of task weights ({len(task_weights)}) does not match "
                                f"number of tasks ({num_tasks})")
            print(f"Using task weights: {task_weights}")

    dm = PrecomputedDataModule(
        train_dir, val_dir, test_dir,
        batch_size=config['batch_size'], num_workers=num_workers
    )
    dm.setup()

    feats = torch.load(os.path.join(train_dir, 'features.pt'))
    targs = torch.load(os.path.join(train_dir, 'targets.pt'))
    input_dim = feats.shape[1]
    output_dim = targs.shape[1] if mode == 'mtl' else 1

    ModelCls = MTL_FFN if mode == 'mtl' else STL_FFN
    kwargs = {k: v for k, v in config.items() if k in ModelCls.__init__.__code__.co_varnames}
    if mode == 'mtl':
        kwargs['output_dim'] = output_dim
        if task_weights is not None:
            kwargs['task_weights'] = task_weights
    model = ModelCls(input_dim=input_dim, **kwargs)

    if checkpoint_path:
        model = model.load_from_checkpoint(checkpoint_path, strict=False)

    callbacks = []
    # metric callbacks (only for MTL, pass in the val loader)
    if mode == 'mtl' and monitor_metric in ['auroc', 'auprc']:
        val_loader = dm.val_dataloader()
        callbacks.append(
            MultiTaskMetricCallback(
                metric_type   = monitor_metric,
                monitor_tasks = [monitor_task],
                val_dataloader= val_loader
            )
        )

    # define monitor name and mode
    monitor_name = (
        'val_loss' if monitor_metric == 'loss'
        else f"val_{monitor_metric}_task_{monitor_task}"
    )
    mode_minmax = 'min' if monitor_metric == 'loss' else 'max'

    # checkpoint directory and filename
    run_id = f"ensemble_{ensemble_idx}" if ensemble_size > 1 else "single_run"
    ckpt_dir = Path(ckpt_root) / run_id

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{run_id}-" + "{epoch}-{" + monitor_name + ":.4f}"

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=filename,
        monitor=monitor_name,
        save_top_k=1,
        save_last=True, 
        mode=mode_minmax
    )
    callbacks.append(ckpt_cb)

    if early_stop:
        callbacks.append(
            EarlyStopping(
                monitor=monitor_name,
                patience=early_stop_patience,
                mode=mode_minmax
            )
        )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=False,
        default_root_dir=str(ckpt_dir), 
        resume_from_checkpoint=None
    )

    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    test_loader = dm.test_dataloader()
    trainer.test(model, dataloaders=test_loader)

    if return_metric:
        if ckpt_cb.best_model_score is None:
            print("Warning: No checkpoint saved, returning fallback.")
            return float("inf") if monitor_metric == "loss" else float("-inf")
        return ckpt_cb.best_model_score.item()

    return ckpt_cb.best_model_path

def run_hyperopt(
    train_dir, val_dir, test_dir,
    mode, max_epochs, num_workers,
    monitor_metric, monitor_task,
    early_stop_patience,
    n_trials, n_jobs,
    storage_url, study_name="minimol_study"
):
    direction = "minimize" if monitor_metric == "loss" else "maximize"
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )

    if num_workers < 1:
        print("Warning: num_workers can't be smaller than 1, setting to 1")
        num_workers = 1
    if early_stop_patience < 1:
        print("Warning: early_stop_patience can't be smaller than 1, setting to 1")
        early_stop_patience = 1

    def _objective(trial):
        cfg = {
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
            test_dir=test_dir,
            ckpt_root="./optuna_checkpoints",
            mode=mode,
            max_epochs=max_epochs,
            num_workers=num_workers,
            early_stop=True,
            early_stop_patience=early_stop_patience,
            monitor_metric=monitor_metric,
            monitor_task=monitor_task,
            ensemble_size=1,
            ensemble_idx=0,
            return_metric=True
        )
        return best_metric

    study.optimize(_objective, n_trials=n_trials, n_jobs=n_jobs)
    print("Best params:", study.best_trial.params)
    return study.best_trial.params


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Precompute, train, or hyperopt a MiniMol‐FFN model"
    )
    # core directories & mode
    parser.add_argument('--mode',        choices=['stl','mtl'], required=True)
    parser.add_argument('--train_dir',   required=True)
    parser.add_argument('--val_dir',     required=True)
    parser.add_argument('--test_dir',    required=True)

    # precompute
    parser.add_argument('--precompute',      action='store_true')
    parser.add_argument('--input_tsv')
    parser.add_argument('--target_cols',     nargs='+')
    parser.add_argument('--metadata_cols',   nargs='+')
    parser.add_argument('--output_dir')

    # checkpoint (for resume in non‐hyperopt/train)
    parser.add_argument('--checkpoint_path')

    # training settings
    parser.add_argument('--max_epochs',       type=int, default=10)
    parser.add_argument('--num_workers',      type=int, default=4)
    parser.add_argument('--early_stop',       action='store_true')
    parser.add_argument('--early_stop_patience', type=int, default=3)
    parser.add_argument('--monitor_metric',   choices=['loss','auroc','auprc'],
                        default='loss')
    parser.add_argument('--monitor_task',     type=int, default=0)
    parser.add_argument('--ensemble',         type=int, default=1)
    parser.add_argument('--task_weights', type=float, nargs='+', 
                        help="Weights for each task in MTL mode (must match number of tasks)")
    
    # hyperopt (Optuna)
    parser.add_argument('--hyperopt',         action='store_true',
                        help="run Optuna hyperparameter search")
    parser.add_argument('--n_trials',         type=int, default=20,
                        help="number of Optuna trials")
    parser.add_argument('--n_jobs',           type=int, default=1,
                        help="parallel jobs for Optuna.study.optimize")
    parser.add_argument('--optuna_storage',   type=str,
                        default='sqlite:///optuna.db',
                        help="Optuna storage URL (e.g. sqlite or postgresql)")

    # manual hyperparams (for non‐hyperopt train)
    parser.add_argument('--dim_size',          type=int,   default=128)
    parser.add_argument('--shrinking_scale',   type=float, default=0.5)
    parser.add_argument('--num_layers',        type=int,   default=3)
    parser.add_argument('--learning_rate',     type=float, default=1e-3)
    parser.add_argument('--batch_size',        type=int,   default=32)
    parser.add_argument('--dropout_rate',      type=float, default=0.0)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--use_batch_norm',    action='store_true')
    parser.add_argument('--use_residual',      action='store_true')
    parser.add_argument('--L1_weight_norm',    type=float, default=0.0)
    parser.add_argument('--L2_weight_norm',    type=float, default=0.0)
    parser.add_argument('--scheduler_step_size', type=int, default=5)
    parser.add_argument('--scheduler_gamma',   type=float, default=0.5)
    parser.add_argument('--loss_type',         type=str, default='bce_with_logits')

    args = parser.parse_args()

    if args.precompute:
        precompute_features(
            args.input_tsv,
            args.target_cols,
            args.output_dir,
            metadata_cols=args.metadata_cols
        )

    elif args.hyperopt:
        best_params = run_hyperopt(
            train_dir           = args.train_dir,
            val_dir             = args.val_dir,
            test_dir            = args.test_dir,
            mode                = args.mode,
            max_epochs          = args.max_epochs,
            num_workers         = args.num_workers,
            monitor_metric      = args.monitor_metric,
            monitor_task        = args.monitor_task,
            early_stop_patience = args.early_stop_patience,
            n_trials            = args.n_trials,
            n_jobs              = args.n_jobs,
            storage_url         = args.optuna_storage,
        )
        print("Optuna best hyperparameters:", best_params)

    else:
        # assemble config for plain training
        cfg = {
            'dim_size':            args.dim_size,
            'shrinking_scale':     args.shrinking_scale,
            'num_layers':          args.num_layers,
            'learning_rate':       args.learning_rate,
            'batch_size':          args.batch_size,
            'dropout_rate':        args.dropout_rate,
            'activation_function': args.activation_function,
            'use_batch_norm':      args.use_batch_norm,
            'use_residual':        args.use_residual,
            'L1_weight_norm':      args.L1_weight_norm,
            'L2_weight_norm':      args.L2_weight_norm,
            'scheduler_step_size': args.scheduler_step_size,
            'scheduler_gamma':     args.scheduler_gamma,
            'loss_type':           args.loss_type,
        }

        if args.ensemble > 1:
            for idx in range(args.ensemble):
                train_model(
                    config                = cfg,
                    train_dir             = args.train_dir,
                    val_dir               = args.val_dir,
                    test_dir              = args.test_dir,
                    mode                  = args.mode,
                    max_epochs            = args.max_epochs,
                    num_workers           = args.num_workers,
                    checkpoint_path       = args.checkpoint_path,
                    early_stop            = args.early_stop,
                    early_stop_patience   = args.early_stop_patience,
                    monitor_metric        = args.monitor_metric,
                    monitor_task          = args.monitor_task,
                    ensemble_size         = args.ensemble,
                    ensemble_idx          = idx
                )
        else:
            train_model(
                config              = cfg,
                train_dir           = args.train_dir,
                val_dir             = args.val_dir,
                test_dir            = args.test_dir,
                mode                = args.mode,
                max_epochs          = args.max_epochs,
                num_workers         = args.num_workers,
                checkpoint_path     = args.checkpoint_path,
                early_stop          = args.early_stop,
                early_stop_patience = args.early_stop_patience,
                monitor_metric      = args.monitor_metric,
                monitor_task        = args.monitor_task,
                task_weights        =args.task_weights
            )

if __name__ == "__main__":
    main()
