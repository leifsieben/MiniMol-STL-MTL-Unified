import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray
from sklearn.metrics import roc_auc_score, average_precision_score

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
    def __init__(self, metric_type: str, monitor_tasks: list):
        self.metric_type = metric_type
        self.monitor_tasks = monitor_tasks

    def on_validation_epoch_end(self, trainer, pl_module):
        # compute per-task AUROC or AUPRC
        metrics = compute_val_metrics(pl_module, trainer.datamodule.val_dataloader(), self.metric_type)
        for t in self.monitor_tasks:
            name = f"val_{self.metric_type}_task_{t}"
            # log to Lightning
            pl_module.log(
                name,
                metrics[t],
                on_epoch=True,      # aggregate across the epoch
                prog_bar=True,      # show in progress bar
                logger=True         # send to logger → Ray Tune will pick it up
            )


def train_model(
    config,
    train_dir: str,
    val_dir: str,
    test_dir: str,
    mode: str = 'mtl',
    max_epochs: int = 10,
    num_workers: int = 4,
    checkpoint_path: str = None,
    early_stop: bool = False,
    early_stop_patience: int = 3,
    monitor_metric: str = 'loss',  # 'loss', 'auroc', or 'auprc'
    monitor_task: int = 0,
    ensemble_size: int = 1,
    ensemble_idx: int = 0,
    seed: int = 777, 
    report_to_ray: bool = False
):
    # seed differently for each ensemble member
    if ensemble_size > 1:
        pl.seed_everything(seed + ensemble_idx)

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
    model = ModelCls(input_dim=input_dim, **kwargs)

    if checkpoint_path:
        model = model.load_from_checkpoint(checkpoint_path, strict=False)

    callbacks = []
    # metric callbacks
    if monitor_metric in ['auroc', 'auprc']:
        callbacks.append(MultiTaskMetricCallback(monitor_metric, [monitor_task]))

    # define monitor name and mode
    monitor_name = (
        'val_loss' if monitor_metric == 'loss'
        else f"val_{monitor_metric}_task_{monitor_task}"
    )
    mode_minmax = 'min' if monitor_metric == 'loss' else 'max'

    # checkpoint directory and filename
    if ensemble_size > 1:
        ckpt_dir = os.path.join(os.getcwd(), 'checkpoints', f'ensemble_{ensemble_idx}')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_dir = os.getcwd()
        filename = f'ensemble{ensemble_idx}-' + "{epoch}-{" + monitor_name + ":.4f}"
    else:
        ckpt_dir = None
        ckpt_dir = os.getcwd()
        filename = "{epoch}-{" + monitor_name + ":.4f}"

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=filename,
        monitor=monitor_name,
        save_top_k=1,
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
    )

    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    if report_to_ray:
        # best_model_score is a torch scalar
        best_val = ckpt_cb.best_model_score
        # convert to Python float
        ray.tune.report(**{monitor_name: best_val.item()})

    test_loader = dm.test_dataloader()
    trainer.test(model, dataloaders=test_loader)

def run_hyperopt(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    mode: str = 'mtl',
    max_epochs: int = 10,
    num_workers: int = 4,
    checkpoint_path: str = None,
    early_stop: bool = True,
    early_stop_patience: int = 3,
    monitor_metric: str = 'loss',
    monitor_task: int = 0,
    hyperopt_num_samples: int = 20,
    cpus_per_trial: int = 1,
    gpus_per_trial: int = 0,
    tune_dir: str = './ray_results',
):
    # same as before; ensemble not applied here
    search_space = {
        'dim_size': tune.lograndint(128, 4096),
        'shrinking_scale': tune.uniform(0.5, 1.0),
        'num_layers': tune.randint(1, 15),
        'learning_rate': tune.loguniform(1e-6, 1e-1),
        'batch_size': tune.choice([32, 64, 128, 256]),
        'dropout_rate': tune.uniform(0.0, 0.9),
        'activation_function': tune.choice(['relu', 'tanh', 'leaky_relu', 'gelu']),
        'use_batch_norm': tune.choice([True, False]),
        'use_residual': tune.choice([True, False]),
        'L1_weight_norm': tune.loguniform(1e-12, 1e-2),
        'L2_weight_norm': tune.loguniform(1e-8, 1e-2),
        'scheduler_step_size': tune.randint(1, 20),
        'scheduler_gamma': tune.uniform(0.1, 0.9),
        'loss_type': tune.choice(['bce_with_logits']),
    }
    scheduler = ASHAScheduler(
        metric=('val_loss' if monitor_metric == 'loss'
                else f"val_{monitor_metric}_task_{monitor_task}"),
        mode=('min' if monitor_metric == 'loss' else 'max'),
        max_t=max_epochs,
        grace_period=1,
        reduction_factor=2
    )
    analysis = tune.run(
        tune.with_parameters(
            train_model,
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            mode=mode,
            max_epochs=max_epochs,
            num_workers=num_workers,
            checkpoint_path=checkpoint_path,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience,
            monitor_metric=monitor_metric,
            monitor_task=monitor_task,
            report_to_ray=True
        ),
        config=search_space,
        num_samples=hyperopt_num_samples,
        scheduler=scheduler,
        resources_per_trial={'cpu': cpus_per_trial, 'gpu': gpus_per_trial},
        local_dir=tune_dir,
        metric=('val_loss' if monitor_metric == 'loss'
                else f"val_{monitor_metric}_task_{monitor_task}"),
        mode=('min' if monitor_metric == 'loss' else 'max')
    )
    best = analysis.get_best_config(
        metric=('val_loss' if monitor_metric == 'loss'
                else f"val_{monitor_metric}_task_{monitor_task}"),
        mode=('min' if monitor_metric == 'loss' else 'max')
    )
    print("Best hyperparameters:", best)
    return best


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['stl', 'mtl'], required=True)
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--val_dir', required=True)
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--precompute', action='store_true')
    parser.add_argument('--input_tsv')
    parser.add_argument('--fingerprint_col')
    parser.add_argument('--target_cols', nargs='+')
    parser.add_argument('--metadata_cols', nargs='+')
    parser.add_argument('--output_dir')
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--hyperopt', action='store_true')
    parser.add_argument('--hyperopt_num_samples', type=int, default=20)
    parser.add_argument('--cpus_per_trial', type=int, default=1)
    parser.add_argument('--gpus_per_trial', type=int, default=0)
    parser.add_argument('--tune_dir', type=str, default='./ray_results')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--early_stop_patience', type=int, default=3)
    parser.add_argument('--monitor_metric', choices=['loss', 'auroc', 'auprc'], default='loss')
    parser.add_argument('--monitor_task', type=int, default=0)
    parser.add_argument('--ensemble', type=int, default=1)
    # hyperparameters for non-hyperopt runs
    parser.add_argument('--dim_size', type=int, default=128)
    parser.add_argument('--shrinking_scale', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--use_batch_norm', action='store_true')
    parser.add_argument('--use_residual', action='store_true')
    parser.add_argument('--L1_weight_norm', type=float, default=0.0)
    parser.add_argument('--L2_weight_norm', type=float, default=0.0)
    parser.add_argument('--scheduler_step_size', type=int, default=5)
    parser.add_argument('--scheduler_gamma', type=float, default=0.5)
    parser.add_argument('--loss_type', type=str, default='bce_with_logits')
    args = parser.parse_args()

    if args.precompute:
        precompute_features(
            args.input_tsv,
            args.fingerprint_col,
            args.target_cols,
            args.output_dir,
            metadata_cols=args.metadata_cols
        )
    else:
        if args.hyperopt:
            run_hyperopt(
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                test_dir=args.test_dir,
                mode=args.mode,
                max_epochs=args.max_epochs,
                num_workers=args.num_workers,
                checkpoint_path=args.checkpoint_path,
                early_stop=args.early_stop,
                early_stop_patience=args.early_stop_patience,
                monitor_metric=args.monitor_metric,
                monitor_task=args.monitor_task,
                hyperopt_num_samples=args.hyperopt_num_samples,
                cpus_per_trial=args.cpus_per_trial,
                gpus_per_trial=args.gpus_per_trial,
                tune_dir=args.tune_dir
            )
        else:
            cfg = {
                'dim_size': args.dim_size,
                'shrinking_scale': args.shrinking_scale,
                'num_layers': args.num_layers,
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size,
                'dropout_rate': args.dropout_rate,
                'activation_function': args.activation_function,
                'use_batch_norm': args.use_batch_norm,
                'use_residual': args.use_residual,
                'L1_weight_norm': args.L1_weight_norm,
                'L2_weight_norm': args.L2_weight_norm,
                'scheduler_step_size': args.scheduler_step_size,
                'scheduler_gamma': args.scheduler_gamma,
                'loss_type': args.loss_type,
            }
            if args.ensemble > 1:
                for idx in range(args.ensemble):
                    train_model(
                        cfg,
                        args.train_dir,
                        args.val_dir,
                        args.test_dir,
                        mode=args.mode,
                        max_epochs=args.max_epochs,
                        num_workers=args.num_workers,
                        checkpoint_path=args.checkpoint_path,
                        early_stop=args.early_stop,
                        early_stop_patience=args.early_stop_patience,
                        monitor_metric=args.monitor_metric,
                        monitor_task=args.monitor_task,
                        ensemble_size=args.ensemble,
                        ensemble_idx=idx
                    )
            else:
                train_model(
                    cfg,
                    args.train_dir,
                    args.val_dir,
                    args.test_dir,
                    mode=args.mode,
                    max_epochs=args.max_epochs,
                    num_workers=args.num_workers,
                    checkpoint_path=args.checkpoint_path,
                    early_stop=args.early_stop,
                    early_stop_patience=args.early_stop_patience,
                    monitor_metric=args.monitor_metric,
                    monitor_task=args.monitor_task
                )
