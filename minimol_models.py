import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Dict, List, Optional

def _build_shared_layers(input_dim, dim_size, shrinking_scale, num_layers, act_fn, use_bn, dropout, use_res):
    layers = nn.ModuleList()
    in_dim = input_dim
    curr_dim = dim_size
    for _ in range(num_layers):
        seq = [nn.Linear(in_dim, curr_dim)]
        if use_bn: seq.append(nn.BatchNorm1d(curr_dim))
        seq.append(act_fn)
        if dropout > 0: seq.append(nn.Dropout(dropout))
        layers.append(nn.Sequential(*seq))
        in_dim, curr_dim = curr_dim, max(1, int(curr_dim * shrinking_scale))
    return layers


def _forward_shared(x, layers, use_res):
    out = x
    for l in layers:
        nxt = l(out)
        out = out + nxt if use_res and nxt.shape == out.shape else nxt
    return out


class BaseFFN(pl.LightningModule):
    """Base class for feed-forward networks with common functionality"""
    def __init__(
        self, 
        learning_rate: float = 1e-3, 
        batch_size: int = 32,
        L2_weight_norm: float = 0.0, 
        L1_weight_norm: float = 0.0,
        scheduler_step_size: int = 5, 
        scheduler_gamma: float = 0.5,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.L2_weight_norm = L2_weight_norm
        self.L1_weight_norm = L1_weight_norm
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        
    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.L2_weight_norm
        )
        sch = torch.optim.lr_scheduler.StepLR(
            opt, 
            step_size=self.scheduler_step_size, 
            gamma=self.scheduler_gamma
        )
        return [opt], [{'scheduler': sch, 'interval': 'step'}]


class STL_FFN(BaseFFN):
    def __init__(
        self, input_dim: int,
        dim_size: int = 128, shrinking_scale: float = 0.5, num_layers: int = 3,
        learning_rate: float = 1e-3, batch_size: int = 32,
        L2_weight_norm: float = 0.0, L1_weight_norm: float = 0.0,
        activation_function: str = 'relu', dropout_rate: float = 0.0,
        use_batch_norm: bool = False, use_residual: bool = False,
        scheduler_step_size: int = 5, scheduler_gamma: float = 0.5,
        loss_type: str = 'bce_with_logits', 
    ):
        super().__init__(
            learning_rate, batch_size, L2_weight_norm, L1_weight_norm,
            scheduler_step_size, scheduler_gamma
        )
        self.save_hyperparameters()
        acts = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'leaky_relu': nn.LeakyReLU()}
        act = acts.get(activation_function, nn.ReLU())
        self.layers = _build_shared_layers(
            input_dim, dim_size, shrinking_scale,
            num_layers, act, use_batch_norm,
            dropout_rate, use_residual
        )
        final_dim = self.layers[-1][0].out_features
        self.head = nn.Linear(final_dim, 1)
        self.use_residual = use_residual
        self.L1_weight_norm = L1_weight_norm

    def forward(self, x):
        out = _forward_shared(x, self.layers, self.use_residual)
        return self.head(out).squeeze(-1)

    def _step(self, batch, stage):
        x = batch['x']
        # pull targets onto the right device and dtype
        y = batch['y'].to(self.device).float()
        # if y is (batch,1) squeeze to (batch,)
        if y.ndim == 2 and y.size(1) == 1:
            y = y.squeeze(1)
        # if somehow you ended up with a scalar, bump it to a 1-element batch
        elif y.ndim == 0:
            y = y.unsqueeze(0)        

        logits = self(x)
        if logits.ndim == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        # your forward already does .squeeze(-1), so logits should be (batch,)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        if self.L1_weight_norm > 0:
            loss += self.L1_weight_norm * sum(p.abs().sum() for p in self.parameters())
        # for both train and val, log per-epoch; show prog_bar only on val
        batch_size = x.size(0)
        self.log(
            f"{stage}_loss",
            loss,
            batch_size=batch_size,
            on_epoch=True,
            prog_bar=(stage == "val"),
            on_step=False
        )
        return loss

    def training_step(self, b, i): return self._step(b, 'train')
    def validation_step(self, b, i): return self._step(b, 'val')
    def test_step(self, b, i): return self._step(b, 'test')

def compute_weighted_task_loss(preds, y, task_weights=None, l1_weight_norm=0, parameters=None):
    """
    Utility function to compute weighted loss across tasks.
    
    Args:
        preds: Model predictions (logits)
        y: Target values with -1 indicating missing labels
        task_weights: Optional list of weights for each task
        l1_weight_norm: L1 regularization coefficient
        parameters: Model parameters for L1 regularization
        
    Returns:
        Weighted average loss across all valid tasks
    """
    device = preds.device
    total_loss = torch.tensor(0.0, device=device)
    scale_sum = 0.0
    output_dim = preds.shape[1]
    
    for t in range(output_dim):
        # Mask out unknown labels
        mask = (y[:, t] >= 0.0)
        if mask.any():
            preds_t = preds[mask, t]
            y_t = y[mask, t].float()
            
            loss_t = F.binary_cross_entropy_with_logits(preds_t, y_t, reduction='mean')
            
            # Get weight for this task
            weight = 1.0  # Default weight
            if task_weights is not None and t < len(task_weights):
                weight = task_weights[t]
            
            total_loss = total_loss + weight * loss_t
            scale_sum += weight
    
    if scale_sum > 0:
        total_loss = total_loss / scale_sum
    else:
        # If there are no valid labels in this batch, produce 0 but keep gradient
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Add L1 regularization if needed
    if l1_weight_norm > 0 and parameters is not None:
        total_loss += l1_weight_norm * sum(p.abs().sum() for p in parameters)
    
    return total_loss

class MTL_FFN(BaseFFN):
    def __init__(
        self, input_dim: int, output_dim: int,
        dim_size: int = 128, shrinking_scale: float = 0.5, num_layers: int = 3,
        learning_rate: float = 1e-3, batch_size: int = 32,
        L2_weight_norm: float = 0.0, L1_weight_norm: float = 0.0,
        activation_function: str = 'relu', dropout_rate: float = 0.0,
        use_batch_norm: bool = False, use_residual: bool = False,
        scheduler_step_size: int = 5, scheduler_gamma: float = 0.5,
        loss_type: str = 'bce_with_logits', task_weights: list = None,
    ):
        super().__init__(
            learning_rate, batch_size, L2_weight_norm, L1_weight_norm,
            scheduler_step_size, scheduler_gamma
        )
        self.save_hyperparameters()
        acts = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'leaky_relu': nn.LeakyReLU()}
        act = acts.get(activation_function, nn.ReLU())
        self.layers = _build_shared_layers(
            input_dim, dim_size, shrinking_scale,
            num_layers, act, use_batch_norm,
            dropout_rate, use_residual
        )
        final_dim = self.layers[-1][0].out_features
        self.head = nn.Linear(final_dim, output_dim)
        self.use_residual = use_residual
        self.output_dim = output_dim
        self.L1_weight_norm = L1_weight_norm

    def forward(self, x): 
        return self.head(_forward_shared(x, self.layers, self.use_residual))

    def _compute_loss(self, preds, y):
        return compute_weighted_task_loss(
            preds=preds, 
            y=y, 
            task_weights=self.task_weights if hasattr(self, 'task_weights') else None,
            l1_weight_norm=self.L1_weight_norm,
            parameters=self.parameters()
        )

    def training_step(self, b, i): 
        x, y = b['x'], b['y']
        batch_size = x.size(0)
        loss = self._compute_loss(self(x), y)
        self.log('train_loss', loss, batch_size=batch_size)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        loss = self._compute_loss(self(x), y)
        self.log('val_loss', loss, batch_size=x.size(0), on_epoch=True, prog_bar=True)
        
    def test_step(self, b, i): 
        x, y = b['x'], b['y']
        batch_size = x.size(0)
        loss = self._compute_loss(self(x), y)
        self.log('test_loss', loss, batch_size=batch_size)


class BaseTaskHead(BaseFFN):
    """Base class for TaskHead models with residual connections"""
    def __init__(
        self, input_dim: int, 
        hidden_dim: int = 128, 
        num_layers: int = 2, 
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3, 
        batch_size: int = 32,
        L2_weight_norm: float = 0.0, 
        L1_weight_norm: float = 0.0,
        scheduler_step_size: int = 5, 
        scheduler_gamma: float = 0.5,
    ):
        super().__init__(
            learning_rate, batch_size, L2_weight_norm, L1_weight_norm,
            scheduler_step_size, scheduler_gamma
        )
        self.save_hyperparameters()
        
        # Create layers with batch normalization
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout_rate)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
    def _process_features(self, x):
        """Process input through hidden layers with BatchNorm, ReLU, and Dropout"""
        original_x = x
        hidden = x
        
        # Process through hidden layers
        for dense, bn in zip(self.layers, self.batch_norms):
            hidden = dense(hidden)
            hidden = bn(hidden)
            hidden = F.relu(hidden)
            hidden = self.dropout(hidden)
        
        # Concatenate original input with processed features
        concatenated = torch.cat((original_x, hidden), dim=1)
        return concatenated


class TaskHeadSTL(BaseTaskHead):
    """STL model with direct residual connection"""
    def __init__(
        self, input_dim: int,
        hidden_dim: int = 128, 
        num_layers: int = 2, 
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3, 
        batch_size: int = 32,
        L2_weight_norm: float = 0.0, 
        L1_weight_norm: float = 0.0,
        scheduler_step_size: int = 5, 
        scheduler_gamma: float = 0.5,
        task_weights: list = None,
    ):
        super().__init__(
            input_dim, hidden_dim, num_layers, dropout_rate,
            learning_rate, batch_size, L2_weight_norm, L1_weight_norm,
            scheduler_step_size, scheduler_gamma
        )
        # Final layer takes concatenated input and hidden
        self.final_dense = nn.Linear(input_dim + hidden_dim, 1)
        
    def forward(self, x):
        concatenated = self._process_features(x)
        output = self.final_dense(concatenated).squeeze(-1)
        return output
    
    # Reuse the _step method from STL_FFN with a wrapper
    def _step(self, batch, stage):
        x = batch['x']
        y = batch['y'].to(self.device).float()
        if y.ndim == 2 and y.size(1) == 1:
            y = y.squeeze(1)
        elif y.ndim == 0:
            y = y.unsqueeze(0)        

        logits = self(x)
        if logits.ndim == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)

        loss = F.binary_cross_entropy_with_logits(logits, y)
        if self.hparams.L1_weight_norm > 0:
            loss += self.hparams.L1_weight_norm * sum(p.abs().sum() for p in self.parameters())

        batch_size = x.size(0)
        self.log(
            f"{stage}_loss", loss, batch_size=batch_size,
            on_epoch=True, prog_bar=(stage == "val"), on_step=False
        )
        return loss

    def training_step(self, b, i): return self._step(b, 'train')
    def validation_step(self, b, i): return self._step(b, 'val')
    def test_step(self, b, i): return self._step(b, 'test')


class TaskHeadMTL(BaseTaskHead):
    """MTL model with direct residual connection"""
    def __init__(
        self, input_dim: int, output_dim: int,
        hidden_dim: int = 128, 
        num_layers: int = 2, 
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3, 
        batch_size: int = 32,
        L2_weight_norm: float = 0.0, 
        L1_weight_norm: float = 0.0,
        scheduler_step_size: int = 5, 
        scheduler_gamma: float = 0.5,
    ):
        super().__init__(
            input_dim, hidden_dim, num_layers, dropout_rate,
            learning_rate, batch_size, L2_weight_norm, L1_weight_norm,
            scheduler_step_size, scheduler_gamma
        )
        # Final layer takes concatenated input and hidden
        self.final_dense = nn.Linear(input_dim + hidden_dim, output_dim)
        self.output_dim = output_dim
        
    def forward(self, x):
        concatenated = self._process_features(x)
        output = self.final_dense(concatenated)
        return output
    
    def _compute_loss(self, preds, y):
        return compute_weighted_task_loss(
            preds=preds, 
            y=y, 
            task_weights=self.task_weights if hasattr(self, 'task_weights') else None,
            l1_weight_norm=self.hparams.L1_weight_norm,
            parameters=self.parameters()
        )

    def training_step(self, b, i): 
        x, y = b['x'], b['y']
        batch_size = x.size(0)
        loss = self._compute_loss(self(x), y)
        self.log('train_loss', loss, batch_size=batch_size)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        loss = self._compute_loss(self(x), y)
        self.log('val_loss', loss, batch_size=x.size(0), on_epoch=True, prog_bar=True)
        
    def test_step(self, b, i): 
        x, y = b['x'], b['y']
        batch_size = x.size(0)
        loss = self._compute_loss(self(x), y)
        self.log('test_loss', loss, batch_size=batch_size)