import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class STL_FFN(pl.LightningModule):
    def __init__(
        self, input_dim: int,
        dim_size: int=128, shrinking_scale: float=0.5, num_layers: int=3,
        learning_rate: float=1e-3, batch_size: int=32,
        L2_weight_norm: float=0.0, L1_weight_norm: float=0.0,
        activation_function: str='relu', dropout_rate: float=0.0,
        use_batch_norm: bool=False, use_residual: bool=False,
        scheduler_step_size: int=5, scheduler_gamma: float=0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        acts = {'relu':nn.ReLU(), 'tanh':nn.Tanh(), 'leaky_relu':nn.LeakyReLU()}
        act = acts.get(self.hparams.activation_function, nn.ReLU())
        self.layers = _build_shared_layers(
            input_dim, self.hparams.dim_size, self.hparams.shrinking_scale,
            self.hparams.num_layers, act, self.hparams.use_batch_norm,
            self.hparams.dropout_rate, self.hparams.use_residual
        )
        final_dim = self.layers[-1][0].out_features
        self.head = nn.Linear(final_dim, 1)

    def forward(self, x):
        out = _forward_shared(x, self.layers, self.hparams.use_residual)
        return self.head(out).squeeze(-1)

    def _step(self, batch, stage):
        x = batch['x']
        # pull targets on to the right device and dtype…
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

        if self.hparams.L1_weight_norm>0:
            loss += self.hparams.L1_weight_norm * sum(p.abs().sum() for p in self.parameters())
        # for both train and val, log per‐epoch; show prog_bar only on val
        self.log(
            f"{stage}_loss",
            loss,
            on_epoch=True,
            prog_bar=(stage == "val"),
            on_step=False
        )
        return loss

    def training_step(self, b, i): return self._step(b, 'train')
    def validation_step(self, b, i): return self._step(b, 'val')
    def test_step(self, b, i): return self._step(b, 'test')

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.L2_weight_norm)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=self.hparams.scheduler_step_size, gamma=self.hparams.scheduler_gamma)
        return [opt], [{'scheduler':sch,'interval':'step'}]


class MTL_FFN(pl.LightningModule):
    def __init__(
        self, input_dim: int, output_dim: int,
        dim_size: int=128, shrinking_scale: float=0.5, num_layers: int=3,
        learning_rate: float=1e-3, batch_size: int=32,
        L2_weight_norm: float=0.0, L1_weight_norm: float=0.0,
        activation_function: str='relu', dropout_rate: float=0.0,
        use_batch_norm: bool=False, use_residual: bool=False,
        scheduler_step_size: int=5, scheduler_gamma: float=0.5,
        loss_type: str='bce_with_logits'
    ):
        super().__init__()
        self.save_hyperparameters()
        acts = {'relu':nn.ReLU(), 'tanh':nn.Tanh(), 'leaky_relu':nn.LeakyReLU()}
        act = acts.get(self.hparams.activation_function, nn.ReLU())
        self.layers = _build_shared_layers(
            input_dim, self.hparams.dim_size, self.hparams.shrinking_scale,
            self.hparams.num_layers, act, self.hparams.use_batch_norm,
            self.hparams.dropout_rate, self.hparams.use_residual
        )
        final_dim = self.layers[-1][0].out_features
        self.head = nn.Linear(final_dim, self.hparams.output_dim)

    def forward(self, x): return self.head(_forward_shared(x, self.layers, self.hparams.use_residual))

    def _compute_loss(self, preds, y):
        total, cnt = 0, 0
        for t in range(self.hparams.output_dim):
            mask = y[:,t]>=0
            if mask.any():
                logits = preds[mask,t]
                tgt = y[mask,t].float()
                total += F.binary_cross_entropy_with_logits(logits,tgt)
                cnt += 1
        loss = total/cnt if cnt>0 else torch.tensor(0.0,device=preds.device)
        if self.hparams.L1_weight_norm>0:
            loss += self.hparams.L1_weight_norm * sum(p.abs().sum() for p in self.parameters())
        return loss

    def training_step(self, b, i): x,y=b['x'],b['y']; loss=self._compute_loss(self(x),y); self.log('train_loss',loss); return loss
    def validation_step(self, b, i): x,y=b['x'],b['y']; loss=self._compute_loss(self(x),y); self.log('val_loss',loss,prog_bar=True)
    def test_step(self, b, i): x,y=b['x'],b['y']; loss=self._compute_loss(self(x),y); self.log('test_loss',loss)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.L2_weight_norm)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=self.hparams.scheduler_step_size, gamma=self.hparams.scheduler_gamma)
        return [opt], [{'scheduler':sch,'interval':'step'}]
