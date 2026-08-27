import warnings

import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage

from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry


class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=2e-4, help="The learning rate (2e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=1e-5, help="The minimum time (1e-5 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20, help="Number of evaluation files.")
        parser.add_argument("--loss_type", type=str, default="mse", choices=("mse", "mae"), help="The type of loss function to use.")
        parser.add_argument(
            "--loss_reduction",
            type=str,
            default="mean",
            choices=("mean", "sum"),
            help=(
                "Reduce over spectrogram elements with a mean (recommended) or "
                "the legacy sum."
            ),
        )
        return parser

    def __init__(
        self, backbone, sde, lr=2e-4, ema_decay=0.999, t_eps=1e-5,
        num_eval_files=20, loss_type='mse', loss_reduction="mean",
        sigma_min=None, data_module=None, **kwargs
    ):
        super().__init__()

        if not 0.0 <= t_eps < 1.0:
            raise ValueError(f"t_eps must be in [0, 1), got {t_eps}")
        if loss_reduction not in ("mean", "sum"):
            raise ValueError(
                f"loss_reduction must be 'mean' or 'sum', got {loss_reduction!r}"
            )
        if isinstance(sde, str) and sde != "ot_flow":
            raise ValueError(
                "sgmse.model_fm.ScoreModel requires --sde ot_flow; "
                f"received {sde!r}."
            )

        # Keep older FM checkpoints loadable while using OTFlow as the single
        # source of truth for sigma_min in new runs.
        if sigma_min is not None and "sigma_min" not in kwargs:
            kwargs["sigma_min"] = sigma_min

        # Initialize Backbone DNN
        if isinstance(backbone, str):
            dnn_cls = BackboneRegistry.get_by_name(backbone)
            self.dnn = dnn_cls(**kwargs)
        else:
            self.dnn = backbone

        # Initialize SDE / Flow
        if isinstance(sde, str):
            sde_cls = SDERegistry.get_by_name(sde)
            self.sde = sde_cls(**kwargs)
        else:
            self.sde = sde
        if not hasattr(self.sde, "sample_path"):
            raise TypeError(
                "Flow-matching training requires an OTFlow-compatible object "
                "with a sample_path(x1, t) method."
            )

        # Store hyperparams
        self.lr = lr
        self.sigma_min = float(self.sde.sigma_min)
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.loss_reduction = loss_reduction
        self.num_eval_files = num_eval_files

        self.save_hyperparameters(ignore=['data_module'])
        self.data_module = data_module

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    def on_load_checkpoint(self, checkpoint):
        self._data_module_hparams = checkpoint.get(
            'data_module_hyper_parameters',
            checkpoint.get('datamodule_hyper_parameters', {}),
        )
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()
        if self.data_module is not None and hasattr(self.data_module, 'hparams'):
            checkpoint['data_module_hyper_parameters'] = dict(self.data_module.hparams)

    def train(self, mode, no_ema=False):
        res = super().train(mode)
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                self.ema.store(self.parameters())
                self.ema.copy_to(self.parameters())
            else:
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, err):
        if self.loss_type == 'mse':
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            losses = err.abs()
        losses = losses.reshape(losses.shape[0], -1)
        if self.loss_reduction == "mean":
            return 0.5 * losses.mean()
        return torch.mean(0.5 * torch.sum(losses, dim=-1))

    def _step(self, batch, batch_idx):
        x1, txt_emb = batch
        if x1.ndim != 4 or not torch.is_complex(x1):
            raise ValueError(
                "Flow matching expects complex spectrograms shaped [B, C, F, T], "
                f"got shape {tuple(x1.shape)} and dtype {x1.dtype}"
            )
        if txt_emb.ndim != 2 or txt_emb.shape[0] != x1.shape[0]:
            raise ValueError(
                "Text conditions must be shaped [B, D] with the same batch size "
                f"as the spectrograms; got {tuple(txt_emb.shape)}"
            )

        t = torch.rand(x1.shape[0], device=x1.device) * (1.0 - self.t_eps) + self.t_eps
        xt, target_velocity = self.sde.sample_path(x1, t)

        predicted_velocity = self(xt, t, txt_emb)
        err = predicted_velocity - target_velocity

        loss = self._loss(err)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log(
            'valid_loss',
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        return loss

    def forward(self, x, t, txt_emb):
        velocity = self.dnn(x, t, txt_emb)
        return velocity

    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)
