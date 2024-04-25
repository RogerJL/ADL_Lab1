from typing import Any

import torch
from overrides import overrides
from torch import nn, Tensor
import torch.nn.functional as F
import sklearn.metrics as metrics

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


class LitVanilla(L.LightningModule):
    def __init__(self, name: str,
                 total_net: nn.Module,
                 optimizer: str, lr, weight_decay,
                 loss: str, loss_reduction: str,
                 example_input_array=None):
        super().__init__()
        self.save_hyperparameters(ignore=['total_net'])
        self.name = name
        self.total_net = total_net
        self.optimizer = optimizer
        self.example_input_array = example_input_array
        self.lr = lr
        self.wd = weight_decay
        self.loss_reduction = loss_reduction

        if loss == "bce_w_logits":
            def bcewl(y_est, y_true):
                return F.binary_cross_entropy_with_logits(y_est,
                                                          F.one_hot(y_true, 2).type(torch.float),
                                                          reduction=self.loss_reduction)
            self.loss_r = bcewl
        elif loss == "ce_sigmoid":
            def ce_with_sigmoid(y_est, y_true):
                return F.cross_entropy(F.sigmoid(y_est), y_true, reduction=self.loss_reduction)
            self.loss_r = ce_with_sigmoid
        elif loss == "ce":
            def ce(y_est, y_true):
                return F.cross_entropy(y_est, y_true, reduction=self.loss_reduction)
            self.loss_r = ce
        elif loss == "mml":
            def hinge(y_est, y_true):
                return F.multi_margin_loss(y_est,
                                           y_true,
                                           reduction=self.loss_reduction)
            self.loss_r = hinge
        else:
            raise NotImplementedError(f"Loss {loss} not implemented")

    def forward(self, x):
        y_est = self.total_net(x)
        return y_est

    def _step(self, x, y, stage=None):
        # Last dimension is Classes
        y_est = self.forward(x).reshape(-1, 2)
        y_true = y.type(torch.int64)  # index

        loss = self.loss_r(y_est, y_true)

        guess = torch.argmax(y_est, dim=1).cpu()
        y_true = y_true.cpu()
        cm = metrics.confusion_matrix(y_true, guess, labels=range(2))
        acc = metrics.accuracy_score(y_true, guess)

        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, batch_size=x.shape[0])
        self.log(f'{stage}_loss_{self.loss_reduction}', loss, on_step=True, on_epoch=False, batch_size=x.shape[0])
        self.log(f'{stage}_acc', acc, on_epoch=True, batch_size=x.shape[0])

        return loss, cm

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs) -> Tensor:
        # training_step defines the train loop.
        x, y = batch
        loss, cm = self._step(x, y, stage='train')
        return loss

    @overrides
    def validation_step(self, batch, batch_idx=0, *args, **kwargs):
        # validation_step defines the validation loop.
        x, y = batch
        loss, cm = self._step(x, y, stage='val')
        return loss

    @overrides
    def configure_optimizers(self):
        if self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.optimizer == "RAdam":
            optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            raise NotImplementedError("Currently only supports SGD and AdamW, got " + self.optimizer)
        return optimizer


def fit_and_save(model: L.LightningModule,
                 filename: str,
                 train_data: L.LightningDataModule | None = None,  val_data: Any | None = None,
                 accumulate_grad_batches=1, gradient_clip_val=None):
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        filename="model-{val_acc:.3f}-{val_loss:.2f}",
    )
    early_stopping = EarlyStopping('val_loss', patience=200, strict=True)
    logger = TensorBoardLogger("lightning_logs", name=f"{model.name}/{model.optimizer}", log_graph=True, )
    trainer = L.Trainer(callbacks=[checkpoint_callback, early_stopping],
                        logger=logger, max_epochs=5000,
                        accumulate_grad_batches=accumulate_grad_batches,
                        gradient_clip_val=gradient_clip_val)
    trainer.fit(model=model,
                train_dataloaders=train_data,
                val_dataloaders=val_data)
    print("\nReload best model:", checkpoint_callback.best_model_path)
    best_checkpoint = torch.load(checkpoint_callback.best_model_path)
    model.load_state_dict(best_checkpoint['state_dict'])
    torch.save(model.total_net, filename)
