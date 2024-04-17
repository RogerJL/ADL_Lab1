from typing import Any, Optional

import torch
from lightning.pytorch.loggers import TensorBoardLogger
from overrides import overrides
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
import sklearn.metrics as metrics
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from chatbot import SimpleBotNet

BATCH_SIZE = 32

model_ = "RNN"
optimizer_ = "SGD"  # "AdamW"

#%% NN modules
INPUT_WIDTH = 7277
HIDDEN_WIDTH = 16
OUTPUT_WIDTH = 2

class LitVanilla(L.LightningModule):
    def __init__(self, total_net: nn.Module):
        super().__init__()
        self.save_hyperparameters()
        self.total_net = total_net
        self.example_input_array = F.one_hot(torch.tensor([5]), INPUT_WIDTH).type(torch.FloatTensor)

    def forward(self, x):
        y_est = self.total_net(x)
        return y_est

    def _step(self, x, y, stage=None):
        y_est = self.forward(x)
        loss = None
        cm = metrics.confusion_matrix(y.cpu(),
                                      torch.argmax(y_est, dim=1).cpu())
        acc = np.sum(np.diag(cm)) / np.sum(cm)
        loss = F.binary_cross_entropy_with_logits(y_est,
                                                  F.one_hot(y, 2).type(torch.FloatTensor).to(y_est.device),
                                                  reduction='sum')
#        loss = F.cross_entropy(F.sigmoid(y_est),
#                               F.one_hot(y, 2).type(torch.FloatTensor).to(y_est.device),
#                               reduction='sum')
        loss = F.cross_entropy(y_est,
                               y,
                               reduction='sum')
        if stage == 'train':
            self.log('train_loss', loss, on_epoch=True)
            self.log('train_acc', acc, on_epoch=True)

        if stage == 'val':
            self.log('val_loss', loss, on_epoch=True)
            self.log('val_acc', acc, on_epoch=True)

        return loss, cm

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs) -> Tensor:
        # training_step defines the train loop.
        x, y = batch
        loss, cm = self._step(x, y, stage='train')
        return loss

    @overrides
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        # validation_step defines the validation loop.
        x, y = batch
        loss, cm = self._step(x, y, stage='val')
        return loss

    @overrides
    def configure_optimizers(self):
        if optimizer_ == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=3e-5, weight_decay=1e-3)
        elif optimizer_ == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=3e-4, weight_decay=1e-3)
        else:
            raise NotImplementedError("Currently only supports AdamW")
        return optimizer


if __name__ == '__main__':
    import data_loading_code

    vocab_size, train_set, validation_set = data_loading_code.load()

    # model
    product_judge = LitVanilla(SimpleBotNet(INPUT_WIDTH, HIDDEN_WIDTH, OUTPUT_WIDTH))


    # train model
    from lightning.pytorch.callbacks import ModelCheckpoint

    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        filename="model-{val_acc:.3f}-{val_loss:.2f}",
    )
    early_stopping = EarlyStopping('val_loss', patience=200, strict=True)
    logger = TensorBoardLogger("lightning_logs", name=f"{model_}/{optimizer_}", log_graph=True,)
    trainer = L.Trainer(callbacks=[checkpoint_callback, early_stopping], logger=logger, max_epochs=5000)
    trainer.fit(model=product_judge,
                train_dataloaders=DataLoader(train_set, batch_size=BATCH_SIZE),
                val_dataloaders=DataLoader(validation_set, batch_size=BATCH_SIZE))
    torch.save(product_judge.total_net, 'product_judge.pt')

