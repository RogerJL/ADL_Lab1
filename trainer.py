import os
from typing import Any, Optional

import torch
from overrides import overrides
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping

model_ = "RNN"
optimizer_ = "AdamW"

#%% NN modules
INPUT_WIDTH = 7277
HIDDEN_WIDTH = 64
RECURRENT_WIDTH = 64
OUTPUT_WIDTH = 2

class InputNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in1 = nn.Sequential(nn.Linear(INPUT_WIDTH, HIDDEN_WIDTH),
                                 nn.ReLU())

    def forward(self, x):
        """
        Computes the forward pass of a vanilla RNN.
        """
        x =  self.in1(x)

        return x

class RecurrentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = nn.Sequential(nn.Linear(HIDDEN_WIDTH + RECURRENT_WIDTH, HIDDEN_WIDTH + RECURRENT_WIDTH),
                                       nn.Tanh())
        self.hidden_state = None

    def on_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        # reinit H
        x, y = batch
        batch_size = x.size(0)
        self.hidden_state = [torch.zeros(size=(batch_size, HIDDEN_WIDTH)).to(x.device)]
        return None

    def forward(self, x):
        """
        Computes the forward pass of a vanilla RNN.
        """
        c = torch.concat([x, self.hidden_state[-1]], dim=1)
        c = self.recurrent(c)
        self.hidden_state.append(c[:, HIDDEN_WIDTH:])
        x2 = c[:, :HIDDEN_WIDTH]
        return x2

class OutputNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.out1 = nn.Sequential(nn.Linear(HIDDEN_WIDTH, OUTPUT_WIDTH))

    def forward(self, x):
        """
        Computes the output forward pass of a vanilla RNN.
        """
        x = self.out1(x)
        return x



class LitVanillaRNN(L.LightningModule):
    def __init__(self, input_net, recurrent_net, output_net):
        super().__init__()
        self.input_net = input_net
        self.recurrent_net = recurrent_net
        self.output_net = output_net

    def _step(self, x, y):
        h = self.input_net(x)
        h = self.recurrent_net(h)
        y_est = self.output_net(h)
        loss = F.binary_cross_entropy_with_logits(y_est,
                                                  F.one_hot(y, 2).type(torch.FloatTensor).to(y_est.device),
                                                  reduction='sum')
        return loss

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs) -> Tensor:
        # training_step defines the train loop.
        x, y = batch
        loss = self._step(x, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    @overrides
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        # training_step defines the train loop.
        x, y = batch
        loss = self._step(x, y)
        self.log('val_loss', loss, on_epoch=True)
        return loss

    @overrides
    def configure_optimizers(self):
        if optimizer_ == "AdamW":
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        else:
            raise NotImplementedError("Currently only supports AdamW")
        return optimizer

    @overrides
    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        # reinit H
        return self.recurrent_net.on_batch_start(batch, batch_idx)

    @overrides
    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # reinit H
        self.recurrent_net.on_batch_start(batch, batch_idx)

if __name__ == '__main__':
    import data_loading_code

    vocab_size, train_set, validation_set = data_loading_code.load()

    # model
    product_judge = LitVanillaRNN(InputNet(), RecurrentNet(), OutputNet())

    # train model
    early_stopping = EarlyStopping('val_loss', patience=10, strict=True)
    trainer = L.Trainer(callbacks=[early_stopping])
    trainer.fit(model=product_judge,
                train_dataloaders=DataLoader(train_set, batch_size=32),
                val_dataloaders=DataLoader(validation_set, batch_size=32))
    torch.save(product_judge, 'product_judge.pt')

