
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from overrides import overrides
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import sklearn.metrics as metrics

from chatbot import SimpleBotNet


class LitVanilla(L.LightningModule):
    def __init__(self, total_net: nn.Module,
                 optimizer: str, lr, weight_decay,
                 loss: str, loss_reduction: str,
                 example_input_array=None):
        super().__init__()
        self.save_hyperparameters(ignore=['total_net'])
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
        else:
            raise NotImplementedError("Currently only supports SGD and AdamW, got " + self.optimizer)
        return optimizer


if __name__ == '__main__':
    import data_loading_code

    vocab_size, train_set, validation_set = data_loading_code.load()

    # model
    # %% NN modules
    INPUT_WIDTH = vocab_size
    HIDDEN_WIDTH = 16
    OUTPUT_WIDTH = 2
    BATCH_SIZE = 32

    model_ = "Chat"

    product_judge = LitVanilla(SimpleBotNet(INPUT_WIDTH, HIDDEN_WIDTH, OUTPUT_WIDTH),
                               optimizer="SGD",
                               lr=2e-4,
                               weight_decay=1e-5,
                               loss="ce",
                               loss_reduction="sum",
                               example_input_array=F.one_hot(torch.tensor([5]), INPUT_WIDTH).type(torch.float))

    # train model

    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        filename="model-{val_acc:.3f}-{val_loss:.2f}",
    )
    early_stopping = EarlyStopping('val_loss', patience=200, strict=True)
    logger = TensorBoardLogger("lightning_logs", name=f"{model_}/{product_judge.optimizer}", log_graph=True,)
    trainer = L.Trainer(callbacks=[checkpoint_callback, early_stopping], logger=logger, max_epochs=5000)
    trainer.fit(model=product_judge,
                train_dataloaders=DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
                val_dataloaders=DataLoader(validation_set, batch_size=BATCH_SIZE))
    torch.save(product_judge.total_net, 'product_judge.pt')
