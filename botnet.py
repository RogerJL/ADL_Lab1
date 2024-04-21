import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from trainer import LitVanilla


class SimpleBotNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.total_net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                       nn.ReLU(),
                                       nn.Dropout(),
                                       nn.Linear(hidden_size, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, output_size),
                                       )

    def forward(self, x):
        return self.total_net(x)


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
    print("Reload best model:", checkpoint_callback.best_model_path)
    best_checkpoint = torch.load(checkpoint_callback.best_model_path)
    product_judge.load_state_dict(best_checkpoint['state_dict'])
    torch.save(product_judge.total_net, 'product_judge_latest.pt')
