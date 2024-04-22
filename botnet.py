import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from trainer import LitVanilla, fit_and_save


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

    product_judge = LitVanilla("Chat",
                               SimpleBotNet(INPUT_WIDTH, HIDDEN_WIDTH, OUTPUT_WIDTH),
                               optimizer="SGD",
                               lr=2e-4,
                               weight_decay=1e-5,
                               loss="ce",
                               loss_reduction="sum",
                               example_input_array=F.one_hot(torch.tensor([5]), INPUT_WIDTH).type(torch.float))

    # train model
    fit_and_save(product_judge,
                 'product_judge_latest.pt',
                 train_data=DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
                 val_data=DataLoader(validation_set, batch_size=BATCH_SIZE))
