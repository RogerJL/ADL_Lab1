from torch import nn


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
