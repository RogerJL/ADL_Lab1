import json
import random

import data_loading_code
import torch
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





translator = data_loading_code.Translator.load_persistent()
product_judge = torch.load('product_judge.pt')

with open('data/bot_talk.json', 'r') as json_data:
    messages = json.load(json_data)

while True:
    review = input('Enter a review (or q to quit): ')
    if review == 'q':
        break

    x = translator.encode([review], as_tensor=True)
    judgement_vector = product_judge.forward(x)
    judgement = 'positive' if torch.argmax(judgement_vector) else 'negative'

    msg = judgement # random.choice(messages['answer_judgement'][judgement])
    print(f"Anna: {msg}")