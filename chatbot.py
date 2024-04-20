import json
import random

from torchtext.data import get_tokenizer

import data_loading_code
import torch

import data_loading_two
from botnet import SimpleBotNet  # used by torch.load
from transformer import TransformerModel, PositionalEncoding  # used by torch.load

if __name__ == '__main__':

    transformer_ = False
    if transformer_:
        # judge_model = transformer.TransformerModel(500, 2)
        judge_model = torch.load('transform_judge_latest.pt')
        encoder = data_loading_two.build_encoder(get_tokenizer('basic_english'),
                                                 torch.load("data/vocab.pt"))
    else:
        judge_model = torch.load('product_judge.pt')
        translator = data_loading_code.Translator.load_persistent(use_tensor=True)
        encoder = translator.encode

    judge_model.eval()

    with open('data/bot_talk.json', 'r') as json_data:
        messages = json.load(json_data)

    while True:
        review = input('Enter a review (or q to quit): ')
        if review == 'q':
            break

        x = encoder([review])
        judgement_vector = judge_model.forward(x[0])
        judgement = 'positive' if torch.argmax(judgement_vector) else 'negative'

        msg = random.choice(messages['answer_judgement'][judgement])
        print(f"Anna: {msg}")