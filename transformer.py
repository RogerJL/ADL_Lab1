import math

import torch
from torch import nn, Tensor
from torch.nn import Transformer

import data_loading_two
from trainer import LitVanilla, fit_and_save


class TransformerModel(nn.Module):

    SOS = torch.LongTensor([[0]])

    def __init__(self, in_tokens: int, out_tokens: int,
                 d_model: int = 512, nhead: int = 8,
                 num_encoder_layers=2, num_decoder_layers=1,
                 dim_feedforward=2048, dropout: float = 0.2):
        super().__init__()
        self.sos = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = Transformer(d_model=d_model,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       )
        self.embedding = nn.Embedding(in_tokens, d_model)
        self.tgt_embedding = nn.Embedding(1, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, out_tokens)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]`` | True => causal mask | None => None

        Returns:
            output Tensor of shape ``[seq_len, batch_size, out_tokens]``
        """
        assert len(src.shape) == 2, 'src must be of shape [seq_len, batch_size]'
        if self.sos is None:
            self.sos = TransformerModel.SOS.to(src.device)
        src = self.embedding(src) / math.sqrt(self.d_model)  # was * in example, is / in report
        src = self.pos_encoder(src)
        if src_mask is True:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[0])
        tgt = self.tgt_embedding(self.sos).reshape(1, 1, self.d_model)  # target message in target language
        output = self.transformer(src=src,
                                  tgt=tgt,  # TODO: stack same for batch
                                  src_mask=src_mask)
        output = self.linear(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, src: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        src = src + self.pe[:src.size(0)]
        return self.dropout(src)


if __name__ == '__main__':
    ntokens, train_data, val_data = data_loading_two.load()

    model = TransformerModel(in_tokens=ntokens,
                             out_tokens=2,
                             d_model=8,  # embedding dimension, usually 200
                             nhead=4,  # number of heads in ``nn.MultiheadAttention``
                             num_encoder_layers=2,
                             num_decoder_layers=1,
                             dim_feedforward=8,
                             dropout=0.5,  # dropout probability
                             )

    transform_judge = LitVanilla("Transform",
                                 model,
                                 optimizer="AdamW",
                                 lr=1e-4,
                                 weight_decay=1e-5,
                                 loss="ce",
                                 loss_reduction="sum",
                                 example_input_array=torch.tensor([174, 1, 3]).reshape(-1, 1))  # S,B

    # train and saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    fit_and_save(transform_judge,
                 'transform_judge_latest.pt',
                 train_data=train_data,
                 val_data=val_data,
                 accumulate_grad_batches=1,
                 gradient_clip_val=10,
                 )
