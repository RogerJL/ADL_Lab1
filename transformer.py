import math

import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from torch import nn, Tensor
from torch.nn import Transformer
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

import data_loading_two
from trainer import LitVanilla

class TransformerModel(nn.Module):

    SOS = torch.LongTensor([[0]])

    def __init__(self, in_tokens: int, out_tokens: int,
                 d_model: int=512, nhead: int=8,
                 num_encoder_layers=2, num_decoder_layers=1,
                 dim_feedforward=2048, dropout: float = 0.2):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = Transformer(d_model=d_model,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       #                activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                                       #                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                                       #                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                                       #                 bias: bool = True, device=None, dtype=None
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
            tgt: Tensor, shape ``[seq_len, batch_size]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, out_tokens]``
        """
        assert len(src.shape) == 2, 'src must be of shape [seq_len, batch_size]'
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is True:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[0])
        tgt = self.tgt_embedding(TransformerModel.SOS.to(src.device)).reshape(1, 1, self.d_model)  # target message in target language
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

    model_ = "Transformer"

    model = TransformerModel(in_tokens=ntokens,
                             out_tokens=2,
                             d_model=8,  # embedding dimension, usually 200
                             nhead=8,  # number of heads in ``nn.MultiheadAttention``
                             num_encoder_layers=2,
                             num_decoder_layers=1,
                             dim_feedforward=8,
                             dropout=0.5,  # dropout probability
                             )

    transform_judge = LitVanilla(model,
                                 optimizer="AdamW",
                                 lr=2e-4,
                                 weight_decay=1e-4,
                                 loss="ce",
                                 loss_reduction="sum",
                                 example_input_array=torch.tensor([174, 1, 3]).reshape(-1, 1))  # S,B

    # train model

    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        filename="model-{val_acc:.3f}-{val_loss:.2f}",
    )
    early_stopping = EarlyStopping('val_loss', patience=200, strict=True)
    logger = TensorBoardLogger("lightning_logs", name=f"{model_}/{transform_judge.optimizer}", log_graph=True,)
    trainer = L.Trainer(callbacks=[checkpoint_callback, early_stopping], logger=logger, max_epochs=5000)
    trainer.fit(model=transform_judge,
                train_dataloaders=train_data,
                val_dataloaders=val_data)

    print("Reload best model:", checkpoint_callback.best_model_path)
    best_checkpoint = torch.load(checkpoint_callback.best_model_path)
    transform_judge.load_state_dict(best_checkpoint['state_dict'])
    torch.save(transform_judge.total_net, 'transform_judge_latest.pt')
