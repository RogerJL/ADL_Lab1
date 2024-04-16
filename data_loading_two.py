import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import dataset
from torchtext.datasets import WikiText2  # + torchdata, portalocker
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch import Tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load():
    # get data, pre-process and split
    data = pd.read_csv("data/amazon_cells_labelled.txt", delimiter='\t', header=None)
    data.columns = ['Sentence', 'Class']

    training_data, validation_data, training_labels, validation_labels = train_test_split(
        # split the data into training, validation, and test splits
        data['Sentence'].values.astype('U'),
        data['Class'].values.astype('int32'),
        test_size=0.10,
        random_state=0,
        shuffle=True
    )

    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, training_data), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    train_data = data_process(training_data)
    val_data = data_process(validation_data)


    def batchify(data: Tensor, bsz: int) -> Tensor:
        """Divides the data into ``bsz`` separate sequences, removing extra elements
        that wouldn't cleanly fit.

        Arguments:
            data: Tensor, shape ``[N]``
            bsz: int, batch size

        Returns:
            Tensor of shape ``[N // bsz, bsz]``
        """
        seq_len = data.size(0) // bsz
        data = data[:seq_len * bsz]
        data = data.view(bsz, seq_len).t().contiguous()
        return data.to(device)

    batch_size = 20
    eval_batch_size = 10
    train_data = batchify(train_data, batch_size)  # shape ``[seq_len, batch_size]``
    val_data = batchify(val_data, eval_batch_size)

    return train_data, val_data

if __name__ == '__main__':
    load()