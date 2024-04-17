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
        data = [t if t.numel() > 0 else None for t in data]
        return data

    train_data = data_process(training_data)
    val_data = data_process(validation_data)

    train_labels = torch.tensor(training_labels).to(device)
    val_labels = torch.tensor(validation_labels).to(device)

    return len(vocab), (train_data, train_labels), (val_data, val_labels)

if __name__ == '__main__':
    load()