from typing import Union

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from matplotlib import pyplot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
from torch.utils.data import Dataset

def preprocess_pandas(data):
    data['Sentence'] = data['Sentence'].str.lower()
    data['Sentence'] = data['Sentence'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)                      # remove emails
    data['Sentence'] = data['Sentence'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)    # remove IP address
    data['Sentence'] = data['Sentence'].str.replace('[^\w\s]','')                                                       # remove special characters
    data['Sentence'] = data['Sentence'].replace('\d', '', regex=True)                                                   # remove numbers
    return data


class Sentences(Dataset):
    def __init__(self, data, labels, as_is=False):
        self.data =  data if as_is else torch.from_numpy(np.array(data)).type(torch.float)
        self.labels = labels if as_is else torch.from_numpy(np.array(labels)).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load():
    # get data, pre-process and split
    data = pd.read_csv("data/amazon_cells_labelled.txt", delimiter='\t', header=None)
    data.columns = ['Sentence', 'Class']
    data['index'] = data.index  # add new column index
    data = preprocess_pandas(data)  # pre-process
    training_data, validation_data, training_labels, validation_labels = train_test_split(
        # split the data into training, validation, and test splits
        data['Sentence'].values.astype('U'),
        data['Class'].values.astype('int32'),
        test_size=0.10,
        random_state=0,
        shuffle=True
    )
    # vectorize data using TFIDF and transform for PyTorch for scalability
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=50000, max_df=0.5, use_idf=True,
                                      norm='l2')
    training_data = word_vectorizer.fit_transform(training_data)  # transform texts to sparse matrix
    training_data = training_data.todense()  # convert to dense matrix for Pytorch
    assert word_vectorizer.get_stop_words() is None

    vocab_size = len(word_vectorizer.vocabulary_)

    translator = Translator.create_persistent(word_vectorizer)
    validation_data = translator.encode(validation_data)

    return vocab_size, Sentences(training_data, training_labels), Sentences(validation_data, validation_labels)

class Translator:
    def __init__(self, word_vectorizer):
        self.word_vectorizer = word_vectorizer

    @classmethod
    def create_persistent(cls, word_vectorizer):
        t = Translator(word_vectorizer)
        joblib.dump(word_vectorizer, 'vectorizer.pkl')
        return t

    @classmethod
    def load_persistent(cls):
        return Translator(joblib.load('vectorizer.pkl'))

    def encode(self, raw_documents, as_tensor=False):
        matrix = self.word_vectorizer.transform(raw_documents).todense()
        if as_tensor:
            return torch.from_numpy(np.array(matrix)).type(torch.float)
        return matrix

    def decode(self, x):
        return self.word_vectorizer.inverse_transform(x)


# If this is the primary file that is executed (ie not an import of another file)
if __name__ == "__main__":
    load()
