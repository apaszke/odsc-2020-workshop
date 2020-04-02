import os
from io import open
import torch


class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def tokenize(path):
    """Tokenizes a text file."""
    assert os.path.exists(path)
    dictionary = Dictionary()
    # Add words to the dictionary
    with open(path, 'r', encoding="utf8") as f:
        for line in f:
            words = line.split() + ['<eos>']
            for word in words:
                dictionary.add_word(word)

    # Tokenize file content
    with open(path, 'r', encoding="utf8") as f:
        idss = []
        for line in f:
            words = line.split() + ['<eos>']
            ids = []
            for word in words:
                ids.append(dictionary.word2idx[word])
            idss.append(torch.tensor(ids).type(torch.int64))
        ids = torch.cat(idss)

    return ids, dictionary


class Corpus:
    # 90% data used for training, 10% for validation, 10% for test
    train_percent, val_percent = 0.8, 0.1

    def __init__(self, path):
        # Tokenize the data
        data, dictionary = tokenize(path)
        self.idx2word = dictionary.idx2word
        # Split input training, validation and test sets
        train_start, train_end = 0, int(len(data) * self.train_percent)
        val_start, val_end = train_end, train_end + int(len(data) * self.val_percent)
        test_start = val_end
        self.train = data[train_start:train_end]
        self.valid = data[val_start:val_end]
        self.test = data[test_start:]

    @property
    def ntokens(self):
        return len(self.idx2word)
