import os
from io import open
import torch
import torch.utils.cpp_extension as cpp

from benchmark import measure

cpp_tokenizer = cpp.load(name='tokenizer',
                         sources=['cpp/tokenizer.cpp'],
                         extra_cflags=['-O3'])

class Corpus:
    # 90% data used for training, 10% for validation, 10% for test
    train_percent, val_percent = 0.8, 0.1

    def __init__(self, path):
        # Tokenize the data
        data, dictionary = cpp_tokenizer.fast_tokenize(path)
        data = torch.tensor(data, dtype=torch.int64)
        self.idx2word = dictionary
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
