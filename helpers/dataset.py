from itertools import cycle
from torch.utils.data import IterableDataset
import pandas as pd
import numpy as np


class MyIterableDataset(IterableDataset):
    def __init__(self, file_path):
        self.max_token_len = pd.read_csv(file_path, header=None)[0].apply(lambda x: x.count("\t") + 1).max()
        self.file_path = file_path

    def parse_file(self, file_path):
        indices = {
            "text_tokens": 0,
            "replied": 20,
            "retweeted": 21,
            "retweeted_with_comment": 22,
            "liked": 23,
        }
        with open(file_path, encoding="utf-8") as fileobject:
            for line in fileobject:
                features = line.strip().split("\x01")
                tokens = np.fromstring(features[0], sep="\t", dtype=int)
                yield [
                    np.concatenate((tokens, np.zeros(self.max_token_len - len(tokens), dtype=int))),
                    1 if features[20] else 0,
                    1 if features[21] else 0,
                    1 if features[22] else 0,
                    1 if features[23] else 0
                ]

    def get_stream(self, file_path):
        return self.parse_file(file_path)

    def __iter__(self):
        return self.get_stream(self.file_path)


class PredictionDataset(IterableDataset):
    def __init__(self, file_path, fixed_token_vector_length=None):
        if fixed_token_vector_length:
            self.max_token_len = fixed_token_vector_length
        else:
            self.max_token_len = min(
                pd.read_csv(file_path, header=None)[0].apply(lambda x: x.count("\t") + 1).max(),
                512, # BERT hard limit
            )
        self.file_path = file_path

    def parse_file(self, file_path):
        with open(file_path, encoding="utf-8") as fileobject:
            for line in fileobject:
                features = line.strip().split("\x01")
                pad = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0], dtype=int)))
                tokens = np.fromstring(features[0], sep="\t", dtype=int)
                yield [pad(tokens, self.max_token_len)]

    def get_stream(self, file_path):
        return self.parse_file(file_path)

    def __iter__(self):
        return self.get_stream(self.file_path)
