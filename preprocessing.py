import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


class URLProcessor:

    def __init__(self, max_len=200):
        self.max_len = max_len
        self.char_index = {}

    def fit(self, urls):

        unique_chars = set()

        for url in urls:
            unique_chars.update(url)

        self.char_index = {
            char: idx + 1
            for idx, char in enumerate(sorted(unique_chars))
        }

    def transform(self, urls):

        sequences = []

        for url in urls:

            seq = [
                self.char_index.get(char, 0)
                for char in url
            ]

            sequences.append(seq)

        padded = pad_sequences(
            sequences,
            maxlen=self.max_len,
            padding="post",
            truncating="post"
        )

        return padded

    def fit_transform(self, urls):

        self.fit(urls)

        return self.transform(urls)

    def get_vocab_size(self):

        return len(self.char_index) + 1