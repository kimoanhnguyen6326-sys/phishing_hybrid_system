"""
url_processor.py
----------------
Convert URL strings thành padded character sequences
cho BiLSTM branch.
"""

import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences


class URLProcessor:

    """
    Character-level tokenizer cho URLs.
    """

    def __init__(

        self,

        max_len: int = 200
    ):

        self.max_len = max_len

        self.tokenizer = Tokenizer(

            char_level=True,

            lower=True,

            oov_token="[UNK]"
        )

    # ======================================
    # FIT + TRANSFORM
    # ======================================

    def fit_transform(

        self,

        urls: list
    ) -> np.ndarray:

        """
        Fit tokenizer và transform URLs.
        """

        self.tokenizer.fit_on_texts(urls)

        sequences = self.tokenizer.texts_to_sequences(urls)

        padded = pad_sequences(

            sequences,

            maxlen=self.max_len,

            padding='post',

            truncating='post'
        )

        return padded

    # ======================================
    # TRANSFORM ONLY
    # ======================================

    def transform(

        self,

        urls: list
    ) -> np.ndarray:

        """
        Transform URLs using fitted tokenizer.
        """

        sequences = self.tokenizer.texts_to_sequences(urls)

        padded = pad_sequences(

            sequences,

            maxlen=self.max_len,

            padding='post',

            truncating='post'
        )

        return padded

    # ======================================
    # VOCAB SIZE
    # ======================================

    def get_vocab_size(self) -> int:

        """
        Return vocabulary size.
        """

        return len(self.tokenizer.word_index) + 1