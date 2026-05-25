"""
bilstm_branch.py
----------------
BiLSTM branch cho phishing URL detection.
"""

from tensorflow.keras.layers import (

    Input,

    Embedding,

    Bidirectional,

    LSTM,

    Dense,

    Dropout
)

from tensorflow.keras.models import Model


def build_bilstm_branch(

    input_len: int,

    vocab_size: int,

    embed_dim: int = 32,

    lstm_units: int = 64,

    dropout_rate: float = 0.3
):

    """
    Build BiLSTM branch.

    Args:
        input_len   : max URL length
        vocab_size  : tokenizer vocabulary size
        embed_dim   : embedding dimension
        lstm_units  : BiLSTM hidden units
        dropout_rate: dropout

    Returns:
        input_layer
        latent_output
    """

    # ======================================
    # INPUT
    # ======================================

    input_layer = Input(

        shape=(input_len,),

        name="url_sequence_input"
    )

    # ======================================
    # EMBEDDING
    # ======================================

    x = Embedding(

        input_dim=vocab_size,

        output_dim=embed_dim,

        input_length=input_len,

        name="embedding_layer"
    )(input_layer)

    # ======================================
    # BILSTM
    # ======================================

    x = Bidirectional(

        LSTM(

            lstm_units,

            return_sequences=False
        ),

        name="bilstm_layer"
    )(x)

    # ======================================
    # DROPOUT
    # ======================================

    x = Dropout(

        dropout_rate,

        name="dropout_layer"
    )(x)

    # ======================================
    # LATENT VECTOR
    # ======================================

    latent_output = Dense(

        64,

        activation='relu',

        name="latent_vector"
    )(x)

    return input_layer, latent_output