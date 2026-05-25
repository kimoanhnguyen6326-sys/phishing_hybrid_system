"""Fusion model that combines Bi-LSTM features and XGBoost probability."""

from __future__ import annotations

from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

def build_hybrid_model(
    bilstm_input,
    bilstm_output,
    xgb_feature_dim: int = 1,
    dropout_rate: float = 0.3,
) -> Model:
    """Build and compile the final binary classifier."""
    if xgb_feature_dim <= 0:
        raise ValueError("xgb_feature_dim must be positive.")

    xgb_input = layers.Input(shape=(xgb_feature_dim,), name="xgb_prob_input")
    merged = layers.Concatenate(name="fusion")([bilstm_output, xgb_input])

    x = layers.Dense(64, activation="relu", name="decision_dense_1")(merged)
    x = layers.BatchNormalization(name="decision_batch_norm")(x)
    x = layers.Dropout(dropout_rate, name="decision_dropout")(x)
    x = layers.Dense(32, activation="relu", name="decision_dense_2")(x)
    output = layers.Dense(1, activation="sigmoid", name="phishing_probability")(x)

    model = Model(
        inputs=[bilstm_input, xgb_input],
        outputs=output,
        name="Hybrid_BiLSTM_XGBoost",
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def get_callbacks(patience: int = 5) -> list:
    """Return standard callbacks for stable training."""
    return [

    EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    ),

    ModelCheckpoint(
        filepath="best_model.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]


class HybridPhishingDetector:
    """Convenience wrapper around the full phishing detection pipeline."""

    def __init__(
        self,
        max_url_len: int = 200,
        embed_dim: int = 32,
        lstm_units: int = 64,
        dropout_rate: float = 0.3,
    ):
        from src.feature_extractor import extract_batch
        from src.url_processor import URLProcessor
        from src.xgboost_branch import XGBoostBranch

        self.max_url_len = max_url_len
        self.embed_dim = embed_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.url_processor = URLProcessor(max_len=max_url_len)
        self.xgb_branch = XGBoostBranch()
        self.keras_model = None
        self._extract_batch = extract_batch

    def fit(
        self,
        urls_train: list,
        y_train,
        urls_val: list | None = None,
        y_val=None,
        epochs: int = 20,
        batch_size: int = 64,
    ):
        from src.bilstm_branch import build_bilstm_branch

        X_seq_train = self.url_processor.fit_transform(urls_train)
        X_feat_train = self._extract_batch(urls_train)

        if urls_val is not None and y_val is not None:
            X_feat_val = self._extract_batch(urls_val)
            self.xgb_branch.train(X_feat_train, y_train, X_feat_val, y_val)
        else:
            self.xgb_branch.train(X_feat_train, y_train)

        xgb_prob_train = self.xgb_branch.get_proba(X_feat_train)
        bilstm_input, bilstm_output = build_bilstm_branch(
            input_len=self.max_url_len,
            vocab_size=self.url_processor.get_vocab_size(),
            embed_dim=self.embed_dim,
            lstm_units=self.lstm_units,
            dropout_rate=self.dropout_rate,
        )
        self.keras_model = build_hybrid_model(
            bilstm_input,
            bilstm_output,
            xgb_feature_dim=1,
            dropout_rate=self.dropout_rate,
        )

        validation_data = None
        if urls_val is not None and y_val is not None:
            X_seq_val = self.url_processor.transform(urls_val)
            xgb_prob_val = self.xgb_branch.get_proba(self._extract_batch(urls_val))
            validation_data = ([X_seq_val, xgb_prob_val], y_val)

        return self.keras_model.fit(
            [X_seq_train, xgb_prob_train],
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=get_callbacks(patience=5),
            verbose=1,
        )

    def predict_proba(self, urls: list):
        if self.keras_model is None:
            raise RuntimeError("Model has not been fitted.")
        X_seq = self.url_processor.transform(urls)
        X_feat = self._extract_batch(urls)
        xgb_prob = self.xgb_branch.get_proba(X_feat)
        return self.keras_model.predict([X_seq, xgb_prob], verbose=0).flatten()

    def predict(self, urls: list, threshold: float = 0.5):
        return (self.predict_proba(urls) >= threshold).astype(int)
