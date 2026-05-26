import xgboost as xgb
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Concatenate,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
class XGBoostBranch:

    def __init__(self):

        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
        )

    def train(self, X_train, y_train):

        self.model.fit(X_train, y_train)

    def predict(self, X):

        return self.model.predict(X)

    def get_proba(self, X):

        probs = self.model.predict_proba(X)

        return probs[:, 1].reshape(-1, 1)

    def evaluate(self, X, y):

        pred = self.predict(X)

        return {
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred),
            "recall": recall_score(y, pred),
            "f1": f1_score(y, pred),
        }
    
def build_bilstm_branch(
    input_len,
    vocab_size,
    embed_dim=32,
    lstm_units=64,
    dropout_rate=0.3,
):

    input_layer = Input(
        shape=(input_len,),
        name="url_input"
    )

    x = Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
    )(input_layer)

    x = Bidirectional(
        LSTM(
            lstm_units,
            return_sequences=False
        )
    )(x)

    x = Dropout(dropout_rate)(x)

    x = Dense(
        64,
        activation="relu"
    )(x)

    return input_layer, x
def build_hybrid_model(
    bilstm_input,
    bilstm_output,
    xgb_feature_dim=1,
    dropout_rate=0.3,
):

    xgb_input = Input(
        shape=(xgb_feature_dim,),
        name="xgb_input"
    )

    merged = Concatenate()([
        bilstm_output,
        xgb_input
    ])

    x = Dense(
        32,
        activation="relu"
    )(merged)

    x = Dropout(dropout_rate)(x)

    output = Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(
        inputs=[bilstm_input, xgb_input],
        outputs=output
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model