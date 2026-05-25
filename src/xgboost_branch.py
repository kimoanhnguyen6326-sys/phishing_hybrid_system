"""
xgboost_branch.py
-----------------
XGBoost branch cho phishing URL detection.
"""

import joblib
import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.model_selection import cross_val_score


class XGBoostBranch:

    """
    XGBoost model xử lý handcrafted URL features.
    """

    def __init__(

        self,

        n_estimators: int = 200,

        max_depth: int = 6,

        learning_rate: float = 0.1,

        subsample: float = 0.8,

        colsample_bytree: float = 0.8,

        random_state: int = 42
    ):

        self.model = XGBClassifier(

            n_estimators=n_estimators,

            max_depth=max_depth,

            learning_rate=learning_rate,

            subsample=subsample,

            colsample_bytree=colsample_bytree,

            random_state=random_state,

            eval_metric='logloss',

            use_label_encoder=False,

            n_jobs=-1
        )

        self.is_trained = False

    # ======================================
    # TRAIN
    # ======================================

    def train(

        self,

        X_train: np.ndarray,

        y_train: np.ndarray,

        X_val: np.ndarray = None,

        y_val: np.ndarray = None
    ):

        """
        Train XGBoost model.
        """

        print("\n[XGBoost] Training started...")

        if X_val is not None and y_val is not None:

            self.model.fit(

                X_train,
                y_train,

                eval_set=[(X_val, y_val)],

                verbose=False
            )

        else:

            self.model.fit(

                X_train,
                y_train,

                verbose=False
            )

        self.is_trained = True

        print("[XGBoost] Training completed.")

    # ======================================
    # PREDICT
    # ======================================

    def predict(

        self,

        X: np.ndarray
    ) -> np.ndarray:

        """
        Predict labels.
        """

        self._check_trained()

        return self.model.predict(X)

    # ======================================
    # PREDICT PROBA
    # ======================================

    def get_proba(

        self,

        X: np.ndarray
    ) -> np.ndarray:

        """
        Predict phishing probability.
        """

        self._check_trained()

        probabilities = self.model.predict_proba(X)[:, 1]

        return probabilities.reshape(-1, 1)

    # ======================================
    # EVALUATE
    # ======================================

    def evaluate(

        self,

        X_test: np.ndarray,

        y_test: np.ndarray
    ) -> dict:

        """
        Evaluate model performance.
        """

        self._check_trained()

        y_pred = self.predict(X_test)

        y_prob = self.model.predict_proba(X_test)[:, 1]

        results = {

            'accuracy': accuracy_score(
                y_test,
                y_pred
            ),

            'precision': precision_score(
                y_test,
                y_pred
            ),

            'recall': recall_score(
                y_test,
                y_pred
            ),

            'f1': f1_score(
                y_test,
                y_pred
            ),

            'roc_auc': roc_auc_score(
                y_test,
                y_prob
            )
        }

        return results

    # ======================================
    # CROSS VALIDATION
    # ======================================

    def cross_validate(

        self,

        X: np.ndarray,

        y: np.ndarray,

        cv: int = 5
    ) -> float:

        """
        Cross-validation score.
        """

        scores = cross_val_score(

            self.model,

            X,
            y,

            cv=cv,

            scoring='f1'
        )

        print(f"\nCross-validation F1 scores: {scores}")

        return scores.mean()

    # ======================================
    # FEATURE IMPORTANCE
    # ======================================

    def feature_importance(

        self,

        feature_names: list = None
    ) -> dict:

        """
        Return feature importance.
        """

        self._check_trained()

        importance = self.model.feature_importances_

        if feature_names:

            return dict(

                sorted(

                    zip(feature_names, importance),

                    key=lambda x: x[1],

                    reverse=True
                )
            )

        return dict(enumerate(importance))

    # ======================================
    # SAVE MODEL
    # ======================================

    def save(

        self,

        path: str
    ):

        """
        Save trained model.
        """

        self._check_trained()

        joblib.dump(

            self.model,

            path
        )

        print(f"\n[XGBoost] Model saved: {path}")

    # ======================================
    # LOAD MODEL
    # ======================================

    def load(

        self,

        path: str
    ):

        """
        Load trained model.
        """

        self.model = joblib.load(path)

        self.is_trained = True

        print(f"\n[XGBoost] Model loaded: {path}")

    # ======================================
    # CHECK TRAINED
    # ======================================

    def _check_trained(self):

        """
        Internal safety check.
        """

        if not self.is_trained:

            raise RuntimeError(

                "Model chưa được train."
            )