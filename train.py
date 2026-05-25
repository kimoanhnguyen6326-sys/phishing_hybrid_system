"""Train the Hybrid Bi-LSTM + XGBoost phishing URL detector."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import pandas as pd
import random
import numpy as np
import tensorflow as tf
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bilstm_branch import build_bilstm_branch
from src.feature_extractor import FEATURE_NAMES, extract_batch
from src.hybrid_model import build_hybrid_model, get_callbacks
from src.url_processor import URLProcessor
from src.utils import (
    evaluate_model,
    load_custom_dataset,
    load_iscx_dataset,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_training_history,
    print_results_table,
)
from src.xgboost_branch import XGBoostBranch

DEFAULT_DATA_PATH = PROJECT_ROOT / "data_src_notebooks" / "raw" / "malicious_phish.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train phishing URL detector")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--data-format", choices=["iscx", "custom"], default="iscx")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-url-len", type=int, default=200)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_dataset(path: Path, data_format: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Pass --data-path or generate a demo dataset with create_demo_dataset.py."
        )
    return load_iscx_dataset(path) if data_format == "iscx" else load_custom_dataset(path)


def split_dataset(urls, labels, args: argparse.Namespace):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        urls,
        labels,
        test_size=args.test_size,
        stratify=labels,
        random_state=args.random_state,
    )
    relative_val_size = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=relative_val_size,
        stratify=y_trainval,
        random_state=args.random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("HYBRID BI-LSTM + XGBOOST PHISHING URL DETECTOR")
    print("=" * 72)

    print("\n[1/6] Loading dataset...")
    df = load_dataset(args.data_path, args.data_format)
    urls = df["url"].tolist()
    labels = df["label"].to_numpy()

    print("\n[2/6] Splitting train/validation/test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(urls, labels, args)
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    print("\n[3/6] Tokenizing URL strings...")
    processor = URLProcessor(max_len=args.max_url_len)
    X_seq_train = processor.fit_transform(X_train)
    X_seq_val = processor.transform(X_val)
    X_seq_test = processor.transform(X_test)

    print("\n[4/6] Extracting handcrafted URL features...")
    X_feat_train = extract_batch(X_train)
    X_feat_val = extract_batch(X_val)
    X_feat_test = extract_batch(X_test)
    print(f"Feature matrix: {X_feat_train.shape}")

    print("\n[5/6] Training XGBoost branch...")
    xgb = XGBoostBranch()
    start = time.time()
    xgb.train(X_feat_train, y_train, X_feat_val, y_val)
    xgb_train_time = time.time() - start

    xgb_prob_train = xgb.get_proba(X_feat_train)
    xgb_prob_val = xgb.get_proba(X_feat_val)
    xgb_prob_test = xgb.get_proba(X_feat_test)
    xgb_results = xgb.evaluate(X_feat_test, y_test)
    print(f"XGBoost alone: accuracy={xgb_results['accuracy']:.4f}, f1={xgb_results['f1']:.4f}")

    print("\n[6/6] Training hybrid Keras model...")
    bilstm_input, bilstm_output = build_bilstm_branch(
        input_len=args.max_url_len,
        vocab_size=processor.get_vocab_size(),
        embed_dim=args.embed_dim,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout,
    )
    hybrid_model = build_hybrid_model(
        bilstm_input,
        bilstm_output,
        xgb_feature_dim=1,
        dropout_rate=args.dropout,
    )

    start = time.time()
    history = hybrid_model.fit(
        [X_seq_train, xgb_prob_train],
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=([X_seq_val, xgb_prob_val], y_val),
        callbacks=get_callbacks(patience=5),
        verbose=1,
    )
    hybrid_train_time = time.time() - start

    start = time.time()
    y_prob = hybrid_model.predict([X_seq_test, xgb_prob_test], verbose=0).flatten()
    hybrid_test_time = time.time() - start
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_prob)

    print(f"ROC-AUC: {auc:.4f}")

    results = [
        evaluate_model(
            "XGBoost (standalone)",
            y_test,
            (xgb_prob_test.flatten() >= 0.5).astype(int),
            train_time=xgb_train_time,
        ),
        evaluate_model(
            "Hybrid Bi-LSTM+XGBoost",
            y_test,
            y_pred,
            train_time=hybrid_train_time,
            test_time=hybrid_test_time,
        ),
    ]
    print_results_table(results)

    hybrid_model.save(model_dir / "hybrid_keras.keras ")
    xgb.save(model_dir / "xgb_branch.pkl")
    joblib.dump(processor, model_dir / "url_processor.pkl")

    plot_confusion_matrix(y_test, y_pred, "Hybrid Model Confusion Matrix", args.output_dir / "confusion_matrix.png")
    plot_training_history(history, args.output_dir / "training_history.png")
    plot_feature_importance(xgb.feature_importance(FEATURE_NAMES), args.output_dir / "feature_importance.png")
    pd.DataFrame(results).to_csv(args.output_dir / "results.csv", index=False)

    print(f"\nDone. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
