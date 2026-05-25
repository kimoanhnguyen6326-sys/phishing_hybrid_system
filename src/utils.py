"""Shared utilities for data loading, metrics, and plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _find_column(columns: list[str], candidates: set[str]) -> str | None:
    normalized = {column.lower().strip(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def load_iscx_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load URL dataset and map all non-benign classes to phishing=1."""
    df = pd.read_csv(csv_path)
    url_col = _find_column(list(df.columns), {"url"})
    type_col = _find_column(list(df.columns), {"type", "label", "class"})

    if url_col is None or type_col is None:
        raise ValueError(f"Expected url/type columns, got: {list(df.columns)}")

    out = df[[url_col, type_col]].rename(columns={url_col: "url", type_col: "type"})
    out = out.dropna(subset=["url", "type"]).drop_duplicates(subset="url")
    out["url"] = out["url"].astype(str).str.strip()
    out["label"] = out["type"].astype(str).str.lower().ne("benign").astype(int)
    out = out[["url", "label"]].reset_index(drop=True)

    print(f"[Dataset] Loaded {len(out):,} samples from {csv_path}")
    print(f"[Dataset] Legit: {(out.label == 0).sum():,} | Phishing: {(out.label == 1).sum():,}")
    return out


def load_custom_dataset(
    csv_path: str | Path,
    url_col: str = "url",
    label_col: str = "label",
) -> pd.DataFrame:
    """Load a custom CSV with url and binary label columns."""
    df = pd.read_csv(csv_path)
    missing = {url_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out = df[[url_col, label_col]].rename(columns={url_col: "url", label_col: "label"})
    out = out.dropna(subset=["url", "label"]).drop_duplicates(subset="url")
    out["url"] = out["url"].astype(str).str.strip()
    out["label"] = out["label"].astype(int)
    if not set(out["label"].unique()).issubset({0, 1}):
        raise ValueError("Custom labels must be binary values: 0 or 1.")
    return out.reset_index(drop=True)


def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_time: float = 0.0,
    test_time: float = 0.0,
) -> dict:
    return {
        "classifier": model_name,
        "train_time": round(train_time, 3),
        "test_time": round(test_time, 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 6),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 6),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 6),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 6),
    }


def print_results_table(results: list[dict]) -> None:
    header = (
        f"{'Classifier':<28} | {'Train(s)':<10} | {'Test(s)':<10} | "
        f"{'Accuracy':<10} | {'Recall':<10} | {'Precision':<10} | {'F1':<10}"
    )
    print("\n" + "=" * len(header))
    print("CLASSIFICATION RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for item in results:
        print(
            f"{item['classifier']:<28} | {item['train_time']:<10.3f} | "
            f"{item['test_time']:<10.4f} | {item['accuracy']:<10.6f} | "
            f"{item['recall']:<10.6f} | {item['precision']:<10.6f} | {item['f1']:<10.6f}"
        )
    print("=" * len(header))


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred),
        display_labels=["Legit", "Phishing"],
    )
    display.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    _save_or_close(fig, save_path)


def plot_training_history(history, save_path: str | Path | None = None) -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss.plot(history.history.get("loss", []), label="Train Loss")
    ax_loss.plot(history.history.get("val_loss", []), label="Val Loss")
    ax_loss.set_title("Loss over Epochs")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(history.history.get("accuracy", []), label="Train Accuracy")
    ax_acc.plot(history.history.get("val_accuracy", []), label="Val Accuracy")
    ax_acc.set_title("Accuracy over Epochs")
    ax_acc.set_xlabel("Epoch")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)
    _save_or_close(fig, save_path)


def plot_feature_importance(importance: dict, save_path: str | Path | None = None) -> None:
    names = list(importance.keys())
    values = list(importance.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, values, color="steelblue", alpha=0.85)
    ax.set_xlabel("Importance Score")
    ax.set_title("XGBoost Feature Importance")
    ax.invert_yaxis()
    _save_or_close(fig, save_path)


def _save_or_close(fig, save_path: str | Path | None) -> None:
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved: {save_path}")
    plt.close(fig)
