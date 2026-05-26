import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate_model(
    model_name,
    y_true,
    y_pred,
    y_prob=None,
    train_time=None,
    test_time=None,
    latency=None,
):

    results = {
        "Classifier": model_name,
        "Train Time (s)": train_time,
        "Test Time (s)": test_time,
        "Latency (ms)": latency,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
    }

    if y_prob is not None:

        results["ROC-AUC"] = roc_auc_score(
            y_true,
            y_prob
        )

    return results


def measure_latency(
    model,
    X_seq,
    X_xgb,
    num_samples=100,
):

    times = []

    for i in range(num_samples):

        start = time.time()

        _ = model.predict(
            [
                X_seq[i:i+1],
                X_xgb[i:i+1]
            ],
            verbose=0
        )

        end = time.time()

        times.append(end - start)

    avg_latency = np.mean(times) * 1000

    return avg_latency


def plot_confusion_matrix(
    y_true,
    y_pred,
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.show()


def print_results(results):

    df = pd.DataFrame(results)

    print(df)

    return df