"""Run inference with a trained phishing URL detector."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tensorflow.keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import extract_batch
from src.xgboost_branch import XGBoostBranch

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_trained_model(output_dir: Path):
    model_dir = output_dir / "models"
    keras_path = model_dir / "hybrid_keras.h5"
    xgb_path = model_dir / "xgb_branch.pkl"
    processor_path = model_dir / "url_processor.pkl"

    missing = [path for path in [keras_path, xgb_path, processor_path] if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing trained model files:\n{missing_text}\nRun train.py first.")

    import joblib

    keras_model = load_model(keras_path)
    xgb_branch = XGBoostBranch().load(xgb_path)
    processor = joblib.load(processor_path)
    return keras_model, xgb_branch, processor


def predict_url(url: str, keras_model, xgb_branch: XGBoostBranch, processor) -> dict:
    X_seq = processor.transform([url])
    X_feat = extract_batch([url])
    xgb_prob = xgb_branch.get_proba(X_feat)
    probability = float(keras_model.predict([X_seq, xgb_prob], verbose=0).flatten()[0])
    label = int(probability >= 0.5)
    return {
        "url": url,
        "label": label,
        "probability": probability,
        "confidence": probability if label else 1.0 - probability,
        "verdict": "PHISHING" if label else "LEGITIMATE",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phishing URL detector inference")
    parser.add_argument("--url", type=str, default=None, help="Single URL to check")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keras_model, xgb_branch, processor = load_trained_model(args.output_dir)

    urls = [args.url] if args.url else [
        "https://www.google.com",
        "https://github.com/login",
        "http://paypal-secure-login.xyz/account/verify",
        "http://192.168.1.1/bank/login.php?user=admin",
        "https://bit.ly/3abc123",
        "http://www.amazon.com.security-update.tk/signin",
    ]

    print("\n" + "=" * 86)
    print(f"{'URL':<56} | {'Verdict':<10} | {'Phishing Prob':>13} | {'Confidence':>10}")
    print("-" * 86)
    for url in urls:
        result = predict_url(url, keras_model, xgb_branch, processor)
        url_display = url[:53] + "..." if len(url) > 56 else url
        print(
            f"{url_display:<56} | {result['verdict']:<10} | "
            f"{result['probability']:>12.1%} | {result['confidence']:>9.1%}"
        )
    print("=" * 86)


if __name__ == "__main__":
    main()
