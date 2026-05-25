"""Create a small synthetic dataset for smoke tests and demos."""

from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data_src_notebooks" / "raw" / "demo_malicious_phish.csv"

LEGIT_DOMAINS = [
    "google.com", "facebook.com", "youtube.com", "amazon.com", "wikipedia.org",
    "x.com", "instagram.com", "linkedin.com", "github.com", "stackoverflow.com",
    "reddit.com", "microsoft.com", "apple.com", "netflix.com", "spotify.com",
    "coursera.org", "edx.org", "medium.com", "theverge.com", "bbc.com",
]

PHISHING_PATTERNS = [
    "paypal-secure-{}.xyz",
    "account-verify-{}.tk",
    "secure-login-{}.ml",
    "{}-banking-update.cf",
    "signin-{}-verify.gq",
    "{}.phishing-site.xyz",
    "update-account-{}.tk",
    "confirm-{}-identity.ml",
]

LEGIT_PATHS = [
    "/", "/home", "/about", "/contact", "/products",
    "/search?q=python", "/login", "/dashboard", "/blog/post-1",
    "/docs/getting-started", "/api/v2/users", "/news/technology",
]

PHISHING_PATHS = [
    "/login", "/signin", "/verify", "/account/update",
    "/secure/banking", "/confirm-identity", "/reset-password",
    "/billing/update", "/suspended/verify", "/limited/action",
]


def random_string(length: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def generate_legit_url() -> str:
    scheme = random.choice(["https", "https", "https", "http"])
    return f"{scheme}://www.{random.choice(LEGIT_DOMAINS)}{random.choice(LEGIT_PATHS)}"


def generate_phishing_url() -> str:
    domain = random.choice(PHISHING_PATTERNS).format(random_string(random.randint(4, 10)))
    path = random.choice(PHISHING_PATHS)
    trick = random.choice([
        "",
        "?redirect=https://paypal.com",
        f"@{random_string(8)}.xyz{path}",
        "//double-slash-trick.com",
    ])
    return f"http://{domain}{path}{trick}"


def build_demo_dataset(n_legit: int, n_phishing: int, seed: int) -> pd.DataFrame:
    random.seed(seed)
    df = pd.DataFrame({
        "url": [generate_legit_url() for _ in range(n_legit)]
        + [generate_phishing_url() for _ in range(n_phishing)],
        "type": ["benign"] * n_legit + ["phishing"] * n_phishing,
    })
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create demo phishing URL dataset")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--n-legit", type=int, default=1000)
    parser.add_argument("--n-phishing", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_demo_dataset(args.n_legit, args.n_phishing, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Demo dataset saved: {args.output}")
    print(f"Legit: {args.n_legit:,} | Phishing: {args.n_phishing:,} | Total: {len(df):,}")


if __name__ == "__main__":
    main()
