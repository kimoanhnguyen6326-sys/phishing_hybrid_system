import re
import numpy as np


SUSPICIOUS_WORDS = [
    "login",
    "verify",
    "secure",
    "account",
    "update",
    "free",
    "bank",
    "paypal",
    "signin",
]


FEATURE_NAMES = [
    "url_length",
    "digit_count",
    "special_char_count",
    "subdomain_count",
    "https_flag",
    "suspicious_word_count",
]


def extract_features(url):

    url = str(url).lower()

    url_length = len(url)

    digit_count = sum(c.isdigit() for c in url)

    special_char_count = len(
        re.findall(r"[^a-zA-Z0-9]", url)
    )

    subdomain_count = url.count(".")

    https_flag = 1 if "https" in url else 0

    suspicious_word_count = sum(
        word in url
        for word in SUSPICIOUS_WORDS
    )

    return [
        url_length,
        digit_count,
        special_char_count,
        subdomain_count,
        https_flag,
        suspicious_word_count,
    ]


def extract_batch(urls):

    features = [
        extract_features(url)
        for url in urls
    ]

    return np.array(features)