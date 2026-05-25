"""
feature_extractor.py
--------------------
Trích xuất handcrafted URL features cho XGBoost branch.
"""

import re
import numpy as np

from urllib.parse import urlparse


# ==========================================
# URL SHORTENING SERVICES
# ==========================================

SHORTENING_SERVICES = {

    'bit.ly',
    'tinyurl.com',
    'goo.gl',
    't.co',
    'ow.ly',
    'is.gd',
    'buff.ly',
    'adf.ly',
    'tiny.cc',
    'sh.st'
}


# ==========================================
# SUSPICIOUS TLDS
# ==========================================

SUSPICIOUS_TLDS = {

    '.xyz',
    '.tk',
    '.ml',
    '.ga',
    '.cf',
    '.gq',
    '.pw',
    '.top',
    '.click',
    '.link',
    '.online',
    '.site'
}


# ==========================================
# SUSPICIOUS WORDS
# ==========================================

SUSPICIOUS_WORDS = [

    'login',
    'signin',
    'verify',
    'secure',
    'account',
    'update',
    'confirm',
    'banking',
    'password',
    'paypal',
    'ebay'
]


# ==========================================
# MAIN FEATURE EXTRACTION
# ==========================================

def extract_features(url: str) -> list:

    """
    Trích xuất 15 handcrafted features từ URL.

    Args:
        url (str)

    Returns:
        list[float]
    """

    try:

        parsed = urlparse(url)

        domain = parsed.netloc.lower()

        path = parsed.path.lower()

        full_url = url.lower()

    except Exception:

        return [0.0] * 15

    features = []

    # ======================================
    # FEATURE 1
    # URL LENGTH
    # ======================================

    url_length = len(url)

    features.append(

        min(
            np.log1p(url_length) / 10.0,
            1.0
        )
    )

    # ======================================
    # FEATURE 2
    # NUMBER OF DOTS
    # ======================================

    dot_count = domain.count('.')

    features.append(

        min(dot_count / 5.0, 1.0)
    )

    # ======================================
    # FEATURE 3
    # NUMBER OF HYPHENS
    # ======================================

    hyphen_count = domain.count('-')

    features.append(

        min(hyphen_count / 5.0, 1.0)
    )

    # ======================================
    # FEATURE 4
    # NUMBER OF SLASHES
    # ======================================

    slash_count = path.count('/')

    features.append(

        min(slash_count / 10.0, 1.0)
    )

    # ======================================
    # FEATURE 5
    # HAS @ SYMBOL
    # ======================================

    has_at = 1.0 if '@' in url else 0.0

    features.append(has_at)

    # ======================================
    # FEATURE 6
    # HAS IP ADDRESS
    # ======================================

    ip_pattern = r'(\d{1,3}\.){3}\d{1,3}'

    has_ip = 1.0 if re.search(
        ip_pattern,
        domain
    ) else 0.0

    features.append(has_ip)

    # ======================================
    # FEATURE 7
    # URL SHORTENER
    # ======================================

    is_shortener = 1.0 if any(

        service in domain

        for service in SHORTENING_SERVICES

    ) else 0.0

    features.append(is_shortener)

    # ======================================
    # FEATURE 8
    # HTTPS
    # ======================================

    has_https = 1.0 if parsed.scheme == 'https' else 0.0

    features.append(has_https)

    # ======================================
    # FEATURE 9
    # FAKE HTTPS TOKEN
    # ======================================

    fake_https = 1.0 if 'https' in domain else 0.0

    features.append(fake_https)

    # ======================================
    # FEATURE 10
    # SPECIAL CHARACTER COUNT
    # ======================================

    special_chars = sum(

        1 for c in url

        if c in "!$&'()*+,;=%~`^{}[]|\\<>"

    )

    features.append(

        min(special_chars / 20.0, 1.0)
    )

    # ======================================
    # FEATURE 11
    # SUSPICIOUS WORDS
    # ======================================

    has_suspicious_words = 1.0 if any(

        word in full_url

        for word in SUSPICIOUS_WORDS

    ) else 0.0

    features.append(has_suspicious_words)

    # ======================================
    # FEATURE 12
    # NUMBER OF SUBDOMAINS
    # ======================================

    parts = domain.split('.')

    num_subdomains = max(
        0,
        len(parts) - 2
    )

    features.append(

        min(num_subdomains / 3.0, 1.0)
    )

    # ======================================
    # FEATURE 13
    # DOMAIN LENGTH
    # ======================================

    domain_length = len(domain)

    features.append(

        min(domain_length / 50.0, 1.0)
    )

    # ======================================
    # FEATURE 14
    # SUSPICIOUS TLD
    # ======================================

    tld = (

        '.' + domain.split('.')[-1]

        if '.' in domain else ''

    )

    suspicious_tld = 1.0 if tld in SUSPICIOUS_TLDS else 0.0

    features.append(suspicious_tld)

    # ======================================
    # FEATURE 15
    # DIGIT RATIO
    # ======================================

    digit_ratio = sum(

        c.isdigit() for c in url

    ) / max(len(url), 1)

    features.append(

        min(digit_ratio * 3.0, 1.0)
    )

    return features


# ==========================================
# BATCH EXTRACTION
# ==========================================

def extract_batch(url_list: list) -> np.ndarray:

    """
    Extract features cho nhiều URLs.

    Args:
        url_list : List[str]

    Returns:
        np.ndarray shape (N,15)
    """

    feature_matrix = [

        extract_features(url)

        for url in url_list

    ]

    return np.array(
        feature_matrix,
        dtype=np.float32
    )


# ==========================================
# FEATURE NAMES
# ==========================================

FEATURE_NAMES = [

    'url_length_log',

    'dot_count',

    'hyphen_count',

    'slash_count',

    'has_at_symbol',

    'has_ip_address',

    'is_url_shortener',

    'has_https',

    'fake_https_in_domain',

    'special_char_count',

    'has_suspicious_words',

    'num_subdomains',

    'domain_length',

    'suspicious_tld',

    'digit_ratio'
]