"""
Normalization: remove stopwords, apply stemming.
"""

from stopwords import STOPWORDS


def normalize_tokens(tokens, stemmer):
    """
    Given a list of word tokens:
    - Remove stopwords
    - Remove empty strings
    - Apply stemming
    Returns normalized token list.
    """
    normalized = []
    for token in tokens:
        if token and token not in STOPWORDS:
            stemmed = stemmer.stem(token)
            if stemmed:
                normalized.append(stemmed)
    return normalized