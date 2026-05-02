"""
Porter Stemmer wrapper using NLTK.
"""

from nltk.stem import PorterStemmer


def get_stemmer():
    """
    Returns a PorterStemmer instance.
    """
    return PorterStemmer()