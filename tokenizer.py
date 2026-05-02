"""
Sentence and word tokenizers.
"""

import re
import string


def split_sentences(text):
    """
    Split text into sentences.
    Handles punctuation: . ! ?
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def tokenize_words(text):
    """
    Split text into individual word tokens.
    1. Lowercase
    2. Remove punctuation
    3. Split on whitespace
    Returns list of tokens.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return tokens