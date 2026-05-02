"""
TF, IDF, and TF-IDF computation.
"""

import math


def compute_tf(tokens):
    """
    Compute normalized term frequency for a token list.
    Returns: {term: tf_score}
    """
    tf = {}
    total = len(tokens)

    for token in tokens:
        tf[token] = tf.get(token, 0) + 1

    for token in tf:
        tf[token] /= total

    return tf


def compute_idf(all_tokens_list):
    """
    Compute IDF across all sentences.
    all_tokens_list: list of lists of tokens (one list per sentence)
    Returns: {term: idf_score}
    """
    N = len(all_tokens_list)
    df = {}

    for sentence_tokens in all_tokens_list:
        unique_terms = set(sentence_tokens)
        for term in unique_terms:
            df[term] = df.get(term, 0) + 1

    idf = {}
    for term, count in df.items():
        idf[term] = math.log(N / count)

    return idf


def compute_tfidf(all_tokens_list):
    """
    Compute TF-IDF vectors for all sentences.
    Returns:
        tfidf_vectors: list of dicts [{term: tfidf}, ...]
        idf: dict {term: idf}
    """
    idf = compute_idf(all_tokens_list)
    tfidf_vectors = []

    for tokens in all_tokens_list:
        tf = compute_tf(tokens)
        tfidf = {}
        for term, tf_val in tf.items():
            if term in idf:
                tfidf[term] = tf_val * idf[term]
        tfidf_vectors.append(tfidf)

    return tfidf_vectors, idf