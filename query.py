"""
Query formation and similarity computation.
"""

import math


def build_centroid_query(tfidf_vectors):
    """
    Build a pseudo-query by averaging all sentence vectors.
    This represents the "average" content of the document.

    Args:
        tfidf_vectors: list of dicts [{term: tfidf}, ...]

    Returns:
        dict: {term: avg_weight} representing the centroid
    """
    centroid = {}
    N = len(tfidf_vectors)

    if N == 0:
        return centroid

    for tfidf in tfidf_vectors:
        for term, weight in tfidf.items():
            centroid[term] = centroid.get(term, 0) + weight

    for term in centroid:
        centroid[term] /= N

    return centroid


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two sparse vectors (dicts).

    cos(A, B) = (A · B) / (||A|| * ||B||)

    Args:
        vec1, vec2: dict {term: weight}

    Returns:
        float: cosine similarity score
    """
    dot_product = 0
    for term, weight in vec1.items():
        if term in vec2:
            dot_product += weight * vec2[term]

    mag1 = math.sqrt(sum(w ** 2 for w in vec1.values()))
    mag2 = math.sqrt(sum(w ** 2 for w in vec2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)


def rank_sentences(query_vector, tfidf_vectors, original_sentences):
    """
    Rank sentences by cosine similarity to the query.

    Returns:
        list of tuples: [(sent_id, similarity_score, sentence_text), ...]
        sorted by score descending
    """
    scored = []

    for sent_id, sent_vec in enumerate(tfidf_vectors):
        score = cosine_similarity(query_vector, sent_vec)
        scored.append((sent_id, score, original_sentences[sent_id]))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored