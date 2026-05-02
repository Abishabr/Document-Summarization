"""
Inverted index construction and lookup.
"""


def build_inverted_index(tfidf_vectors):
    """
    Build inverted index from TF-IDF sentence vectors.
    Structure: {term: {sentence_id: tfidf_weight}}

    Args:
        tfidf_vectors: list of dicts [{term: tfidf}, ...]

    Returns:
        dict: inverted index
    """
    inverted_index = {}

    for sent_id, tfidf_dict in enumerate(tfidf_vectors):
        for term, weight in tfidf_dict.items():
            if term not in inverted_index:
                inverted_index[term] = {}
            inverted_index[term][sent_id] = weight

    return inverted_index


def search_inverted_index(index, query_terms):
    """
    Retrieve sentence IDs and total weights for a list of query terms.

    Args:
        index: inverted index dict
        query_terms: list of terms to search

    Returns:
        dict: {sentence_id: total_weight}
    """
    results = {}
    for term in query_terms:
        if term in index:
            for sent_id, weight in index[term].items():
                results[sent_id] = results.get(sent_id, 0) + weight
    return results