"""
Main summarization pipeline.
"""

from tokenizer import split_sentences, tokenize_words
from normalizer import normalize_tokens
from stemmer import get_stemmer
from weighting import compute_tfidf
from indexer import build_inverted_index
from query import build_centroid_query, rank_sentences


class DocumentSummarizer:
    """
    Extractive document summarizer using IR techniques.

    Usage:
        summarizer = DocumentSummarizer()
        summary, metadata = summarizer.summarize(document_text, num_sentences=3)
    """

    def __init__(self):
        self.stemmer = get_stemmer()
        self.inverted_index = None
        self.tfidf_vectors = None
        self.original_sentences = None
        self.tokenized_sentences = None
        self.sentence_scores = None

    def preprocess(self, document):
        """
        Full pipeline: tokenization → normalization → tokenized sentences.
        """
        # 1. TOKENIZATION: Split into sentences
        raw_sentences = split_sentences(document)

        # 2. TOKENIZATION + NORMALIZATION: Tokenize each sentence
        tokenized_sentences = []
        valid_sentences = []

        for sent in raw_sentences:
            tokens = tokenize_words(sent)
            normalized = normalize_tokens(tokens, self.stemmer)
            if normalized:
                tokenized_sentences.append(normalized)
                valid_sentences.append(sent)

        return tokenized_sentences, valid_sentences

    def summarize(self, document, num_sentences=3):
        """
        Generate extractive summary.

        Args:
            document: str - input text
            num_sentences: int - number of sentences in summary

        Returns:
            summary: str - the generated summary
            metadata: dict - inverted_index, scores, etc.
        """
        # Step 1 & 2: Tokenization + Normalization
        self.tokenized_sentences, self.original_sentences = self.preprocess(document)

        # Edge case: document is already short enough
        if len(self.original_sentences) <= num_sentences:
            summary = " ".join(self.original_sentences)
            return summary, {
                "inverted_index": {},
                "tfidf_vectors": [],
                "idf_scores": {},
                "scores": [],
                "query_vector": {},
                "summary": summary,
                "tokenized_sentences": self.tokenized_sentences
            }

        # Step 3: WEIGHTING - Compute TF-IDF
        self.tfidf_vectors, idf_scores = compute_tfidf(self.tokenized_sentences)

        # Step 4: INDEXING - Build inverted index
        self.inverted_index = build_inverted_index(self.tfidf_vectors)

        # Step 5: QUERY - Build centroid pseudo-query
        query_vector = build_centroid_query(self.tfidf_vectors)

        # Step 6: RETRIEVAL - Rank sentences by cosine similarity
        self.sentence_scores = rank_sentences(
            query_vector,
            self.tfidf_vectors,
            self.original_sentences
        )

        # Select top N sentences and restore original order
        top_n = self.sentence_scores[:num_sentences]
        top_n.sort(key=lambda x: x[0])

        summary_sentences = [sent for _, _, sent in top_n]
        summary = " ".join(summary_sentences)

        metadata = {
            "inverted_index": self.inverted_index,
            "tfidf_vectors": self.tfidf_vectors,
            "idf_scores": idf_scores,
            "scores": self.sentence_scores,
            "query_vector": query_vector,
            "summary": summary,
            "tokenized_sentences": self.tokenized_sentences
        }

        return summary, metadata