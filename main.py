"""
Document Summarizer - Entry Point
Uses IR techniques: Tokenization, Normalization, Weighting, Indexing, Query
"""

from summarizer import DocumentSummarizer
from storage import (
    get_user_document_choice,
    save_full_report
)


def print_separator(title=""):
    """Print a formatted separator."""
    print("\n" + "=" * 65)
    if title:
        print(f"  {title}")
        print("=" * 65)


def print_scores(scores):
    """Print ranked sentences with scores."""
    print_separator("SENTENCE RANKINGS (Cosine Similarity)")
    for i, (sent_id, score, sent) in enumerate(scores, 1):
        display = sent[:100] + "..." if len(sent) > 100 else sent
        print(f"\n  Rank {i:2d} | Score: {score:.4f} | Original Position: {sent_id}")
        print(f"  \"{display}\"")


def print_inverted_index(index, max_terms=15):
    """Print a sample of the inverted index."""
    total = len(index)
    show = min(max_terms, total)
    print_separator(f"INVERTED INDEX ({total} unique terms, showing {show})")
    for i, (term, postings) in enumerate(sorted(index.items())):
        if i >= max_terms:
            break
        sent_ids = sorted(postings.keys())
        print(f"  '{term}' → sentences {sent_ids}")


def print_tokenized_sentences(tokenized_sentences):
    """Print tokenized and normalized sentences."""
    print_separator("TOKENIZED & NORMALIZED SENTENCES")
    for i, tokens in enumerate(tokenized_sentences):
        print(f"  [{i}] {tokens}")


def main():
    # ─── GET DOCUMENT ───
    document, source = get_user_document_choice()
    print(f"\n  Source: {source}")

    # ─── GET SUMMARY LENGTH ───
    try:
        num = input("\n  How many sentences in summary? (default: 4): ").strip()
        num_sentences = int(num) if num else 4
    except ValueError:
        num_sentences = 4

    # ─── SUMMARIZE ───
    print("\n  Processing...")
    summarizer = DocumentSummarizer()
    summary, metadata = summarizer.summarize(document, num_sentences=num_sentences)

    # ─── DISPLAY ───
    print_tokenized_sentences(metadata["tokenized_sentences"])
    print_scores(metadata["scores"])
    print_inverted_index(metadata["inverted_index"])

    print_separator("FINAL SUMMARY")
    print(f"  {summary}")
    print_separator()

    # ─── SAVE ───
    print("\n  Saving results to 'output/' folder...")
    save_full_report(
        summary_text=summary,
        inverted_index=metadata["inverted_index"],
        scores=metadata["scores"],
        query_terms=list(metadata["query_vector"].keys())
    )

    print("\n  Done! Check the 'output/' folder for all saved files.\n")


if __name__ == "__main__":
    main()