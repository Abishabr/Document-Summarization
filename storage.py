"""
Document storage and I/O operations.
"""

import os
import json


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def ensure_directories():
    """Create data/ and output/ directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Document Input Methods ────────────────────────

def get_sample_document():
    """Built-in sample document for testing."""
    return """
    Information retrieval is the science of searching for information in documents.
    It involves searching through large collections of text to find relevant content.
    Modern search engines like Google use sophisticated IR techniques.
    Document summarization is an important application of information retrieval.
    Summarization helps users quickly understand large texts without reading everything.
    There are two main types of summarization: extractive and abstractive.
    Extractive summarization selects important sentences from the original text.
    Abstractive summarization generates new sentences that capture the meaning.
    TF-IDF is a common weighting scheme used in many IR systems.
    An inverted index maps terms to the documents that contain them.
    """


def read_from_console():
    """
    Let user type or paste a document directly in the console.
    User presses Enter twice (empty line) to finish.
    """
    print("\n" + "=" * 60)
    print("  PASTE / TYPE YOUR DOCUMENT BELOW")
    print("  Press ENTER TWICE (empty line) when done")
    print("=" * 60)
    print()

    lines = []
    empty_count = 0

    while True:
        line = input()
        if line.strip() == "":
            empty_count += 1
        else:
            empty_count = 0

        lines.append(line)

        if empty_count >= 2:
            break

    document = "\n".join(lines[:-2])

    if not document.strip():
        print("\n  ⚠️  No text entered. Using sample document instead.")
        return get_sample_document(), "sample (fallback)"

    print(f"\n  ✅ Document received: {len(document)} characters")
    return document, "console input"


def load_from_file(filepath):
    """Load document from a text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read(), filepath


def get_user_document_choice():
    """
    Present options to the user and return the document.
    """
    print("\n" + "=" * 60)
    print("  DOCUMENT SUMMARIZER")
    print("=" * 60)
    print("\n  How would you like to input your document?")
    print("    1. Type or paste in console")
    print("    2. Load from a file")
    print("    3. Use built-in sample document")
    print()

    while True:
        choice = input("  Enter choice (1/2/3): ").strip()

        if choice == "1":
            return read_from_console()

        elif choice == "2":
            filepath = input("  Enter file path: ").strip()
            if os.path.exists(filepath):
                return load_from_file(filepath)
            else:
                print(f"\n  File not found: {filepath}")
                print("  Try again.\n")
                continue

        elif choice == "3":
            return get_sample_document(), "built-in sample"

        else:
            print("  Invalid choice. Please enter 1, 2, or 3.\n")


# ─── Saving Results ────────────────────────────────

def save_summary(summary_text, filename="summary.txt"):
    """Save generated summary to output/ folder."""
    ensure_directories()
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"  Summary saved to: {filepath}")
    return filepath


def save_inverted_index(inverted_index, filename="inverted_index.json"):
    """Save inverted index as JSON."""
    ensure_directories()
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(inverted_index, f, indent=2)
    print(f"  Inverted index saved to: {filepath}")
    return filepath


def save_scores(scores, filename="sentence_scores.json"):
    """Save sentence scores as JSON."""
    ensure_directories()
    filepath = os.path.join(OUTPUT_DIR, filename)
    scores_data = [
        {
            "sentence_id": sent_id,
            "score": round(score, 4),
            "text": text
        }
        for sent_id, score, text in scores
    ]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(scores_data, f, indent=2)
    print(f"  Scores saved to: {filepath}")
    return filepath


def save_full_report(summary_text, inverted_index, scores, query_terms=None):
    """Save complete report with all results."""
    ensure_directories()
    save_summary(summary_text)
    save_inverted_index(inverted_index)
    save_scores(scores)

    report = {
        "summary": summary_text,
        "total_document_sentences": len(scores),
        "query_terms": query_terms or [],
        "ranked_sentences": [
            {
                "rank": i + 1,
                "sentence_id": sent_id,
                "score": round(score, 4),
                "text": text
            }
            for i, (sent_id, score, text) in enumerate(scores)
        ]
    }

    report_path = os.path.join(OUTPUT_DIR, "full_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"  Full report saved to: {report_path}")
    return report_path