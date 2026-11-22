"""Inspect what the RAG layer would feed the evaluator for a given prompt.

The app calls this retrieval automatically; this script is just for eyeballing it.
"""
import os

from document_processor import DocumentProcessor

IMAGES_LOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "app", "imagesdata", "images_log.csv",
)

if __name__ == "__main__":
    processor = DocumentProcessor()
    # Build only if absent. The old version called reset_all() unconditionally,
    # which deleted the index and re-paid for embedding on every single run.
    processor.process_document(IMAGES_LOG)

    query = input("Enter Prompt: ")
    for i, prompt in enumerate(processor.retrieve_prompts(query), 1):
        print(f"\n--- Example {i} ---\n{prompt}")
