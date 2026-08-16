from config import SAMPLE_DOC
from chunkers.recursive_chunker import recursive_chunk_text


def recursive_chunking_workflow():
    """Workflow function to execute and display Recursive Chunking results."""
    print("--- EVALUATING RECURSIVE TEXT SPLITTING ---")

    # Call recursive_chunk_text with specified target chunk size and overlap
    chunks = recursive_chunk_text(SAMPLE_DOC, chunk_size=300, chunk_overlap=50)

    print(f"\n========================================")
    print(f"RECURSIVE CHUNKS GENERATED ({len(chunks)})")
    print(f"========================================")

    for idx, chunk in enumerate(chunks, 1):
        print(f"\nChunk {idx}")
        print("-" * 40)
        print(chunk)

    return chunks