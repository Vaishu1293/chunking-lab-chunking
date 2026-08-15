from config import SAMPLE_DOC
from chunkers.semantic_chunker import semantic_chunk_text

def semantic_chunking_workflow(gemini_client):
    print("--- EVALUATING ADJACENT SENTENCE SIMILARITY ---")
    chunks = semantic_chunk_text(gemini_client, SAMPLE_DOC, similarity_threshold=0.65)
    print(f"\n========================================")
    print(f"SEMANTIC CHUNKS GENERATED ({len(chunks)})")
    print(f"========================================")
    for idx, chunk in enumerate(chunks, 1):
        print(f"\nChunk {idx}")
        print("-" * 40)
        print(chunk)