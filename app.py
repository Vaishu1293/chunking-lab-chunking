"""Semantic Chunking Verification App."""
from utils.environment import get_setup_components
from utils.display import banner_workers
from chunkers.semantic_chunker import semantic_chunk_text
from workflows.parent_child_chunker_workflow import parent_child_chunking

sample_doc = """
Python is widely used for AI development.
NumPy provides numerical array operations.
PyTorch is commonly used to train neural networks.

Full-time employees receive 20 days of paid annual leave per year.
Leave requests must be submitted through the corporate HR portal.
Managers approve requests exceeding standard limits.

Vector databases store high-dimensional embeddings efficiently.
ChromaDB performs efficient vector similarity search for RAG.
"""

def main():
    banner_workers()
    _, gemini_client, _, _ = get_setup_components()
    print("Gemini Client: OK\n")
    # print("--- EVALUATING ADJACENT SENTENCE SIMILARITY ---")
    chunks = semantic_chunk_text(gemini_client, sample_doc, similarity_threshold=0.65)
    # print(f"\n========================================")
    # print(f"SEMANTIC CHUNKS GENERATED ({len(chunks)})")
    # print(f"========================================")
    # for idx, chunk in enumerate(chunks, 1):
    #     print(f"\nChunk {idx}")
    #     print("-" * 40)
    #     print(chunk)
    parent_child_chunking(gemini_client)

if __name__ == "__main__":
    main()
