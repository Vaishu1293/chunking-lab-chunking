from config import SYSTEM_INSTRUCTION, DOCUMENT, MODEL, BANNER_WIDTH
from chunkers.agentic_chunker import agentic_chunk_text


def agentic_chunking(gemini_client):
    """Workflow function to execute and output Agentic Chunking."""
    
    # 1. Call agentic chunker to get parsed chunks
    chunks = agentic_chunk_text(
        gemini_client=gemini_client,
        model=MODEL,
        document=DOCUMENT,
        system_instruction=SYSTEM_INSTRUCTION
    )

    # 2. Print output using returned list
    print("\n" + "=" * BANNER_WIDTH)
    print(f"AGENTIC CHUNKS GENERATED ({len(chunks)})")
    print("=" * BANNER_WIDTH)

    for i, chunk in enumerate(chunks):
        print(f"\n[Chunk {i}] Topic: {chunk['topic']}")
        print("-" * BANNER_WIDTH)
        print(chunk['text'])

    return chunks