from config import DOCUMENT, SYSTEM_INSTRUCTION, MODEL, BANNER_WIDTH, EMBEDDING_MODEL
from chunkers.recursive_chunker import recursive_chunk_text
from chunkers.semantic_chunker import semantic_chunk_text
from chunkers.agentic_chunker import agentic_chunk_text
from services.gemini_service import generate_embedding
from services.similarity import cosine_similarity


def find_best_chunk(gemini_client, query, chunks):
    query_embedding = generate_embedding(gemini_client, EMBEDDING_MODEL, query)
    best_chunk = None
    best_score = -1.0

    for chunk in chunks:
        # Step 3: Handle str, dict, or object structures for chunk text extraction
        if isinstance(chunk, str):
            chunk_text = chunk
        elif isinstance(chunk, dict):
            # Tries common key names used by agentic chunkers ('text', 'content', 'chunk')
            chunk_text = chunk.get("text") or chunk.get("content") or chunk.get("chunk") or str(chunk)
        else:
            chunk_text = getattr(chunk, "text", str(chunk))

        chunk_embedding = generate_embedding(gemini_client, EMBEDDING_MODEL, chunk_text)
        similarity = cosine_similarity(query_embedding, chunk_embedding)

        if similarity > best_score:
            best_score = similarity
            best_chunk = chunk

    return best_chunk, best_score


def run_retrieval_comparison(gemini_client):
    query = "How is the AI platform deployed and monitored?"

    print("\n" + "=" * BANNER_WIDTH)
    print("RETRIEVAL STRATEGY COMPARISON")
    print("=" * BANNER_WIDTH)
    print(f"QUERY: '{query}'\n")

    # Strategy 1: Recursive Chunking
    recursive_chunks = recursive_chunk_text(DOCUMENT, chunk_size=300)
    print("Recursive finished:", type(recursive_chunks))

    # Strategy 2: Semantic Chunking
    semantic_chunks = semantic_chunk_text(gemini_client, DOCUMENT, similarity_threshold=0.65)
    print("Semantic finished:", type(semantic_chunks))

    # Strategy 3: Agentic Chunking
    agentic_chunks = agentic_chunk_text(
        gemini_client=gemini_client,
        model=MODEL,
        document=DOCUMENT,
        system_instruction=SYSTEM_INSTRUCTION
    )
    print("Agentic finished:", type(agentic_chunks))

    # Chunk counts summary
    print("-" * BANNER_WIDTH)
    print("CHUNK GENERATION SUMMARY:")
    print("-" * BANNER_WIDTH)
    print(f"Recursive Chunks: {len(recursive_chunks)}")
    print(f"Semantic Chunks:  {len(semantic_chunks)}")
    print(f"Agentic Chunks:   {len(agentic_chunks)}")
    print("=" * BANNER_WIDTH)

    # Step 4: Evaluate best chunk for each strategy
    recursive_best, recursive_score = find_best_chunk(gemini_client, query, recursive_chunks)
    semantic_best, semantic_score = find_best_chunk(gemini_client, query, semantic_chunks)
    agentic_best, agentic_score = find_best_chunk(gemini_client, query, agentic_chunks)

    # Step 5: Print retrieval scores
    print("\n" + "=" * BANNER_WIDTH)
    print("RETRIEVAL RESULTS")
    print("=" * BANNER_WIDTH)
    print(f"Recursive Best Score: {recursive_score:.4f}")
    print(f"Semantic Best Score:  {semantic_score:.4f}")
    print(f"Agentic Best Score:   {agentic_score:.4f}")
    print("=" * BANNER_WIDTH)

    return