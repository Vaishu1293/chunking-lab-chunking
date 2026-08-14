"""
Semantic Text Chunker.

Splits text based on semantic meaning changes between adjacent sentences 
using sentence-level embeddings and cosine similarity thresholds.
"""

import re
from typing import List, Any
from config import EMBEDDING_MODEL
from services.gemini_service import generate_embedding
from services.similarity import cosine_similarity


def semantic_chunk_text(
    client: Any,
    document: str,
    similarity_threshold: float = 0.60,
) -> List[str]:
    """Splits text into chunks based on semantic similarity between sentences.

    Args:
        client (Any): Configured Gemini client instance.
        document (str): Target text document to chunk.
        similarity_threshold (float): Minimum similarity score to stay in same chunk.

    Returns:
        List[str]: List of merged semantic text chunks.
    """
    if not document or not document.strip():
        return []

    # 1. Split document into sentences and filter out empty strings
    sentences = re.split(r"(?<=[.!?])\s+", document.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    if len(sentences) == 1:
        return [sentences[0]]

    # 2. Generate embeddings for all sentences
    embeddings = []
    for sentence in sentences:
        emb = generate_embedding(client, EMBEDDING_MODEL, sentence)
        embeddings.append(emb)

    # 3. Process adjacent sentences and evaluate semantic boundaries
    chunks: List[str] = []
    current_chunk: List[str] = [sentences[0]]

    for i in range(1, len(sentences)):
        # Compare vector for sentence[i-1] with vector for sentence[i]
        similarity = cosine_similarity(embeddings[i - 1], embeddings[i])

        # Debug print to observe exact semantic similarity transitions
        # print(f"   [Similarity] S{i-1} ↔ S{i}: {similarity:.4f} "
        #       f"{'(BREAK)' if similarity < similarity_threshold else '(KEEP)'}")

        if similarity >= similarity_threshold:
            # Same topic -> Keep sentence in current chunk
            current_chunk.append(sentences[i])
        else:
            # Topic shifted -> Flush current chunk and start a new one
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    # 4. Flush final accumulated chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks