"""
Recursive Character Text Chunker with Merging and Overlap.

Recursively splits text into chunks using a hierarchical list of separators,
merges adjacent small pieces, and adds context overlaps between adjacent chunks.
"""

from typing import List, Optional


def merge_chunks(
    pieces: List[str],
    chunk_size: int,
    separator: str = " ",
) -> List[str]:
    """Merges smaller text pieces into larger chunks up to a maximum size limit."""
    merged_chunks: List[str] = []
    current_chunk: str = ""

    for piece in pieces:
        if current_chunk:
            candidate = current_chunk + separator + piece
        else:
            candidate = piece

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk:
                merged_chunks.append(current_chunk)
            current_chunk = piece

    if current_chunk:
        merged_chunks.append(current_chunk)

    return merged_chunks


def recursive_chunk_text(
    text: Optional[str],
    chunk_size: int,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """Recursively chunks text using hierarchical separators."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    if not text or not text.strip():
        return []

    text = text.strip()

    if len(text) <= chunk_size:
        return [text]

    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator = separators[0]
    remaining_separators = separators[1:]

    if separator == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    pieces = text.split(separator)
    results: List[str] = []

    for piece in pieces:
        piece = piece.strip()

        if not piece:
            continue

        if len(piece) <= chunk_size:
            results.append(piece)
        else:
            smaller_chunks = recursive_chunk_text(
                text=piece,
                chunk_size=chunk_size,
                separators=remaining_separators,
            )
            results.extend(smaller_chunks)

    return results


def add_overlap(
    chunks: List[str],
    chunk_overlap: int,
    separator: str = " ",
) -> List[str]:
    """Applies a sliding context overlap to adjacent text chunks."""
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")

    if not chunks:
        return []

    if chunk_overlap == 0:
        return chunks.copy()

    results: List[str] = [chunks[0]]

    for i in range(1, len(chunks)):
        previous_chunk = chunks[i - 1]
        current_chunk = chunks[i]

        overlap_text = previous_chunk[-chunk_overlap:]
        combined = overlap_text + separator + current_chunk
        results.append(combined)

    return results