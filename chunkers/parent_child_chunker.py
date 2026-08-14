from chunkers.recursive_chunker import recursive_chunk_text

def create_parent_child_chunks(
    document,
    parent_id,
    source,
    category,
    child_size
):
    child_texts = recursive_chunk_text(document,child_size)
    return