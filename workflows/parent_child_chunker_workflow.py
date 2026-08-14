from config import BANNER_WIDTH, EMBEDDING_MODEL
from chunkers.recursive_chunker import recursive_chunk_text
from services.gemini_service import generate_embedding
from services.similarity import cosine_similarity


def parent_child_chunking(gemini_client):
    document = """
    IT Equipment Policy

    Company laptops are normally replaced every three years.
    Employees must report damaged equipment to the IT department.
    Critical hardware failures may qualify for immediate replacement.
    Old equipment must be returned before a replacement is issued.
    """
    document_id = "policy_001"

    # Step 1: Define parent document metadata
    document_metadata = {
        "source": "IT Equipment Policy",
        "department": "IT",
        "document_type": "policy",
        "year": 2026
    }

    # Step 2: Build Parent Node with nested metadata
    parent = {
        "id": document_id,
        "text": document.strip(),
        "metadata": document_metadata
    }

    print("\nPARENT NODE")
    print("=" * BANNER_WIDTH)
    print(f"ID: {parent['id']}")
    print(f"Text:\n{parent['text']}\n")
    print("Metadata:")
    for key, value in parent["metadata"].items():
        print(f"  {key}: {value}")

    # Verify nested metadata access
    print("\n[Parent Metadata Verification]")
    print(f"Department: {parent['metadata']['department']}")
    print(f"Source:     {parent['metadata']['source']}")

    # Step 3: Create Child Nodes & inherit metadata + relationship info
    children_text = recursive_chunk_text(document, 100, None)

    children = []
    for i, child_text in enumerate(children_text):
        # Child inherits parent document metadata + gets specific index & parent_id
        child_metadata = {
            "parent_id": parent["id"],
            "child_index": i,
            "source": parent["metadata"]["source"],
            "department": parent["metadata"]["department"],
            "document_type": parent["metadata"]["document_type"],
            "year": parent["metadata"]["year"]
        }

        child = {
            "id": f"{parent['id']}_child_{i}",
            "text": child_text,
            "metadata": child_metadata
        }
        children.append(child)

    # Step 4: Debug-inspect Child 3 structure & nested metadata
    sample_child = children[3]
    print("\n" + "=" * BANNER_WIDTH)
    print("DEBUG INSPECTION: CHILD 3")
    print("=" * BANNER_WIDTH)
    print(f"ID:   {sample_child['id']}")
    print(f"Text: '{sample_child['text']}'")
    print("Nested Metadata Access Verification:")
    print(f"  Parent ID:   {sample_child['metadata']['parent_id']}")
    print(f"  Child Index: {sample_child['metadata']['child_index']}")
    print(f"  Department:  {sample_child['metadata']['department']}")
    print(f"  Year:        {sample_child['metadata']['year']}")

    # Step 5: Semantic search over child chunks
    query = "What happens if a laptop has a critical hardware failure?"
    print("\n" + "=" * BANNER_WIDTH)
    print(f"USER QUERY:\n'{query}'")
    print("=" * BANNER_WIDTH)

    query_embedding = generate_embedding(gemini_client, EMBEDDING_MODEL, query)

    scored_children = []

    for child in children:
        child_embedding = generate_embedding(gemini_client, EMBEDDING_MODEL, child["text"])
        similarity = cosine_similarity(query_embedding, child_embedding)

        print(f"Child [{child['metadata']['child_index']}] ID: {child['id']} | Similarity: {similarity:.4f}")

        scored_children.append({
            "child": child,
            "similarity": similarity
        })

    # Step 6: Select winning child
    best_match = max(scored_children, key=lambda item: item["similarity"])
    matched_child = best_match["child"]
    highest_score = best_match["similarity"]

    # Step 7: Print winning child's metadata & follow updated parent_id path
    print("\n" + "=" * BANNER_WIDTH)
    print(f"WINNING CHILD MATCH (Score: {highest_score:.4f})")
    print("=" * BANNER_WIDTH)
    print(f"Matched Child Text:\n'{matched_child['text']}'\n")

    print("Child Metadata:")
    for key, value in matched_child["metadata"].items():
        print(f"  {key}: {value}")

    # Updated parent lookup path via matched_child["metadata"]["parent_id"]
    retrieved_parent_id = matched_child["metadata"]["parent_id"]
    is_valid_parent = retrieved_parent_id == parent["id"]

    print(f"\nParent Relationship Verification: {is_valid_parent}")
    print(f"Retrieved Parent ID: {retrieved_parent_id}")

    print("\nPARENT CONTEXT RETRIEVED FOR LLM:")
    print("-" * BANNER_WIDTH)
    print(f"{parent['text']}\n")

    return parent, children