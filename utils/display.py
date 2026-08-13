"""Console display helpers."""

from config import AUTHOR, BANNER_WIDTH, PROJECT_NAME, VERSION


def display_banner() -> None:
    """Display the standard application header."""
    print("=" * BANNER_WIDTH)
    print(PROJECT_NAME)
    print(f"Version: {VERSION}")
    print(f"Author:  {AUTHOR}")
    print("=" * BANNER_WIDTH)


def display_startup_message() -> None:
    """Print the application startup message."""
    print("\nApplication Started Successfully\n")


def banner_workers() -> None:
    """Display the application banner and startup message."""
    display_banner()
    display_startup_message()

def print_chroma_results(results: dict) -> None:
    """Print ranked search results returned by ChromaDB."""
    print("\nChromaDB Semantic Search")
    print("=" * BANNER_WIDTH)

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (doc_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1,
    ):
        print(f"\nRank:     {rank}")
        print(f"ID:       {doc_id}")
        print(f"Document: {document}")
        print(f"Category: {metadata.get('category', 'N/A')}")
        print(f"Source:   {metadata.get('source', 'N/A')}")
        print(f"Distance: {distance}")
