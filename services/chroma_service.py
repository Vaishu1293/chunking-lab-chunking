"""
ChromaDB Vector Store Service Module.

Provides helper functions for managing persistent ChromaDB instances,
creating collections, inserting documents, querying vector spaces, and 
handling record updates and deletions.
"""

from typing import Any, Dict, List, Optional
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from config import DATABASE_PATH, COLLECTION_NAME


def create_chroma_client() -> ClientAPI:
    """Initializes and returns a persistent ChromaDB disk-backed client.

    Returns:
        ClientAPI: A persistent ChromaDB client instance targeting DATABASE_PATH.
    """
    client = chromadb.PersistentClient(path=DATABASE_PATH)
    return client


def get_knowledge_collection(client: ClientAPI) -> Collection:
    """Retrieves an existing ChromaDB collection or creates it if it doesn't exist.

    Args:
        client (ClientAPI): Active ChromaDB client instance.

    Returns:
        Collection: ChromaDB collection instance matching COLLECTION_NAME.
    """
    collection = client.get_or_create_collection(COLLECTION_NAME)
    return collection


def add_documents(
    collection: Collection,
    ids: List[str],
    documents: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
) -> None:
    """Inserts document payloads, metadata, and embeddings into a collection.

    Args:
        collection (Collection): Target ChromaDB collection.
        ids (List[str]): List of unique string identifiers for each record.
        documents (List[str]): Raw document text chunks.
        embeddings (List[List[float]]): Vector embeddings corresponding to each chunk.
        metadatas (List[Dict[str, Any]]): Key-value metadata dictionaries.
    """
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def get_document_count(collection: Collection) -> int:
    """Gets total count of records stored in the specified collection.

    Args:
        collection (Collection): Target ChromaDB collection.

    Returns:
        int: Number of items present in the collection.
    """
    return collection.count()


def get_all_documents(collection: Collection) -> Dict[str, Any]:
    """Fetches all stored items from the collection.

    Args:
        collection (Collection): Target ChromaDB collection.

    Returns:
        Dict[str, Any]: ChromaDB payload dictionary containing stored ids, documents,
            metadatas, and embeddings.
    """
    return collection.get()


def search_documents(
    collection: Collection,
    query_embedding: List[float],
    number_of_results: int,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Queries the ChromaDB collection for nearest neighbors using vector similarity.

    Args:
        collection (Collection): Target ChromaDB collection.
        query_embedding (List[float]): Vector representation of the search query.
        number_of_results (int): Top-K nearest results to return.
        where (Optional[Dict[str, Any]], optional): Metadata filter conditions. Defaults to None.

    Returns:
        Dict[str, Any]: Query results dictionary containing matched documents, metadatas,
            and similarity distances.
    """
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=number_of_results,
        where=where,
    )


def update_document(
    collection: Collection,
    document_id: str,
    document: str,
    embedding: List[float],
    metadata: Dict[str, Any],
) -> None:
    """Updates an existing document, vector embedding, and metadata record by ID.

    Args:
        collection (Collection): Target ChromaDB collection.
        document_id (str): Unique record ID to update.
        document (str): Updated text document payload.
        embedding (List[float]): Updated vector embedding array.
        metadata (Dict[str, Any]): Updated metadata dictionary.
    """
    collection.update(
        ids=[document_id],
        documents=[document],
        embeddings=[embedding],
        metadatas=[metadata],
    )


def delete_document(collection: Collection, document_id: str) -> None:
    """Deletes a document record from the collection using its unique ID.

    Args:
        collection (Collection): Target ChromaDB collection.
        document_id (str): Unique identifier of the record to remove.
    """
    collection.delete(ids=[document_id])