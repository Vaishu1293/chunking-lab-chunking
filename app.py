"""Knowledge Search Application Entry Point."""

from config import QUERY, DEPARTMENT, ACCESS_LEVEL
from utils.environment import get_setup_components
from utils.display import banner_workers
from workflows.enterprise_workflow import (
    ingest_enterprise_records,
    search_enterprise,
    answer_question,
)
from data.enterprise_data import ENTERPRISE_RECORDS


def main() -> None:
    """Run the Knowledge Search application."""
    banner_workers()
    
    # Initialize environment clients
    _, gemini_client, _, collection = get_setup_components()
    print('Gemini Client: OK' if gemini_client else 'Error connecting Gemini Client')
    print("Collection:", collection.name)
    
    # 1. Ingest Data
    ingest_enterprise_records(gemini_client, collection, ENTERPRISE_RECORDS)
    
    # 2. Search Vector Store
    query = QUERY
    department = DEPARTMENT
    access_level = ACCESS_LEVEL
    print(f"\n[?] Executing Query: '{query}'")
    search_results = search_enterprise(gemini_client, collection, query, department, access_level, top_k=3)
    
    # 3. Generate Grounded Answer
    print("\n--- GROUNDED ANSWER ---")
    answer = answer_question(gemini_client, query, search_results)
    print(answer)


if __name__ == "__main__":
    main()