"""Semantic Chunking Verification App."""
from utils.display import banner_workers
from utils.environment import get_setup_components
from workflows.retrieval_comparison_workflow import run_retrieval_comparison

def main():
    banner_workers()
    _, gemini_client, _, _ = get_setup_components()
    print("Gemini Client: OK\n")
    run_retrieval_comparison(gemini_client)

if __name__ == "__main__":
    main()
