"""Semantic Chunking Verification App."""
from utils.environment import get_setup_components
from utils.display import banner_workers
from workflows.parent_child_chunker_workflow import parent_child_chunking
from workflows.semantic_chunking_workflow import semantic_chunking_workflow
from workflows.agentic_chunking_workflow import agentic_chunking

def main():
    banner_workers()
    _, gemini_client, _, _ = get_setup_components()
    print("Gemini Client: OK\n")
    # semantic_chunking_workflow(gemini_client)
    # parent_child_chunking(gemini_client)
    agentic_chunking(gemini_client)
   

if __name__ == "__main__":
    main()
