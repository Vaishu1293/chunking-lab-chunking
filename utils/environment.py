"""Environment and application dependency setup."""

import os
from dotenv import load_dotenv
from services.gemini_service import load_genai_client
from services.chroma_service import create_chroma_client, get_knowledge_collection

def load_api_key() -> str:
    """Load and return the Gemini API key from the .env file."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from the environment.")
    return api_key

def get_setup_components() -> tuple:
    """Create the Gemini client, Chroma client, and requested collection."""
    api_key = load_api_key()
    client = load_genai_client(api_key)
    chroma_client = create_chroma_client()
    collection = get_knowledge_collection(chroma_client)
    return api_key, client, chroma_client, collection
