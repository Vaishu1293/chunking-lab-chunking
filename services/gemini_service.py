"""
Gemini API Service Module.

Provides wrapper functions for initializing the official Google GenAI Client,
generating completions from text prompts, and computing dense vector embeddings.
"""

from typing import List
from google import genai


def load_genai_client(api_key: str) -> genai.Client:
    """Initializes and returns a Google GenAI client instance.

    Args:
        api_key (str): Valid Google Gemini API Key.

    Returns:
        genai.Client: Authenticated Google GenAI SDK Client instance.
    """
    client = genai.Client(api_key=api_key)
    return client


def generate_response(client: genai.Client, model: str, contents: str, config) -> str:
    """Generates a text completion response using a Gemini LLM.

    Args:
        client (genai.Client): Active Google GenAI client instance.
        model (str): Name of the target Gemini model (e.g., 'gemini-2.5-flash').
        contents (str): Formatted user prompt or context payload.

    Returns:
        str: Generated text response from the model.
    """
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config
    )
    return response.text


def generate_embedding(client: genai.Client, model: str, contents: str) -> List[float]:
    """Generates a dense numerical vector embedding for the input text.

    Args:
        client (genai.Client): Active Google GenAI client instance.
        model (str): Name of the embedding model (e.g., 'text-embedding-004').
        contents (str): Target text string to vectorize.

    Returns:
        List[float]: High-dimensional floating-point vector representation.
    """
    response = client.models.embed_content(
        model=model,
        contents=contents
    )
    return response.embeddings[0].values