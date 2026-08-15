import json
from typing import List, Dict, Any
from google.genai import types
from pydantic import BaseModel, Field
from services.gemini_service import generate_response


class AgenticChunk(BaseModel):
    topic: str = Field(description="Concise topic label summarizing this grouped chunk")
    text: str = Field(description="Exact verbatim text of grouped paragraphs")


class AgenticChunkResponse(BaseModel):
    chunks: List[AgenticChunk]


def agentic_chunk_text(gemini_client, model: str, document: str, system_instruction: str) -> List[Dict[str, Any]]:
    """Groups document paragraphs into semantic chunks using an LLM reasoning prompt.

    Args:
        gemini_client: Active Google GenAI client instance.
        model (str): Gemini model identifier.
        document (str): Target text document to chunk.
        system_instruction (str): Prompt instructions for semantic segmentation.

    Returns:
        List[Dict[str, Any]]: List of dictionary chunks with 'topic' and 'text' keys.
    """
    prompt = f"""
{system_instruction}

---
DOCUMENT TO CHUNK:
{document.strip()}
"""

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=AgenticChunkResponse,
        temperature=0.0
    )

    response_text = generate_response(gemini_client, model, prompt, config=config)

    # Convert structured response into standard Python dictionaries
    parsed_json = json.loads(response_text)
    return parsed_json.get("chunks", [])