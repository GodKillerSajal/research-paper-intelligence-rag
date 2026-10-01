import os
import time
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from google import genai

from app.generation.prompts import build_prompt

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")
        _client = genai.Client(api_key=api_key)
    return _client


def generate_answer(
    query: str,
    sources: List[Dict[str, Any]],
    model: str = "gemini-2.5-flash",
) -> Tuple[str, float]:
    """
    Generate a grounded answer from retrieved sources.

    Returns:
        (answer_text, generation_latency_ms)
    """
    client = _get_client()
    prompt = build_prompt(query, sources)

    start = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        answer = response.text or ""
    except Exception as e:
        answer = f"LLM_ERROR: {e}"
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000

    return answer, elapsed_ms


def is_insufficient(answer: str) -> bool:
    """Check if LLM flagged insufficient evidence."""
    return answer.strip().startswith("INSUFFICIENT_EVIDENCE")
