import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv(
        "GEMINI_API_KEY"
    )
)


def generate_answer(
    query: str,
    contexts: list[str]
):

    try:

        context_text = "\n\n".join(
            contexts
        )

        prompt = f"""
You are answering questions about a research paper.

Context:

{context_text}

Question:

{query}

Instructions:
- Use only the provided context.
- Do not hallucinate.
- If the answer is not present in the context, say:
  'The answer is missing.'

Answer:
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        print(
            f"Gemini Error: {e}"
        )

        return (
            "The language model is currently unavailable. "
            "Please try again later."
        )