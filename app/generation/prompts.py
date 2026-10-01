from typing import List, Dict, Any


def build_context_block(sources: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a numbered source block for the LLM prompt.
    """
    lines = []
    for i, src in enumerate(sources, start=1):
        meta = src.get("metadata", {})
        lines.append(
            f"[{i}] Paper: {meta.get('filename', 'Unknown')}\n"
            f"    Page: {meta.get('page', '?')}\n"
            f"    Section: {meta.get('section', 'Unknown')}\n"
            f"    Text: {src['text'][:600]}"
        )
    return "\n\n".join(lines)


SYSTEM_PROMPT = """You are a research assistant specializing in academic papers.

Rules:
1. Answer ONLY using the provided SOURCES below.
2. Every factual claim must be supported by citing [N] where N is the source number.
3. If the sources do not contain sufficient information to answer, respond EXACTLY with:
   INSUFFICIENT_EVIDENCE: <brief reason>
4. Never introduce external knowledge or hallucinate facts.
5. Be precise and technical — this is for a research audience."""


def build_prompt(query: str, sources: List[Dict[str, Any]]) -> str:
    context = build_context_block(sources)
    return f"""{SYSTEM_PROMPT}

SOURCES:
{context}

QUESTION: {query}

ANSWER (cite sources as [N]):"""
