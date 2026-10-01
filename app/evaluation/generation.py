from typing import Tuple


def answer_relevance(answer: str, question: str) -> float:
    """
    Heuristic relevance score.
    A production system would use an LLM-as-judge here.
    For now: simple length + keyword overlap heuristic.
    Returns score in [0, 1].
    """
    if not answer or len(answer.strip()) < 10:
        return 0.0
    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())
    overlap = len(question_words & answer_words) / max(len(question_words), 1)
    return min(1.0, overlap * 2.5)  # scale to [0,1]


def faithfulness(answer: str, contexts: list) -> float:
    """
    Heuristic faithfulness: fraction of answer sentences
    that have at least some token overlap with retrieved context.
    """
    if not answer or not contexts:
        return 0.0

    context_text = " ".join(contexts).lower()
    context_words = set(context_text.split())

    sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 10]
    if not sentences:
        return 1.0  # short answer, can't penalize

    grounded = 0
    for sent in sentences:
        sent_words = set(sent.lower().split())
        overlap = len(sent_words & context_words)
        if overlap >= 3:  # at least 3 shared tokens
            grounded += 1

    return grounded / len(sentences)
