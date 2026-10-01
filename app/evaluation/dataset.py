import json
import os
from typing import List, Dict, Any


def load_dataset(path: str = "evaluation/questions.json") -> List[Dict[str, Any]]:
    """
    Load evaluation dataset.
    Expected format:
    [
      {
        "question": "...",
        "expected_answer": "...",
        "relevant_document": "paper.pdf",
        "relevant_page": 4
      }
    ]
    """
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
