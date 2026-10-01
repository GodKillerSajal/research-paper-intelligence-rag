import re


def clean_text(text: str) -> str:
    """Normalise extracted PDF text."""
    # Remove page numbers standing alone on a line like "- 7 -" or "7"
    text = re.sub(r'^\s*-?\s*\d+\s*-?\s*$', '', text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Normalize whitespace within lines
    lines = [re.sub(r' {2,}', ' ', line) for line in text.split('\n')]
    return '\n'.join(lines).strip()
