"""
Extracts numeric signals from paper text:
  - p-values
  - sample sizes
  - effect sizes
  - all numeric values (for Benford's law)
"""
import re
from typing import List, Dict, Any


# ── p-values ──────────────────────────────────────────────────────────────────
_PVAL_PATTERNS = [
    r'p\s*[=<>≤≥]\s*\.?([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)',
    r'p(?:-value)?\s*(?:of|=|<|>|≤|≥)\s*\.?([0-9]+(?:\.[0-9]+)?)',
    r'(?:significance|significant).*?p\s*[=<>]\s*\.?([0-9]+(?:\.[0-9]+)?)',
]


def extract_pvalues(text: str) -> List[Dict[str, Any]]:
    """Return list of {value: float, raw: str, context: str}."""
    results = []
    for pat in _PVAL_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            raw = m.group(0)
            val_str = m.group(1)
            try:
                val = float(val_str if '.' in val_str else '0.' + val_str)
                if 0 < val <= 1:
                    start = max(0, m.start() - 80)
                    end = min(len(text), m.end() + 80)
                    results.append({
                        'value': val,
                        'raw': raw,
                        'context': text[start:end].replace('\n', ' ').strip(),
                    })
            except ValueError:
                continue
    seen = set()
    unique = []
    for r in results:
        key = round(r['value'], 4)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


# ── sample sizes ──────────────────────────────────────────────────────────────
_N_PATTERNS = [
    r'(?:n|N)\s*=\s*([0-9,]+)',
    r'sample\s+(?:size|of)\s+(?:was\s+)?(?:n\s*=\s*)?([0-9,]+)',
    r'(?:total|final)\s+(?:sample|participants|subjects|patients).*?([0-9,]+)',
    r'([0-9,]+)\s+(?:participants|subjects|patients|respondents|observations)',
]


def extract_sample_sizes(text: str) -> List[Dict[str, Any]]:
    results = []
    for pat in _N_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            raw_n = m.group(1).replace(',', '')
            try:
                n = int(raw_n)
                if 2 <= n <= 1_000_000:
                    start = max(0, m.start() - 60)
                    end = min(len(text), m.end() + 60)
                    results.append({
                        'n': n,
                        'raw': m.group(0),
                        'context': text[start:end].replace('\n', ' ').strip(),
                    })
            except ValueError:
                continue
    seen = set()
    unique = []
    for r in results:
        if r['n'] not in seen:
            seen.add(r['n'])
            unique.append(r)
    return unique


# ── all numeric values for Benford's law ──────────────────────────────────────
def extract_all_numbers(text: str) -> List[float]:
    """Extract all numeric values (excluding years, page numbers etc.)."""
    numbers = []
    for m in re.finditer(r'\b([0-9]+(?:\.[0-9]+)?)\b', text):
        try:
            val = float(m.group(1))
            if val >= 10 and not (1900 <= val <= 2100):
                numbers.append(val)
        except ValueError:
            continue
    return numbers


# ── statistical test mentions ─────────────────────────────────────────────────
_TEST_PATTERNS = [
    r't-test', r'chi.?square', r'anova', r'manova', r'ancova',
    r'mann.?whitney', r'wilcoxon', r'kruskal.?wallis', r'pearson',
    r'spearman', r'regression', r'correlation', r'fisher',
    r'log.?rank', r'mcnemar', r'friedman',
]


def count_statistical_tests(text: str) -> int:
    """Count distinct statistical test mentions."""
    count = 0
    lower = text.lower()
    for pat in _TEST_PATTERNS:
        if re.search(pat, lower):
            count += 1
    return count


# ── multiple comparison correction mentions ───────────────────────────────────
_CORRECTION_PATTERNS = [
    r'bonferroni', r'holm', r'fdr', r'false discovery',
    r'multiple.{0,20}compar', r'family.?wise', r'adjusted.{0,10}p',
    r'correction for multiple',
]


def mentions_multiple_comparison_correction(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in _CORRECTION_PATTERNS)
