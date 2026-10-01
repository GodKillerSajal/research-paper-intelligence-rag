"""
Rule-based integrity flags derived from extracted numeric signals.
"""
import re
import math
from collections import Counter
from typing import List, Dict, Any

from app.analysis.extractor import (
    extract_pvalues,
    extract_sample_sizes,
    extract_all_numbers,
    count_statistical_tests,
    mentions_multiple_comparison_correction,
)


def run_rule_flags(full_text: str, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run all rule-based checks on full paper text.
    Returns list of flag dicts:
      {type, severity, title, description, evidence, values_found}
    """
    flags = []
    flags += _flag_pvalue_clustering(full_text)
    flags += _flag_underpowered(full_text)
    flags += _flag_multiple_comparisons(full_text)
    flags += _flag_benford(full_text)
    flags += _flag_vague_significance(full_text)
    return flags


# ── p-value clustering just below 0.05 ───────────────────────────────────────
def _flag_pvalue_clustering(text: str) -> List[Dict]:
    pvals = extract_pvalues(text)
    if len(pvals) < 4:
        return []

    values = [p['value'] for p in pvals]
    just_below = [v for v in values if 0.04 <= v < 0.05]
    just_above = [v for v in values if 0.05 < v <= 0.06]
    sig = [v for v in values if v < 0.05]

    flags = []

    if len(values) >= 4 and len(just_below) >= 2:
        ratio = len(just_below) / max(len(just_above), 1)
        if ratio >= 3:
            flags.append({
                'type': 'P_VALUE_CLUSTERING',
                'severity': 'HIGH',
                'title': 'Suspicious p-value clustering below 0.05',
                'description': (
                    f'{len(just_below)} p-values fall in [0.04, 0.05) vs '
                    f'{len(just_above)} in (0.05, 0.06]. '
                    f'This {ratio:.1f}x asymmetry is a common indicator of p-hacking '
                    f'or selective stopping rules.'
                ),
                'evidence': [p['context'] for p in pvals if 0.04 <= p['value'] < 0.05][:3],
                'values_found': [round(v, 4) for v in values],
            })

    if len(values) >= 5 and len(sig) / len(values) >= 0.85:
        flags.append({
            'type': 'HIGH_SIGNIFICANCE_RATE',
            'severity': 'MEDIUM',
            'title': f'{len(sig)}/{len(values)} results are statistically significant',
            'description': (
                f'{100*len(sig)/len(values):.0f}% of reported p-values are significant (< 0.05). '
                'In real experiments a mix of significant and non-significant results is typical. '
                'A very high rate may indicate selective reporting of only significant outcomes.'
            ),
            'evidence': [p['context'] for p in pvals if p['value'] < 0.05][:3],
            'values_found': [round(v, 4) for v in values],
        })

    return flags


# ── underpowered studies ──────────────────────────────────────────────────────
def _flag_underpowered(text: str) -> List[Dict]:
    sample_sizes = extract_sample_sizes(text)
    if not sample_sizes:
        return []

    pvals = extract_pvalues(text)
    has_sig = any(p['value'] < 0.05 for p in pvals)

    small_samples = [s for s in sample_sizes if s['n'] < 30]
    flags = []
    if small_samples and has_sig:
        n_vals = [s['n'] for s in small_samples]
        flags.append({
            'type': 'SMALL_SAMPLE',
            'severity': 'MEDIUM',
            'title': f'Small sample size(s) with significant claims (n={min(n_vals)})',
            'description': (
                f'Sample size(s) as small as n={min(n_vals)} detected alongside '
                f'statistically significant results. Studies with n < 30 are generally '
                f'underpowered and prone to Type I and Type II errors.'
            ),
            'evidence': [s['context'] for s in small_samples][:2],
            'values_found': n_vals,
        })
    return flags


# ── uncorrected multiple comparisons ─────────────────────────────────────────
def _flag_multiple_comparisons(text: str) -> List[Dict]:
    test_count = count_statistical_tests(text)
    corrected = mentions_multiple_comparison_correction(text)
    pvals = extract_pvalues(text)
    sig_count = sum(1 for p in pvals if p['value'] < 0.05)

    flags = []
    if test_count >= 5 and not corrected:
        flags.append({
            'type': 'MULTIPLE_COMPARISONS',
            'severity': 'HIGH',
            'title': f'{test_count} statistical tests with no multiple-comparison correction',
            'description': (
                f'The paper uses {test_count} different statistical tests '
                f'with {sig_count} significant results, but no mention of '
                f'Bonferroni, FDR, Holm, or other corrections was found. '
                f'With {test_count} tests at α=0.05, the expected number of '
                f'false positives by chance is {test_count * 0.05:.1f}.'
            ),
            'evidence': [],
            'values_found': {'tests': test_count, 'sig_results': sig_count},
        })
    elif test_count >= 3 and not corrected and sig_count >= 3:
        flags.append({
            'type': 'MULTIPLE_COMPARISONS',
            'severity': 'LOW',
            'title': f'{test_count} statistical tests — correction not mentioned',
            'description': (
                f'{test_count} test types detected. Multiple comparison correction '
                f'was not explicitly mentioned.'
            ),
            'evidence': [],
            'values_found': {'tests': test_count},
        })
    return flags


# ── Benford's law on leading digits ──────────────────────────────────────────
def _flag_benford(text: str) -> List[Dict]:
    numbers = extract_all_numbers(text)
    if len(numbers) < 50:
        return []

    expected = {d: math.log10(1 + 1/d) for d in range(1, 10)}

    leading = []
    for n in numbers:
        s = str(abs(n)).lstrip('0').replace('.', '')
        if s and s[0].isdigit() and s[0] != '0':
            leading.append(int(s[0]))

    if len(leading) < 50:
        return []

    observed = Counter(leading)
    total = len(leading)

    chi2 = sum(
        (observed.get(d, 0) - total * expected[d]) ** 2 / (total * expected[d])
        for d in range(1, 10)
    )

    flags = []
    if chi2 > 20.1:
        flags.append({
            'type': 'BENFORD_VIOLATION',
            'severity': 'HIGH',
            'title': "Reported numbers deviate significantly from Benford's Law",
            'description': (
                f'The leading-digit distribution of {total} numeric values deviates '
                f'significantly from Benford\'s Law (χ²={chi2:.1f}, df=8, p<0.01). '
                f'Natural datasets follow Benford\'s Law; strong deviations can indicate '
                f'data fabrication or heavy rounding.'
            ),
            'evidence': [],
            'values_found': {'chi2': round(chi2, 2), 'n_numbers': total},
        })
    elif chi2 > 15.5:
        flags.append({
            'type': 'BENFORD_VIOLATION',
            'severity': 'MEDIUM',
            'title': "Numeric distribution shows mild Benford's Law deviation",
            'description': (
                f'χ²={chi2:.1f} (df=8, p<0.05). Mild deviation from Benford\'s Law '
                f'across {total} reported values.'
            ),
            'evidence': [],
            'values_found': {'chi2': round(chi2, 2), 'n_numbers': total},
        })
    return flags


# ── vague significance language ───────────────────────────────────────────────
def _flag_vague_significance(text: str) -> List[Dict]:
    pattern = re.compile(
        r'trend toward|approached significance|marginally significant|'
        r'nearly significant|borderline significant|almost significant',
        re.IGNORECASE,
    )
    matches = []
    for m in pattern.finditer(text):
        start = max(0, m.start() - 100)
        end = min(len(text), m.end() + 100)
        matches.append(text[start:end].replace('\n', ' ').strip())

    if len(matches) >= 2:
        return [{
            'type': 'VAGUE_SIGNIFICANCE',
            'severity': 'LOW',
            'title': f'{len(matches)} instances of vague significance language',
            'description': (
                f'Phrases like "trend toward significance" or "marginally significant" '
                f'found {len(matches)} times. These often describe p > 0.05 results '
                f'being re-framed as meaningful.'
            ),
            'evidence': matches[:3],
            'values_found': len(matches),
        }]
    return []
