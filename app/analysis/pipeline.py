"""
Integrity analysis orchestration:
Runs rule-based + LLM checks on all pages of a document.
"""
import re
from typing import List, Dict, Any

from app.ingestion.parser import extract_pages
from app.analysis.extractor import extract_pvalues, extract_sample_sizes
from app.analysis.flags import run_rule_flags
from app.analysis.llm_checks import check_harking, check_selective_reporting, check_data_transparency


def analyse_document(pdf_path: str) -> Dict[str, Any]:
    """
    Full integrity analysis pipeline.
    Returns structured report with flags, summary, and statistics.
    """
    pages = extract_pages(pdf_path)
    if not pages:
        return {'error': 'Could not extract text from PDF'}

    full_text = '\n'.join(p['text'] for p in pages)
    filename = pages[0]['filename']
    num_pages = pages[0]['num_pages']

    # ── Extract raw statistics ─────────────────────────────────────────────────
    pvalues = extract_pvalues(full_text)
    sample_sizes = extract_sample_sizes(full_text)

    # ── Rule-based flags ───────────────────────────────────────────────────────
    rule_flags = run_rule_flags(full_text, pages)

    # ── Extract section text for LLM checks ────────────────────────────────────
    methods_text = _extract_section(full_text, [
        'method', 'methodology', 'experimental setup', 'study design',
        'materials and methods', 'data',
    ])
    results_text = _extract_section(full_text, [
        'result', 'findings', 'analysis',
    ])

    # ── LLM-based flags ────────────────────────────────────────────────────────
    llm_flags = []
    llm_flags += check_harking(full_text)
    llm_flags += check_selective_reporting(methods_text, results_text)
    llm_flags += check_data_transparency(full_text)

    all_flags = rule_flags + llm_flags

    # ── Severity counts ────────────────────────────────────────────────────────
    severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for f in all_flags:
        sev = f.get('severity', 'LOW')
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    # ── Overall risk score ─────────────────────────────────────────────────────
    score = severity_counts['HIGH'] * 3 + severity_counts['MEDIUM'] * 1 + severity_counts['LOW'] * 0.3
    if score >= 5:
        overall_risk = 'HIGH'
    elif score >= 2:
        overall_risk = 'MEDIUM'
    elif score >= 0.5:
        overall_risk = 'LOW'
    else:
        overall_risk = 'CLEAN'

    return {
        'filename': filename,
        'num_pages': num_pages,
        'overall_risk': overall_risk,
        'risk_score': round(score, 1),
        'severity_counts': severity_counts,
        'flags': all_flags,
        'statistics': {
            'p_values_found': len(pvalues),
            'p_values': [
                {'value': round(p['value'], 4), 'context': p['context'][:150]}
                for p in pvalues[:10]
            ],
            'sample_sizes_found': [s['n'] for s in sample_sizes[:5]],
        },
    }


def _extract_section(text: str, keywords: List[str]) -> str:
    """
    Heuristically extract a named section's text.
    Looks for a line matching keywords and returns up to 2000 chars after it.
    """
    lines = text.split('\n')
    pat = re.compile(
        r'^(?:\d+\.?\d*\.?\s+)?(' + '|'.join(keywords) + r')',
        re.IGNORECASE,
    )
    for i, line in enumerate(lines):
        if pat.match(line.strip()) and len(line.strip()) < 80:
            section_lines = lines[i:i+80]
            return '\n'.join(section_lines)[:2000]
    return ''
