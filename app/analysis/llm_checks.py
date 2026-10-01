"""
LLM-based integrity checks using Gemini.
These require understanding language and structure, not just numbers.
"""
import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        _client = genai.Client(api_key=api_key)
    return _client


def _call_gemini(prompt: str, model: str = "gemini-2.5-flash") -> str:
    client = _get_client()
    try:
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text or ""
    except Exception as e:
        return f"LLM_ERROR: {e}"


def _parse_json(raw: str) -> dict:
    """Attempt to parse JSON from LLM output, stripping markdown fences."""
    clean = raw.strip()
    for fence in ["```json", "```"]:
        if clean.startswith(fence):
            clean = clean[len(fence):]
    clean = clean.rstrip("`").strip()
    return json.loads(clean)


def check_harking(text: str) -> List[Dict[str, Any]]:
    """
    HARKing (Hypothesizing After Results are Known):
    Identify hypotheses that appear to be post-hoc but are framed as a priori.
    """
    prompt = f"""You are a research integrity expert reviewing an academic paper.

Identify any signs of HARKing (Hypothesizing After Results are Known).
HARKing occurs when researchers formulate hypotheses AFTER seeing the data but present them as a priori predictions.

Signs to look for:
- Hypotheses that are very specific and happen to match results exactly
- Predictions stated in a way that implies they were made after results were known
- Results that perfectly match every stated hypothesis without exception
- Exploratory analysis framed as confirmatory

PAPER TEXT (excerpt):
{text[:4000]}

Respond in this exact JSON format (no markdown fences):
{{
  "harking_risk": "HIGH" or "MEDIUM" or "LOW" or "NONE",
  "evidence": ["quote from paper supporting concern"],
  "explanation": "brief explanation"
}}"""

    raw = _call_gemini(prompt)
    try:
        result = _parse_json(raw)
        if result.get('harking_risk') in ('HIGH', 'MEDIUM'):
            return [{
                'type': 'HARKING',
                'severity': result['harking_risk'],
                'title': f"Possible HARKing detected (risk: {result['harking_risk']})",
                'description': result.get('explanation', ''),
                'evidence': result.get('evidence', [])[:3],
                'values_found': None,
            }]
    except Exception:
        pass
    return []


def check_selective_reporting(methods_text: str, results_text: str) -> List[Dict[str, Any]]:
    """
    Compare outcomes listed in Methods vs outcomes reported in Results.
    Selective reporting = outcomes disappear between sections.
    """
    if not methods_text.strip() or not results_text.strip():
        return []

    prompt = f"""You are a research integrity expert.

Compare the METHODS section and RESULTS section of this paper.
Identify any outcomes, variables, or measures mentioned in Methods that are NOT reported in Results.
This is called selective outcome reporting and is a form of publication bias.

METHODS:
{methods_text[:2000]}

RESULTS:
{results_text[:2000]}

Respond in this exact JSON format (no markdown fences):
{{
  "selective_reporting_risk": "HIGH" or "MEDIUM" or "LOW" or "NONE",
  "missing_outcomes": ["outcome mentioned in methods but absent from results"],
  "explanation": "brief explanation"
}}"""

    raw = _call_gemini(prompt)
    try:
        result = _parse_json(raw)
        if result.get('selective_reporting_risk') in ('HIGH', 'MEDIUM'):
            missing = result.get('missing_outcomes', [])
            return [{
                'type': 'SELECTIVE_REPORTING',
                'severity': result['selective_reporting_risk'],
                'title': f"Selective outcome reporting ({len(missing)} missing outcome(s))",
                'description': result.get('explanation', ''),
                'evidence': missing[:5],
                'values_found': None,
            }]
    except Exception:
        pass
    return []


def check_data_transparency(text: str) -> List[Dict[str, Any]]:
    """
    Check for missing data handling, exclusion criteria, and data availability.
    """
    prompt = f"""You are a research integrity expert.

Review the following paper text for data transparency issues:
1. Is missing data handling described?
2. Are exclusion criteria clearly stated?
3. Is data availability mentioned (open data, repository, available on request)?
4. Are there suspicious data patterns (outlier removal without justification)?

PAPER TEXT:
{text[:3500]}

Respond in this exact JSON format (no markdown fences):
{{
  "transparency_issues": [
    {{"issue": "description of issue", "severity": "HIGH" or "MEDIUM" or "LOW"}}
  ],
  "overall_transparency": "HIGH" or "MEDIUM" or "LOW"
}}"""

    raw = _call_gemini(prompt)
    flags = []
    try:
        result = _parse_json(raw)
        for issue in result.get('transparency_issues', []):
            if issue.get('severity') in ('HIGH', 'MEDIUM'):
                flags.append({
                    'type': 'DATA_TRANSPARENCY',
                    'severity': issue['severity'],
                    'title': 'Data transparency issue',
                    'description': issue['issue'],
                    'evidence': [],
                    'values_found': None,
                })
    except Exception:
        pass
    return flags
