#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quality Analyzer, Anonymizer, and Strategy Feedback Loop
=========================================================

Provides:
  1) Quality metrics for MAP claims and REDUCE findings
  2) Anonymization of sensitive content for online model feedback
  3) Strategy loop: send metrics + anonymized samples to Groq/online API,
     receive improved prompts and recommendations

Usage (standalone):
  python batch_quality.py analyze runs/20260217_183537_a08047
  python batch_quality.py anonymize runs/20260217_183537_a08047 --samples 5
  python batch_quality.py feedback runs/20260217_183537_a08047 --groq-key $GROQ_API_KEY

Usage (as library):
  from batch_quality import analyze_claims, analyze_findings, anonymize_text
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Quality Analyzer
# ---------------------------------------------------------------------------

def analyze_claims(claims_path: str) -> Dict[str, Any]:
    """Compute quality metrics for MAP claims (claims.jsonl)."""
    if not os.path.exists(claims_path):
        return {"error": f"File not found: {claims_path}"}

    claims = []
    with open(claims_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                claims.append(json.loads(line))

    if not claims:
        return {"total": 0, "error": "No claims found"}

    total = len(claims)

    # Field completeness
    has_finding = sum(1 for c in claims if c.get("finding_candidate"))
    has_recommendation = sum(1 for c in claims if c.get("recommendation_candidate"))
    has_evidence = sum(1 for c in claims if c.get("evidence_quote"))
    has_category = sum(1 for c in claims if c.get("category"))
    has_impact = sum(1 for c in claims if c.get("impact"))

    # Confidence distribution
    confidences = [float(c.get("confidence", 0) or 0) for c in claims]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    high_conf = sum(1 for c in confidences if c >= 0.8)
    low_conf = sum(1 for c in confidences if c < 0.5)

    # Category distribution
    categories = Counter()
    for c in claims:
        cat = c.get("category", "unknown")
        if isinstance(cat, list):
            cat = "|".join(cat)
        categories[str(cat)] += 1

    # Impact distribution
    impacts = Counter()
    for c in claims:
        imp = c.get("impact", "unknown")
        if isinstance(imp, list):
            imp = "|".join(imp)
        impacts[str(imp)] += 1

    # Topic distribution
    topics = Counter(str(c.get("topic", "unknown")) for c in claims)

    # Duplicate detection (by finding_candidate text)
    finding_texts = [c.get("finding_candidate", "") for c in claims if c.get("finding_candidate")]
    unique_findings = len(set(finding_texts))
    duplicate_findings = len(finding_texts) - unique_findings

    # Path diversity (how many unique documents)
    paths = set(c.get("path", "") for c in claims if c.get("path"))

    return {
        "total_claims": total,
        "field_completeness": {
            "has_finding": has_finding,
            "has_finding_pct": round(100 * has_finding / total, 1),
            "has_recommendation": has_recommendation,
            "has_recommendation_pct": round(100 * has_recommendation / total, 1),
            "has_evidence": has_evidence,
            "has_evidence_pct": round(100 * has_evidence / total, 1),
            "has_category": has_category,
            "has_category_pct": round(100 * has_category / total, 1),
            "has_impact": has_impact,
            "has_impact_pct": round(100 * has_impact / total, 1),
        },
        "confidence": {
            "avg": round(avg_conf, 3),
            "high_gte_0.8": high_conf,
            "low_lt_0.5": low_conf,
        },
        "categories": dict(categories.most_common(20)),
        "impacts": dict(impacts.most_common()),
        "topics": dict(topics.most_common(15)),
        "diversity": {
            "unique_documents": len(paths),
            "unique_finding_texts": unique_findings,
            "duplicate_findings": duplicate_findings,
            "duplicate_pct": round(100 * duplicate_findings / max(len(finding_texts), 1), 1),
        },
    }


def analyze_findings(findings_path: str) -> Dict[str, Any]:
    """Compute quality metrics for REDUCE findings (findings.json)."""
    if not os.path.exists(findings_path):
        return {"error": f"File not found: {findings_path}"}

    with open(findings_path, "r", encoding="utf-8") as f:
        findings = json.load(f)

    if not isinstance(findings, list) or not findings:
        return {"total": 0, "error": "No findings or invalid format"}

    total = len(findings)

    # Field completeness
    has_title = sum(1 for f in findings if f.get("title"))
    has_statement = sum(1 for f in findings if f.get("statement"))
    has_recommendation = sum(1 for f in findings if f.get("recommendation"))
    has_evidence = sum(1 for f in findings if f.get("evidence") and len(f["evidence"]) > 0)
    has_category = sum(1 for f in findings if f.get("category"))

    # Evidence quality
    evidence_counts = []
    unique_evidence_paths = set()
    has_quotes = 0
    for f in findings:
        ev = f.get("evidence", [])
        if isinstance(ev, list):
            evidence_counts.append(len(ev))
            for e in ev:
                if isinstance(e, dict):
                    p = e.get("path", e.get("doc", ""))
                    if p:
                        unique_evidence_paths.add(p)
                    if e.get("quote"):
                        has_quotes += 1
        else:
            evidence_counts.append(0)

    avg_evidence = sum(evidence_counts) / len(evidence_counts) if evidence_counts else 0
    no_evidence = sum(1 for c in evidence_counts if c == 0)

    # Confidence
    confidences = [float(f.get("confidence", 0) or 0) for f in findings]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    # Category distribution
    categories = Counter()
    for f in findings:
        cat = f.get("category", "unknown")
        if isinstance(cat, list):
            cat = "|".join(cat)
        categories[str(cat)] += 1

    # Impact distribution
    impacts = Counter()
    for f in findings:
        imp = f.get("impact", "unknown")
        if isinstance(imp, list):
            imp = "|".join(imp)
        impacts[str(imp)] += 1

    # Title duplicate detection
    titles = [str(f.get("title", "")) for f in findings if f.get("title")]
    unique_titles = len(set(titles))
    dup_titles = len(titles) - unique_titles

    # Statement similarity (simple: exact duplicates)
    statements = [str(f.get("statement", "")) for f in findings if f.get("statement")]
    unique_statements = len(set(statements))
    dup_statements = len(statements) - unique_statements

    # Top evidence path (most cited document)
    ev_path_counter = Counter()
    for f in findings:
        ev = f.get("evidence", [])
        if isinstance(ev, list):
            for e in ev:
                if isinstance(e, dict):
                    p = e.get("path", e.get("doc", ""))
                    if p:
                        # Normalize: take last 60 chars
                        ev_path_counter[p[-60:]] += 1

    return {
        "total_findings": total,
        "field_completeness": {
            "has_title": has_title,
            "has_title_pct": round(100 * has_title / total, 1),
            "has_statement": has_statement,
            "has_statement_pct": round(100 * has_statement / total, 1),
            "has_recommendation": has_recommendation,
            "has_recommendation_pct": round(100 * has_recommendation / total, 1),
            "has_evidence": has_evidence,
            "has_evidence_pct": round(100 * has_evidence / total, 1),
            "has_category": has_category,
            "has_category_pct": round(100 * has_category / total, 1),
        },
        "evidence_quality": {
            "avg_evidence_per_finding": round(avg_evidence, 1),
            "no_evidence_count": no_evidence,
            "no_evidence_pct": round(100 * no_evidence / total, 1),
            "unique_evidence_documents": len(unique_evidence_paths),
            "total_quotes": has_quotes,
            "top_cited_documents": dict(ev_path_counter.most_common(5)),
        },
        "confidence": {
            "avg": round(avg_conf, 3),
        },
        "categories": dict(categories.most_common(15)),
        "impacts": dict(impacts.most_common()),
        "duplicates": {
            "duplicate_titles": dup_titles,
            "duplicate_titles_pct": round(100 * dup_titles / max(len(titles), 1), 1),
            "duplicate_statements": dup_statements,
            "duplicate_statements_pct": round(100 * dup_statements / max(len(statements), 1), 1),
        },
    }


def quality_report(run_dir: str) -> Dict[str, Any]:
    """Full quality report for a batch run directory."""
    claims_path = os.path.join(run_dir, "claims.jsonl")
    findings_path = os.path.join(run_dir, "findings.json")
    errors_path = os.path.join(run_dir, "map_errors.jsonl")

    report = {"run_dir": run_dir}

    if os.path.exists(claims_path):
        report["claims"] = analyze_claims(claims_path)

    if os.path.exists(findings_path):
        report["findings"] = analyze_findings(findings_path)

    if os.path.exists(errors_path):
        errors = []
        with open(errors_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    errors.append(json.loads(line))
        report["map_errors"] = {
            "total": len(errors),
            "sample": errors[:5],
        }

    return report


# ---------------------------------------------------------------------------
# Anonymizer
# ---------------------------------------------------------------------------

# Known entities to mask (extend as needed)
_COMPANY_PATTERNS = [
    r"\bBLS\b", r"\bSBB\b", r"\bBAV\b", r"\bCommScope\b", r"\bEnotrac\b",
    r"\bENOTRAC\b", r"\bIMST\b", r"\bPrecisionwave\b", r"\bElbatec\b",
    r"\bRhomberg\b", r"\bRBT\b", r"\bPMC\b", r"\bFrefel\b", r"\bAxpo\b",
    r"\bBKW\b", r"\bSwisscom\b", r"\bHuber\s*\+\s*Suhner\b",
]

_PERSON_PATTERNS = [
    # Common Swiss name patterns (first + last)
    r"\b[A-ZÄÖÜ][a-zäöü]+\s+[A-ZÄÖÜ][a-zäöü]+(?:er|li|mann|ger|ner|chi)\b",
]

_AMOUNT_PATTERNS = [
    # CHF amounts
    r"CHF\s*[\d',\.]+(?:\s*(?:Mio|Tsd|k))?\b",
    r"[\d',\.]+\s*(?:CHF|Fr\.)",
    # Bare large numbers that are likely amounts
    r"\b\d{1,3}(?:['\u2019,]\d{3})+(?:\.\d{2})?\b",
]

_EMAIL_PATTERNS = [
    r"\b[\w.+-]+@[\w.-]+\.\w{2,}\b",
]

_DATE_PATTERNS = [
    r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
]


def anonymize_text(text: str, mask_companies: bool = True,
                   mask_amounts: bool = True, mask_emails: bool = True,
                   mask_dates: bool = False) -> str:
    """Replace sensitive entities with placeholders.
    
    Returns anonymized text safe to send to online models.
    """
    result = text

    if mask_companies:
        company_counter = {}
        company_idx = [0]
        def _replace_company(m):
            name = m.group(0)
            key = name.upper()
            if key not in company_counter:
                company_idx[0] += 1
                company_counter[key] = f"[FIRMA_{chr(64 + company_idx[0])}]"
            return company_counter[key]

        for pat in _COMPANY_PATTERNS:
            result = re.sub(pat, _replace_company, result)

    if mask_amounts:
        result = re.sub(
            r"CHF\s*[\d',\.]+(?:\s*(?:Mio|Tsd|k))?",
            "[BETRAG]", result
        )
        result = re.sub(
            r"[\d',\.]+\s*(?:CHF|Fr\.)",
            "[BETRAG]", result
        )

    if mask_emails:
        result = re.sub(
            r"\b[\w.+-]+@[\w.-]+\.\w{2,}\b",
            "[EMAIL]", result
        )

    if mask_dates:
        result = re.sub(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b", "[DATUM]", result)
        result = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "[DATUM]", result)

    return result


def anonymize_finding(finding: Dict[str, Any]) -> Dict[str, Any]:
    """Anonymize a single finding for safe online transmission."""
    anon = {}
    for key in ["title", "statement", "recommendation"]:
        if finding.get(key):
            anon[key] = anonymize_text(str(finding[key]))
    anon["category"] = finding.get("category", "unknown")
    anon["impact"] = finding.get("impact", "unknown")
    anon["confidence"] = finding.get("confidence", 0)

    # Anonymize evidence: keep structure, mask content
    ev = finding.get("evidence", [])
    anon_ev = []
    if isinstance(ev, list):
        for e in ev:
            if isinstance(e, dict):
                anon_e = {
                    "doc_type": _classify_doc_type(e.get("path", "")),
                }
                if e.get("quote"):
                    anon_e["quote_preview"] = anonymize_text(str(e["quote"])[:100])
                anon_ev.append(anon_e)
    anon["evidence"] = anon_ev
    return anon


def _classify_doc_type(path: str) -> str:
    """Classify document type from path without revealing the path."""
    path_lower = path.lower()
    if ".eml" in path_lower or "mail" in path_lower:
        return "email"
    if ".pdf" in path_lower:
        return "pdf"
    if ".xlsx" in path_lower or ".xls" in path_lower:
        return "spreadsheet"
    if ".docx" in path_lower or ".doc" in path_lower:
        return "document"
    if "protokoll" in path_lower:
        return "protocol"
    return "other"


def anonymize_samples(findings_path: str, n: int = 5,
                      strategy: str = "diverse") -> List[Dict[str, Any]]:
    """Extract and anonymize N sample findings for online feedback.
    
    strategy:
      'diverse' - pick from different categories and impacts
      'random'  - random sample
      'worst'   - lowest confidence (likely quality issues)
    """
    with open(findings_path, "r", encoding="utf-8") as f:
        findings = json.load(f)

    if not findings:
        return []

    if strategy == "worst":
        findings.sort(key=lambda f: float(f.get("confidence", 0) or 0))
        selected = findings[:n]
    elif strategy == "diverse":
        # Pick from different categories
        by_cat = {}
        for f in findings:
            cat = str(f.get("category", "unknown"))
            by_cat.setdefault(cat, []).append(f)
        selected = []
        cats = list(by_cat.keys())
        idx = 0
        while len(selected) < n and idx < len(findings):
            cat = cats[idx % len(cats)]
            if by_cat[cat]:
                selected.append(by_cat[cat].pop(0))
            idx += 1
    else:
        import random
        random.seed(42)
        selected = random.sample(findings, min(n, len(findings)))

    return [anonymize_finding(f) for f in selected]


# ---------------------------------------------------------------------------
# Strategy Feedback Loop (Groq API)
# ---------------------------------------------------------------------------

STRATEGY_SYSTEM = """You are a senior data quality consultant reviewing the output of a batch 
document analysis pipeline. The pipeline extracts structured findings from project documents 
using a local LLM.

You receive:
1. Quality metrics (statistics about the extraction results)
2. Anonymized samples (real findings with sensitive data masked)
3. The current prompts used for extraction

Your task:
- Identify quality issues (duplicates, missing fields, weak evidence, etc.)
- Suggest specific prompt improvements  
- Recommend processing parameter changes
- Rate the overall quality (1-10)

Be specific and actionable. Focus on what can be improved in the PROMPTS and PARAMETERS,
not in the source data. Answer in English."""

STRATEGY_USER_TEMPLATE = """## Quality Metrics
```json
{metrics_json}
```

## Anonymized Samples ({n_samples} findings)
```json
{samples_json}
```

## Current REDUCE Prompt
```
{reduce_prompt}
```

## Current MAP Prompt  
```
{map_prompt}
```

## Questions
1. What are the top 3 quality issues you see in the metrics?
2. What specific changes to the REDUCE prompt would improve evidence attribution?
3. What specific changes to the MAP prompt would reduce duplicates and improve field completeness?
4. Are there any parameter changes you'd recommend (batch size, max findings, etc.)?
5. Overall quality rating (1-10) and brief justification.

Please provide concrete, copy-pasteable prompt improvements."""


def strategy_feedback(run_dir: str, api_key: str,
                      api_url: str = "https://api.groq.com/openai/v1/chat/completions",
                      model: str = "llama-3.3-70b-versatile",
                      n_samples: int = 5) -> Dict[str, Any]:
    """Send quality metrics + anonymized samples to Groq for strategy feedback.
    
    Returns the model's analysis and recommendations.
    NO raw project data is sent — only metrics and anonymized samples.
    """
    import requests

    # Gather metrics
    report = quality_report(run_dir)
    metrics = {}
    if "claims" in report:
        metrics["claims"] = report["claims"]
    if "findings" in report:
        metrics["findings"] = report["findings"]
    if "map_errors" in report:
        metrics["map_errors"] = report["map_errors"]

    # Get anonymized samples
    findings_path = os.path.join(run_dir, "findings.json")
    samples = []
    if os.path.exists(findings_path):
        samples = anonymize_samples(findings_path, n=n_samples, strategy="diverse")

    # Load current prompts (from batch_report.py constants — no data in them)
    try:
        from batch_quality import _load_prompts
        map_prompt, reduce_prompt = _load_prompts()
    except Exception:
        map_prompt = "(Could not load MAP prompt)"
        reduce_prompt = "(Could not load REDUCE prompt)"

    # Build strategy request
    user_msg = STRATEGY_USER_TEMPLATE.format(
        metrics_json=json.dumps(metrics, ensure_ascii=False, indent=2),
        samples_json=json.dumps(samples, ensure_ascii=False, indent=2),
        n_samples=len(samples),
        reduce_prompt=reduce_prompt,
        map_prompt=map_prompt,
    )

    # Call Groq API
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": STRATEGY_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()

    feedback_text = result["choices"][0]["message"]["content"]

    # Save feedback
    feedback_path = os.path.join(run_dir, "strategy_feedback.md")
    with open(feedback_path, "w", encoding="utf-8") as f:
        f.write(f"# Strategy Feedback\n\n")
        f.write(f"Model: {model}\n")
        f.write(f"Generated: {_now()}\n\n")
        f.write(feedback_text)

    return {
        "feedback": feedback_text,
        "feedback_path": feedback_path,
        "model": model,
        "tokens_used": result.get("usage", {}),
        "data_sent": {
            "metrics_chars": len(json.dumps(metrics)),
            "samples_count": len(samples),
            "contains_raw_data": False,
            "anonymized": True,
        },
    }


def _load_prompts() -> Tuple[str, str]:
    """Load MAP and REDUCE prompts from batch_report.py (no data, safe to share)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    br_path = os.path.join(script_dir, "batch_report.py")
    if not os.path.exists(br_path):
        return ("(not found)", "(not found)")

    with open(br_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract MAP_SYSTEM and MAP_USER_TEMPLATE
    map_sys = _extract_between(content, 'MAP_SYSTEM = """', '"""')
    map_user = _extract_between(content, 'MAP_USER_TEMPLATE = """', '"""')
    reduce_sys = _extract_between(content, 'REDUCE_SYSTEM = """', '"""')
    reduce_user = _extract_between(content, 'REDUCE_USER_TEMPLATE = """', '"""')

    map_prompt = f"SYSTEM:\n{map_sys}\n\nUSER:\n{map_user}"
    reduce_prompt = f"SYSTEM:\n{reduce_sys}\n\nUSER:\n{reduce_user}"
    return (map_prompt, reduce_prompt)


def _extract_between(text: str, start: str, end: str) -> str:
    """Extract text between two markers."""
    idx = text.find(start)
    if idx == -1:
        return "(not found)"
    idx += len(start)
    end_idx = text.find(end, idx)
    if end_idx == -1:
        return text[idx:idx + 500]
    return text[idx:end_idx]


def _now() -> str:
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Batch Quality Analyzer, Anonymizer & Strategy Feedback",
    )
    sub = ap.add_subparsers(dest="command")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Quality metrics for a batch run")
    p_analyze.add_argument("run_dir", help="Path to run directory")
    p_analyze.add_argument("--format", choices=["json", "text"], default="text")

    # anonymize
    p_anon = sub.add_parser("anonymize", help="Anonymized sample findings")
    p_anon.add_argument("run_dir", help="Path to run directory")
    p_anon.add_argument("--samples", type=int, default=5)
    p_anon.add_argument("--strategy", choices=["diverse", "random", "worst"], default="diverse")

    # feedback
    p_fb = sub.add_parser("feedback", help="Get strategy feedback from Groq")
    p_fb.add_argument("run_dir", help="Path to run directory")
    p_fb.add_argument("--groq-key", default=os.getenv("GROQ_API_KEY", ""),
                      help="Groq API key (or set GROQ_API_KEY env)")
    p_fb.add_argument("--model", default="llama-3.3-70b-versatile")
    p_fb.add_argument("--samples", type=int, default=5)

    args = ap.parse_args()

    if args.command == "analyze":
        report = quality_report(args.run_dir)
        if args.format == "json":
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            _print_text_report(report)

    elif args.command == "anonymize":
        findings_path = os.path.join(args.run_dir, "findings.json")
        samples = anonymize_samples(findings_path, n=args.samples, strategy=args.strategy)
        print(json.dumps(samples, ensure_ascii=False, indent=2))

    elif args.command == "feedback":
        if not args.groq_key:
            print("Error: --groq-key or GROQ_API_KEY required", file=sys.stderr)
            sys.exit(1)
        result = strategy_feedback(args.run_dir, args.groq_key,
                                   model=args.model, n_samples=args.samples)
        print(f"\n{'='*60}")
        print(f"Strategy Feedback (saved to {result['feedback_path']})")
        print(f"Tokens: {result['tokens_used']}")
        print(f"Data sent: {result['data_sent']}")
        print(f"{'='*60}\n")
        print(result["feedback"])

    else:
        ap.print_help()


def _print_text_report(report: Dict[str, Any]):
    """Pretty-print quality report."""
    print(f"\n{'='*60}")
    print(f"QUALITY REPORT: {report.get('run_dir', '?')}")
    print(f"{'='*60}")

    if "claims" in report:
        c = report["claims"]
        print(f"\n--- MAP Claims ---")
        print(f"Total: {c.get('total_claims', 0)}")
        fc = c.get("field_completeness", {})
        print(f"Fields: finding={fc.get('has_finding_pct',0)}% | "
              f"recommendation={fc.get('has_recommendation_pct',0)}% | "
              f"evidence={fc.get('has_evidence_pct',0)}% | "
              f"category={fc.get('has_category_pct',0)}%")
        conf = c.get("confidence", {})
        print(f"Confidence: avg={conf.get('avg',0)} | "
              f"high(≥0.8)={conf.get('high_gte_0.8',0)} | "
              f"low(<0.5)={conf.get('low_lt_0.5',0)}")
        div = c.get("diversity", {})
        print(f"Diversity: {div.get('unique_documents',0)} unique docs | "
              f"{div.get('duplicate_findings',0)} duplicates ({div.get('duplicate_pct',0)}%)")
        print(f"Top categories: {list(c.get('categories', {}).items())[:5]}")

    if "findings" in report:
        f = report["findings"]
        print(f"\n--- REDUCE Findings ---")
        print(f"Total: {f.get('total_findings', 0)}")
        fc = f.get("field_completeness", {})
        print(f"Fields: title={fc.get('has_title_pct',0)}% | "
              f"statement={fc.get('has_statement_pct',0)}% | "
              f"recommendation={fc.get('has_recommendation_pct',0)}% | "
              f"evidence={fc.get('has_evidence_pct',0)}%")
        eq = f.get("evidence_quality", {})
        print(f"Evidence: avg {eq.get('avg_evidence_per_finding',0)}/finding | "
              f"{eq.get('no_evidence_pct',0)}% without | "
              f"{eq.get('unique_evidence_documents',0)} unique docs")
        if eq.get("top_cited_documents"):
            print(f"Top cited: {list(eq['top_cited_documents'].items())[:3]}")
        dup = f.get("duplicates", {})
        print(f"Duplicates: titles={dup.get('duplicate_titles',0)} ({dup.get('duplicate_titles_pct',0)}%) | "
              f"statements={dup.get('duplicate_statements',0)} ({dup.get('duplicate_statements_pct',0)}%)")
        print(f"Categories: {list(f.get('categories', {}).items())[:5]}")
        print(f"Impacts: {dict(f.get('impacts', {}))}")

    if "map_errors" in report:
        me = report["map_errors"]
        print(f"\n--- MAP Errors ---")
        print(f"Total: {me.get('total', 0)}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
