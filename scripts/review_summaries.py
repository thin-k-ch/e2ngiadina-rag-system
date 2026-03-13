#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Review script for summarization results — prepares statistics and samples
for the 18:00 review meeting.
"""
import json
import sys
from collections import Counter
from pathlib import Path

SUMMARIES = "/media/felix/RAG/AGENTIC/runs/summaries/summaries.jsonl"

FILLER_PHRASES = [
    "nicht im dokument erwähnt", "nicht im dokument genannt",
    "keine explizit genannt", "keine im dokument definiert",
    "keine risiken", "keine pendenzen", "nicht erwähnt",
    "keine offenen punkte", "keine nächsten schritte",
]


def load_results(path: str):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def has_filler(text: str) -> list:
    lower = text.lower()
    return [f for f in FILLER_PHRASES if f in lower]


def print_stats(results):
    ok = [r for r in results if r.get("status") == "ok"]
    skip = [r for r in results if r.get("status") in ("no_content", "too_short")]
    err = [r for r in results if r.get("status") not in ("ok", "no_content", "too_short")]

    # By extension
    ext_counts = Counter(r.get("extension", "?") for r in ok)
    ext_erk = Counter()
    for r in ok:
        ext_erk[r.get("extension", "?")] += len(r.get("erkenntnisse", []))

    # By doc_type
    type_counts = Counter(r.get("doc_type", "?") for r in ok)

    # Filler analysis
    filler_docs = [(r["filename"], has_filler(r.get("onepager", ""))) for r in ok if has_filler(r.get("onepager", ""))]

    # Timing
    times = [r["elapsed_s"] for r in ok if r.get("elapsed_s")]

    # Erkenntnisse
    all_erk = sum(len(r.get("erkenntnisse", [])) for r in ok)

    print("=" * 70)
    print(f"  SUMMARIZATION REVIEW — gpt-oss-120b")
    print("=" * 70)
    print(f"\n  Total processed:     {len(results):>6}")
    print(f"    OK:                {len(ok):>6}")
    print(f"    Skipped:           {len(skip):>6}")
    print(f"    Errors:            {len(err):>6}")
    print(f"    Erkenntnisse:      {all_erk:>6}  (Ø {all_erk / max(len(ok), 1):.1f}/doc)")

    print(f"\n  By Extension:")
    for ext, cnt in ext_counts.most_common():
        erk = ext_erk.get(ext, 0)
        print(f"    .{ext:5} {cnt:>5} docs  {erk:>5} Erk  (Ø {erk / max(cnt, 1):.1f}/doc)")

    print(f"\n  By Document Type:")
    for dt, cnt in type_counts.most_common():
        print(f"    {dt:15} {cnt:>5} docs")

    print(f"\n  Timing:")
    if times:
        print(f"    Ø {sum(times) / len(times):.1f}s/doc  |  Min {min(times):.0f}s  |  Max {max(times):.0f}s  |  Rate {3600 / (sum(times) / len(times)):.0f}/h")

    print(f"\n  Filler Phrases: {len(filler_docs)}/{len(ok)} docs ({len(filler_docs) / max(len(ok), 1) * 100:.1f}%)")
    if filler_docs:
        for fn, phrases in filler_docs[:5]:
            print(f"    ⚠ {fn[:60]}  → {phrases}")

    print("=" * 70)


def print_samples(results, n=5):
    ok = [r for r in results if r.get("status") == "ok"]

    # Best: most Erkenntnisse
    by_erk = sorted(ok, key=lambda r: len(r.get("erkenntnisse", [])), reverse=True)

    print(f"\n{'─' * 70}")
    print(f"TOP {n} DOCUMENTS BY ERKENNTNISSE")
    print(f"{'─' * 70}")
    for r in by_erk[:n]:
        erk = r.get("erkenntnisse", [])
        print(f"\n{'=' * 60}")
        print(f"📄 {r['filename']} ({r.get('extension', '?')}, {r.get('doc_type', '?')})")
        print(f"   {len(erk)} Erkenntnisse, {r.get('elapsed_s', 0):.0f}s, {len(r.get('onepager', ''))} chars")
        print(f"{'=' * 60}")
        print(r.get("onepager", "")[:3000])
        if len(r.get("onepager", "")) > 3000:
            print("... (gekürzt)")
        print()

    # Sample without Erkenntnisse
    no_erk = [r for r in ok if not r.get("erkenntnisse")]
    print(f"\n{'─' * 70}")
    print(f"SAMPLE WITHOUT ERKENNTNISSE ({len(no_erk)} total)")
    print(f"{'─' * 70}")
    for r in no_erk[:3]:
        print(f"\n  📄 {r['filename']} ({r.get('doc_type', '?')})")
        lines = r.get("onepager", "").split("\n")
        # Show just the first section
        for line in lines[:15]:
            print(f"    {line}")
        print()


def print_erkenntnisse_overview(results, top_n=20):
    """Show the most substantive management insights."""
    ok = [r for r in results if r.get("status") == "ok"]
    all_erk = []
    for r in ok:
        for erk in r.get("erkenntnisse", []):
            all_erk.append({"text": erk, "source": r["filename"], "doc_type": r.get("doc_type", "?")})

    # Sort by length (longer = more substantive typically)
    all_erk.sort(key=lambda e: len(e["text"]), reverse=True)

    print(f"\n{'─' * 70}")
    print(f"TOP {top_n} ERKENNTNISSE (by substantiveness)")
    print(f"{'─' * 70}")
    for i, e in enumerate(all_erk[:top_n], 1):
        print(f"\n{i}. [{e['doc_type']}] {e['source'][:50]}")
        print(f"   {e['text'][:300]}")
        if len(e["text"]) > 300:
            print(f"   ...")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else SUMMARIES
    results = load_results(path)

    if not results:
        print(f"No results found in {path}")
        sys.exit(1)

    print_stats(results)
    print_samples(results)
    print_erkenntnisse_overview(results)
