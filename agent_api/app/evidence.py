def group_by_file(hits: list, key="original_path"):
    groups = {}
    for h in hits:
        md = h.get("metadata") or {}
        p = md.get(key) or md.get("file_path") or "unknown"
        groups.setdefault(p, []).append(h)
    return groups

def build_evidence_pack(hits: list, max_sources: int = 6, max_chars_per_source: int = 1600):
    # return formatted context + citation map
    lines = []
    sources = []
    groups = group_by_file(hits)
    # prefer groups with best first hit (already reranked)
    ordered_paths = []
    seen=set()
    for h in hits:
        md=h.get("metadata") or {}
        p=md.get("original_path") or md.get("file_path") or "unknown"
        if p not in seen:
            seen.add(p); ordered_paths.append(p)

    for p in ordered_paths[:max_sources]:
        chunks = groups[p]
        # take up to 3 chunks per file
        taken = []
        total = 0
        for h in chunks[:3]:
            txt = (h.get("text") or "").strip()
            if not txt:
                continue
            if total + len(txt) > max_chars_per_source:
                txt = txt[:max(0, max_chars_per_source-total)]
            if txt:
                taken.append(txt)
                total += len(txt)
            if total >= max_chars_per_source:
                break
        if not taken:
            continue
        idx = len(sources) + 1
        sources.append({"n": idx, "path": p})
        lines.append(f"[{idx}] {p}\n" + "\n---\n".join(taken))
    return "\n\n".join(lines), sources
