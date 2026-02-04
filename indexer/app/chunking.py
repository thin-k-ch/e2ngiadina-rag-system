def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = (text or "").replace("\x00", "")
    if not text.strip():
        return []
    out = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        out.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return out
