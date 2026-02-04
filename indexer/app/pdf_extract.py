import fitz

def extract_pdf_text(path: str) -> list[dict]:
    """Return list of {page:int, text:str}."""
    doc = fitz.open(path)
    pages = []
    for i in range(len(doc)):
        text = doc.load_page(i).get_text("text") or ""
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages
