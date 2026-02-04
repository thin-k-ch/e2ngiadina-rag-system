import pandas as pd
from bs4 import BeautifulSoup
from docx import Document
from pptx import Presentation
import extract_msg


def read_text_file(path: str) -> str:
    """
    Robust text reader with encoding fallbacks.
    """
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception:
            pass
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")


def read_text_bytes(b: bytes) -> str:
    """
    Decode bytes with robust fallbacks.
    """
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return b.decode(enc)
        except Exception:
            pass
    return b.decode("utf-8", errors="ignore")


def read_docx(path: str) -> str:
    """
    Extract text from DOCX paragraphs.
    """
    doc = Document(path)
    parts = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(parts)


def read_pptx(path: str) -> str:
    """
    Extract text from PPTX slides/shapes.
    """
    prs = Presentation(path)
    out = []
    for si, slide in enumerate(prs.slides, 1):
        out.append(f"--- Slide {si} ---")
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                out.append(shape.text)
    return "\n".join(out)


def read_xlsx(path: str) -> str:
    """
    Extract all sheets to CSV-like text.
    """
    xl = pd.ExcelFile(path, engine="openpyxl")
    out = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        out.append(f"--- Sheet: {sheet} ---")
        out.append(df.to_csv(index=False))
    return "\n".join(out)


def read_html(path: str) -> str:
    """
    Extract visible text from HTML.
    """
    raw = read_text_file(path)
    soup = BeautifulSoup(raw, "lxml")
    return soup.get_text("\n")


def read_msg(path: str) -> str:
    """
    Extract subject/from/to/date/body from .msg (Outlook).
    """
    m = extract_msg.Message(path)
    m.process()

    fields = []
    if getattr(m, "subject", None):
        fields.append(f"Subject: {m.subject}")
    if getattr(m, "sender", None):
        fields.append(f"From: {m.sender}")
    if getattr(m, "to", None):
        fields.append(f"To: {m.to}")
    if getattr(m, "date", None):
        fields.append(f"Date: {m.date}")

    body = getattr(m, "body", "") or ""
    return "\n".join(fields + ["", body]).strip()
