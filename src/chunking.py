from typing import Dict, List

from src.utils import clean_text


def is_heading(paragraph: str) -> bool:
    p = paragraph.strip()
    if not p:
        return False
    if len(p) > 90:
        return False
    if p.endswith(":"):
        return True
    if p.lower().startswith(("section ", "chapter ", "step ", "procedure ", "purpose:", "scope:")):
        return True
    if p.isupper():
        return True
    return False


def semantic_chunk_text(
    text: str,
    max_chars: int = 900,
    overlap: int = 120
) -> List[Dict]:
    text = clean_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[Dict] = []

    current_heading = "General"
    current_paragraphs: List[str] = []
    current_len = 0

    def flush():
        nonlocal current_paragraphs, current_len
        if not current_paragraphs:
            return
        chunk_text = "\n\n".join(current_paragraphs).strip()
        chunks.append({
            "heading": current_heading,
            "chunk_text": chunk_text
        })

        if overlap > 0 and chunk_text:
            tail = chunk_text[-overlap:]
            current_paragraphs = [tail]
            current_len = len(tail)
        else:
            current_paragraphs = []
            current_len = 0

    for paragraph in paragraphs:
        if is_heading(paragraph):
            flush()
            current_heading = paragraph.rstrip(":")
            current_paragraphs = [paragraph]
            current_len = len(paragraph)
            continue

        para_len = len(paragraph)
        if current_len + para_len + 2 > max_chars and current_paragraphs:
            flush()

        current_paragraphs.append(paragraph)
        current_len += para_len + 2

    flush()
    return chunks