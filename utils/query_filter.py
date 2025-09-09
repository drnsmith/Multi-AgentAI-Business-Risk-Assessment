# utils/query_filter.py

import re
from typing import List
from langchain_core.documents import Document

def extract_page_number_from_query(query: str):
    match = re.search(r"page\s+(\d+)", query.lower())
    return int(match.group(1)) if match else None

def extract_filename_clue(query: str) -> str | None:
    match = re.search(r"\b(?:WIR)?(?:20)?(1[0-9]|20[0-9]{2}|21[0-9]{2})\b", query)
    if match:
        return f"wir{match.group(1)}"
    return None

def filter_chunks_by_filename(chunks: List[Document], query: str) -> List[Document]:
    clue = extract_filename_clue(query)
    if not clue:
        return chunks  # no clue in query, return all
    return [c for c in chunks if clue in c.metadata.get("source", "")]


def filter_chunks_by_page(chunks, page_number):
    """Filter retrieved chunks to only include those from the requested page number."""
    return [
        chunk for chunk in chunks
        if str(chunk.metadata.get("page", "")).strip() == str(page_number)
    ]

