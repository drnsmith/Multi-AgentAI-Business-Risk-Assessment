# utils/snippet_utils.py
import re
from typing import List, Dict
from langchain_core.documents import Document

def summarize_sources_for_display(docs: List[Document], max_sentences=3) -> List[Dict]:
    summaries = []
    for doc in docs:
        content = doc.page_content
        sentences = re.findall(r'([A-Z][^\.!?]*[\.!?])', content)# Extract sentences using regex    
        
        short_summary = " ".join(sentences[:max_sentences]).strip()
        summaries.append({
            "file": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "summary": short_summary
        })
    return summaries

def is_topic_related(current_question: str, previous_question: str, threshold=0.5) -> bool:
    """Check if the current question is related to the previous one based on a simple heuristic. """
    # This is a placeholder for a more sophisticated topic modeling or similarity check.
    # For now, we use a simple keyword overlap heuristic.
    current_keywords = set(re.findall(r'\w+', current_question.lower()))
    previous_keywords = set(re.findall(r'\w+', previous_question.lower()))
    
    if not previous_keywords:
        return True  # No previous question to compare against
    
    overlap = current_keywords.intersection(previous_keywords)
    score = len(overlap) / len(previous_keywords)
    
    return score >= threshold

