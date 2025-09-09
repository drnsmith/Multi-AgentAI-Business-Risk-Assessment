# agents/synthesiser.py
from __future__ import annotations
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=60, max_retries=2)

_prompt = ChatPromptTemplate.from_template(
    """You are a senior analyst. Write a concise executive summary from the analyst's findings.
If document citations are provided, reflect them briefly (e.g., [p.12 | wir2024.pdf]).

Analyst findings:
{analyst_answer}

Optional citations:
{citations}

Return:
- 3–5 bullet risks (short, sharp)
- 2–3 mitigations
- A one-sentence bottom line"""
)

_parser = StrOutputParser()

def _collect_citations(docs: List[Document]) -> str:
    bits = []
    for d in docs or []:
        src = d.metadata.get("source") or d.metadata.get("file_name") or "document"
        page = d.metadata.get("page")
        bits.append(f"[p.{page} | {src}]" if page is not None else f"[{src}]")
    # de-dupe, keep short
    return ", ".join(sorted(set(bits)))[:800]

def synthesiser_agent(analyst_answer: str, docs: List[Document]) -> str:
    citations = _collect_citations(docs)
    chain = _prompt | _llm | _parser
    return chain.invoke({"analyst_answer": analyst_answer, "citations": citations})

