# agents/analyst.py
# Modern, deprecation-free analyst agent for the 3-agent MVP
from __future__ import annotations

from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain


def _format_docs_with_citations(docs: List[Document]) -> List[Document]:
    """
    Prefix each document's content with a lightweight citation header so
    the LLM tends to echo sources (page + filename) in its answer.
    """
    formatted: List[Document] = []
    for d in docs or []:
        page = d.metadata.get("page")
        src = d.metadata.get("source") or d.metadata.get("file_name") or "document"
        header = f"[p.{page} | {src}]" if page is not None else f"[{src}]"
        formatted.append(Document(page_content=f"{header} {d.page_content}", metadata=d.metadata))
    return formatted


# Configure the chat model (adjust model if you need GPT-4 proper)
llm = ChatOpenAI(
    model="gpt-4o-mini",   # swap to "gpt-4" if needed
    temperature=0,
    streaming=True,
    timeout=60,
    max_retries=2,
)

# Prompt template: keep it grounded in provided evidence
prompt = ChatPromptTemplate.from_template(
    """You are a risk analyst reading investment documents.
Prioritize the provided context, but if it's incomplete, make a best-effort inference to answer.

Query:
{query}

Context:
{context}

Return a clear, concise answer with brief justifications. Where appropriate, include
lightweight citations like [p.X | filename] taken from the context.
"""
)

# Build a "stuff" chain (replacement for deprecated StuffDocumentsChain + LLMChain)
# Default document_variable_name is "context", which we will respect.
analyst_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)


def analyst_agent(input_documents: List[Document], query: str) -> str:
    """
    Run the analyst over retrieved documents to produce a concise, grounded answer.

    Args:
        input_documents: List of LangChain Document objects (ideally with metadata.page/source).
        query: The analyst question (e.g., "Top three risks for the next 12–18 months?").

    Returns:
        String answer suitable for display/logging.
    """
    # Guard rails: handle empty retrieval gracefully
    if not input_documents:
        return (
            "No supporting context was retrieved, so I can't answer confidently. "
            "Please rebuild the index or broaden the query."
        )

    docs_with_citations = _format_docs_with_citations(input_documents)

    # Pass docs under the key "context" (the chain's expected variable)
    response: str = analyst_chain.invoke({"context": docs_with_citations, "query": query})
    print("\n[AnalystAgent] Done processing.\n")
    return response

