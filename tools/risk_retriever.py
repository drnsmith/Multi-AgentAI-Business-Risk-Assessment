# tools/risk_retriever.py
from __future__ import annotations

from typing import List

# Keep BaseTool import as-is to avoid refactors; if it warns, you can switch to langchain_core.tools.BaseTool later.
from langchain.tools import BaseTool
from langchain_core.documents import Document

from utils.vectorstore import load_vectorstore

try:
    from config import TOP_K  # optional, user-configurable
except Exception:
    TOP_K = 10  # sensible default


class RiskRetrieverTool(BaseTool):
    name: str = "RiskRetriever"
    description: str = "Retrieves relevant investment risk information from indexed PDFs."

    def _run(self, query: str) -> List[Document]:
        """Synchronous retrieval entrypoint (BaseTool)."""
        vectorstore = load_vectorstore()
        if vectorstore is None:
            raise RuntimeError(
                "Vectorstore not found. Rebuild it with:\n"
                "  python scripts/build_index.py --src data/wir2024.pdf data/wir2023.pdf --out vectorstore_index"
            )

        # Similarity search is fine for MVP; you can switch to "mmr" for diversity if needed.
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": int(TOP_K)},
        )

        docs: List[Document] = retriever.invoke(query)

        # Debug preview (first 220 chars) — comment out if noisy
        try:
            preview = "\n".join(
                f"- {d.metadata.get('source','doc')} p.{d.metadata.get('page','?')}: {d.page_content[:220].strip()}…"
                for d in (docs[:min(len(docs), 3)])
            )
            print("\n[RetrieverAgent] Retrieved preview:\n" + preview + "\n")
        except Exception:
            pass

        return docs

    def _arun(self, query: str):
        raise NotImplementedError("Async not implemented.")


