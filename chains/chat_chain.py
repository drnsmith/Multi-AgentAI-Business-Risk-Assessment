
# ──────────────────────────────────────────────────────────────
# chains/chat_chain.py
# Build a ConversationalRetrievalChain with a boosted retriever
# ──────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any, List

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.schema import BaseRetriever, Document
from pydantic import Field                                  # ← needed for Pydantic fields

from utils.vectorstore import load_vectorstore
from config import USE_MMR, SCORE_THRESHOLD, TOP_K

# ─── Custom wrapper that boosts exact keyword matches ─────────
class SimpleBoostedRetriever(BaseRetriever):
    """
    Wrap an existing LangChain retriever and push up any chunk that
    contains an exact keyword from the query.
    """

    inner: Any = Field(...)           # required – underlying Retriever / FAISS wrapper
    k: int = Field(default=TOP_K)

    # sync
    def get_relevant_documents(self, query: str) -> List[Document]:
        # pull 2×k candidate docs from the inner retriever
        candidates = self.inner.get_relevant_documents(query)
        keywords = {w.lower() for w in query.split()}

        scored: list[tuple[Document, float]] = []
        for idx, doc in enumerate(candidates):
            text = doc.page_content.lower()
            boost = any(kw in text for kw in keywords)
            score = -1 if boost else idx          # boosted docs float to the top
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1])
        return [doc for doc, _ in scored[: self.k]]

    # async (required by BaseRetriever)
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

# ─── Factory ──────────────────────────────────────────────────
def get_chain() -> ConversationalRetrievalChain:
    # 1. Load the FAISS store you built earlier
    vectorstore = load_vectorstore()

    # 2. Create the standard vectorstore retriever (MMR or similarity)
    base_retriever = vectorstore.as_retriever(
        search_type="mmr" if USE_MMR else "similarity",
        search_kwargs={"k": TOP_K, "score_threshold": SCORE_THRESHOLD},
    )

    # 3. Wrap it with our boosting logic
    boosted_retriever = SimpleBoostedRetriever(inner=base_retriever, k=TOP_K)

    # 4. Add conversational memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # 5. Assemble the chain
    llm = ChatOpenAI(temperature=0)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=boosted_retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",
    )

