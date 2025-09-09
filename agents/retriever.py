# agents/retriever.py
from tools.risk_retriever import RiskRetrieverTool

retriever_tool = RiskRetrieverTool()

def retrieve(query):
    docs = retriever_tool._run(query)  # direct call since we're using BaseTool

    # 🧾 DEBUG: Print out retrieved document content and metadata
    print("\n[RetrieverAgent] Retrieved Documents Preview:")
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")
        print(f"\n--- Document {i + 1} ---")
        print(f"Source: {source} | Page: {page}")
        print(doc.page_content[:500])  # show first 500 characters

    return docs

def retriever_agent(query: str):
    print(f"\n[RetrieverAgent] Processing query: {query}")
    docs = retrieve(query)
    print(f"\n[RetrieverAgent] Done processing. Retrieved {len(docs)} documents.\n")
    return docs

