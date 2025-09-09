# test_vectorstore.py

from utils.vectorstore import build_vectorstore, load_documents_from_list

def test_cybersecurity_risk_retrieval():
    sample_texts = [
        "Risk 1: We may lose access to key suppliers in the event of trade disruption.",
        "Risk 2: Cybersecurity incidents could harm our operations or reputation.",
        "Risk 3: Regulatory changes could impact our business model."
    ]

    docs = load_documents_from_list(sample_texts)
    vs = build_vectorstore(docs)
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    query = "What are some cybersecurity-related risks?"
    results = retriever.get_relevant_documents(query)

    assert any("cybersecurity" in doc.page_content.lower() for doc in results), "Cybersecurity risk not found"

