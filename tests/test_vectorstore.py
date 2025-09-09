# test_vectorstore.py

from utils.vectorstore import build_vectorstore, load_documents_from_list

def test_vectorstore_retrieves_cybersecurity_risk():
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

    assert len(results) > 0, "No documents were retrieved."
    assert "cybersecurity" in results[0].page_content.lower(), "Cybersecurity risk not found in top result."

