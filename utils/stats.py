# utils/stats.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tiktoken import get_encoding

def count_tokens(text: str, model_name="gpt-3.5-turbo") -> int:
    enc = get_encoding("cl100k_base")
    return len(enc.encode(text))

def log_chunk_stats(chunks):
    lengths = [count_tokens(c.page_content) for c in chunks]
    print(f"\n--- Chunk Stats ---")
    print(f"Total chunks: {len(chunks)}")
    print(f"Avg tokens per chunk: {sum(lengths)//len(lengths)}")
    print(f"Max: {max(lengths)} | Min: {min(lengths)}")
    over_limit = [i for i in lengths if i > 3000]
    print(f"Chunks over 3000 tokens: {len(over_limit)}")

def log_document_token_stats(docs, max_tokens=10000):
    print(f"\n--- Document Stats ---")
    offending_docs = []
    for i, doc in enumerate(docs):
        tokens = count_tokens(doc.page_content)
        print(f"Doc {i+1:03}: {tokens} tokens | Source: {doc.metadata.get('source')}")
        if tokens > max_tokens:
            offending_docs.append((i, tokens, doc.metadata.get("source")))
    if offending_docs:
        print(f"\n Documents over {max_tokens} tokens:")
        for i, tokens, src in offending_docs:
            print(f"- Doc {i+1:03}: {tokens} tokens | Source: {src}")
    else:
        print("No oversized documents found.")


