# utils/vectorstore.py

import os
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings

# Load .env once
load_dotenv()

VECTORSTORE_PATH = "vectorstore_index"
EMBEDDING_MODEL = "text-embedding-3-small"

# === DOCUMENT LOADING ===

def load_documents_from_pdf_dir(dir_path="data") -> List[Document]:
    """Load and annotate all PDFs in the directory with source + page metadata."""
    documents = []
    for file in os.listdir(dir_path):
        if file.endswith(".pdf"):
            path = os.path.join(dir_path, file)
            loader = PyMuPDFLoader(path)
            docs = loader.load()
            for i, d in enumerate(docs):
                d.metadata["source"] = file
                d.metadata["page_number"] = i + 1
            documents.extend(docs)
    print(f"Loaded {len(documents)} documents from {dir_path}")
    return documents

def split_documents(docs: List[Document], chunk_size=500, chunk_overlap=100) -> List[Document]:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import re

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # === SMART TAGGING BASED ON COUNTRIES ===
    countries = [
    "mali", "chile", "el salvador", "morocco", "vietnam", "china", "turkiye", 
    "egypt", "india", "mexico", "france", "germany", "united states", 
    "switzerland", "romania", "south africa"
]
    # Add more countries as needed
    for chunk in chunks:
        text = chunk.page_content.lower()
        tags = [country for country in countries if re.search(rf"\b{country}\b", text)]
        chunk.metadata["tags"] = tags
    return chunks


# === VECTORSTORE BUILD / SAVE / LOAD ===

def build_vectorstore(chunks: List[Document], batch_size=100) -> FAISS:
    """Embed text chunks and build a FAISS index."""
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    embedded_texts = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding in batches"):
        batch = texts[i:i + batch_size]
        embedded = embedding_model.embed_documents(batch)
        embedded_texts.extend(embedded)

    faiss_store = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, embedded_texts)),
        embedding=embedding_model,
        metadatas=metadatas
    )
    return faiss_store

def save_vectorstore(chunks: List[Document], path=VECTORSTORE_PATH):
    """Build and save FAISS vectorstore locally."""
    vs = build_vectorstore(chunks)
    vs.save_local(path)
    print(f"Vectorstore saved to {path}")

# def load_vectorstore(path=VECTORSTORE_PATH) -> FAISS:
#     """Load vectorstore from disk."""
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Vectorstore directory not found: {path}")
#     return FAISS.load_local(path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)


def load_vectorstore(path=VECTORSTORE_PATH) -> FAISS:
    """Load FAISS index and return a retriever that prioritises keyword hits."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vectorstore directory not found: {path}")

    faiss_store = FAISS.load_local(path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    # Create a retriever with keyword prioritisation
    # This custom retriever boosts results based on country tags or keyword hits
    # It overrides the default similarity search to apply custom scoring logic
    def boosted_retriever(query: str, k: int = 8):
        """Custom retriever that boosts chunks with country tag matches or keyword hits."""
        results = faiss_store.similarity_search_with_score(query, k=20)
        keywords = [kw.lower() for kw in query.split()]

        boosted = []
        for doc, score in results:
            doc_text = doc.page_content.lower()
            doc_tags = [tag.lower() for tag in doc.metadata.get("tags", [])]

            # Boost logic
            exact_country_match = any(kw in doc_tags for kw in keywords)
            keyword_match = any(kw in doc_text for kw in keywords)

            if exact_country_match:
                boosted_score = score - 8  # Strongest boost
            elif keyword_match:
                boosted_score = score - 3  # Mild boost
            else:
                boosted_score = score

            boosted.append((doc, boosted_score))

        # Sort by boosted score (lower = better)
        boosted.sort(key=lambda x: x[1])

        # Debug output
        for i, (doc, score) in enumerate(boosted[:k]):
            print(f"RESULT {i+1}: score={score:.2f}, tags={doc.metadata.get('tags', [])}")
            print(doc.page_content[:300].replace("\n", " "))
            print("-" * 60)

        return [doc for doc, _ in boosted[:k]]


    # Attach as method
    faiss_store.boosted_retriever = boosted_retriever
    return faiss_store

# === FOR SCRIPT EXECUTION ===

if __name__ == "__main__":
    docs = load_documents_from_pdf_dir()
    chunks = split_documents(docs)

    print(f"Splitting into {len(chunks)} chunks...")
    # Optional: filter specific documents
    chunks = [doc for doc in chunks if 'wir2024' in doc.metadata.get("source", "")]
    save_vectorstore(chunks)
    print("Vectorstore built and persisted.")
    # Uncomment to load and test
