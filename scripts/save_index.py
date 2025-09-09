# scripts/save_index.py
from utils.vectorstore import load_documents_from_pdf_dir, split_documents, save_vectorstore

docs = load_documents_from_pdf_dir()
chunks = split_documents(docs)
chunks = [doc for doc in chunks if 'wir2024' in doc.metadata.get("source", "")]
save_vectorstore(chunks)

