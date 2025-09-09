# utils/loaders.py

import os
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()  # Load environment variables from .env file

def load_and_split_documents(data_dir="data"):
    """Load PDFs from the specified directory and split them into text chunks."""
    documents = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            loader = UnstructuredPDFLoader(os.path.join(data_dir, filename))
            docs = loader.load()
            documents.extend(docs)

    # Split into cleaner overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )
    split_docs = splitter.split_documents(documents)
    return split_docs

def load_documents(data_dir="data"):
    """Alias for load_and_split_documents, for compatibility."""
    return load_and_split_documents(data_dir)
def load_and_split_documents_from_path(path):
    """Load and split documents from a specific path."""
    loader = UnstructuredPDFLoader(path)
    docs = loader.load()

    # Split into cleaner overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )
    split_docs = splitter.split_documents(docs)
    return split_docs   
