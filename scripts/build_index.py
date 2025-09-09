# scripts/build_index.py
# scripts/build_index.py
import argparse, os, pathlib
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def load_docs(src_paths):
    docs=[]
    for p in src_paths:
        if p.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(p).load())
        else:
            docs.extend(TextLoader(p, encoding="utf-8").load())
    return docs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="One or more files (PDF/txt)")
    ap.add_argument("--out", default="vectorstore_index", help="Output dir")
    ap.add_argument("--chunk", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--embedding-model", default="text-embedding-3-small")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    docs = load_docs(args.src)
    splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk, chunk_overlap=args.overlap)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=args.embedding_model)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(args.out)
    print(f"✅ Built index with {len(chunks)} chunks → {args.out}")

if __name__ == "__main__":
    main()
   
