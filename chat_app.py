# chat_app.py
# A simple Flask app to interact with a LangChain-based conversational retrieval system.
# It allows users to ask questions and retrieves answers from a vectorstore built from PDF documents.
# The app uses OpenAI's GPT for evaluation and scoring of responses.
import re
import json
from datetime import datetime
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from utils.vectorstore import build_vectorstore, split_documents, load_documents_from_pdf_dir
from utils.stats import log_chunk_stats, log_document_token_stats
from utils.query_filter import filter_chunks_by_page
from utils.score_responses import score_response_gpt
from config import USE_MMR, SCORE_THRESHOLD, TOP_K
from config import FILENAME_FILTERING_ENABLED
from utils.query_filter import filter_chunks_by_filename
from flask import Flask, render_template, request, jsonify
from chains.chat_chain import get_chain
from utils.score_responses import score_response_gpt

app = Flask(__name__)
chat_chain = get_chain()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "")
    if not question.strip():
        return jsonify({"error": "Question cannot be empty."}), 400

    result = chat_chain.invoke({"question": question, "chat_history": []})
    answer = result.get("answer", "No answer found.")
    sources = result.get("source_documents", [])

    source_info = []
    for doc in sources[:3]:
        source_info.append({
            "file": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "unknown"),
            "snippet": doc.page_content[:300]  # Short preview
        })

    scores = score_response_gpt(question, answer, source_info)

    return render_template("partials/answer_block.html", answer=answer, sources=source_info, scores=scores)
if __name__ == "__main__":
    app.run(debug=True)

    # Load documents
    docs = load_documents_from_pdf_dir("data")
    print(f"Loaded {len(docs)} documents from PDF files.")
    for i, d in enumerate(docs[:3]):
        print("---")
        print(d.page_content[:300])
        print("Source:", d.metadata.get("source"))

    # Log token stats
    log_document_token_stats(docs)

    # Split into smaller chunks
    chunks = split_documents(docs)

    # Optional: filter chunks by filename clue in query
    user_query_preview = input("Type a test query to optimise vectorstore (or press Enter to skip): ")
    filtered_chunks = filter_chunks_by_filename(chunks, user_query_preview) if user_query_preview else chunks
    def filter_chunks_by_filename(chunks, query):
        """Filter chunks based on filename clue in the query."""
        match = re.search(r"\b(?:WIR)?(?:20)?(1[0-9]|20[0-9]{2}|21[0-9]{2})\b", query)
        if match:
            clue = f"wir{match.group(1)}"
            return [c for c in chunks if clue in c.metadata.get("source", "")]
        return chunks  # no clue in query, return all   

    # Log WIR chunk count (optional)
    wir_chunks = [doc for doc in chunks if 'wir2024' in doc.metadata.get("source", "")]
    print(f"Found {len(wir_chunks)} chunks from wir2024.pdf.")
    log_chunk_stats(chunks)

    # Build vectorstore
    # after loading and splitting docs
    query = input("User: ")  # or wherever your query is captured
    user_query_preview = query  # Optional: could use a simplified/normed version

    if FILENAME_FILTERING_ENABLED and user_query_preview:
        filtered_chunks = filter_chunks_by_filename(chunks, user_query_preview)
    else:
        filtered_chunks = chunks

    vs = build_vectorstore(filtered_chunks)


    # Set up memory + LLM
    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    retriever = vs.as_retriever(
        search_type="mmr" if USE_MMR else "similarity",
        search_kwargs={"k": TOP_K, "score_threshold": SCORE_THRESHOLD}
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

    def extract_snippet(text, query, max_length=300):
        query_words = set(query.lower().split())
        sentences = re.split(r'[.?!]\s+', text)
        scored = [
            (sum(word.lower() in query_words for word in sentence.split()), sentence)
            for sentence in sentences
        ]
        best = max(scored, key=lambda x: x[0])[1] if scored else text[:max_length]
        return best.strip()

    log_path = "chat_log.jsonl"

    while True:
        query = input("User: ")
        if query.lower() in {"exit", "quit"}:
            break

        result = qa_chain.invoke({"question": query})
        if "answer" not in result:
            print("No answer found. Please try a different question.")
            continue

        # Filter sources by page if "page X" is in query
        match = re.search(r"\bpage\s+(\d+)", query.lower())
        if match:
            requested_page = int(match.group(1))
            filtered_docs = filter_chunks_by_page(result["source_documents"], requested_page)
            if filtered_docs:
                result["source_documents"] = filtered_docs
                print(f"[Filter] Showing only results from page {requested_page}.")
            else:
                print(f"[Filter] No results for page {requested_page}, showing all original sources.")

        print("\nBot:", result["answer"])

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": query,
            "answer": result["answer"],
            "sources": []
        }

        print("\n--- Source Documents Used ---")
        for i, doc in enumerate(result["source_documents"][:3]):
            source_name = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', 'unknown')
            snippet = extract_snippet(doc.page_content, query)
            print(f"Doc {i+1}: {source_name} — page {page}")
            print("Snippet:", snippet)
            print("---")
            log_entry["sources"].append({"file": source_name, "page": page, "snippet": snippet})

        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Evaluate GPT answer
        scores = score_response_gpt(query, result["answer"], log_entry["sources"])
        print("\n--- GPT Evaluation ---")
        for metric in ["relevance", "completeness", "faithfulness"]:
            print(f"{metric.capitalize()}: {scores[metric]}")

        # Retry if weak score
        if any(s < 0.7 for s in scores.values()):
            print("  Warning: This answer may be incomplete or less relevant. Retrying with new response...")
            result_retry = qa_chain.invoke({"question": query})
            print("\nRetry Bot:", result_retry["answer"])
            scores_retry = score_response_gpt(query, result_retry["answer"], log_entry["sources"])
            print("--- Retry GPT Evaluation ---")
            for metric in ["relevance", "completeness", "faithfulness"]:
                print(f"{metric.capitalize()}: {scores_retry[metric]}")

        print("\n--- Chat History ---")
        for msg in memory.chat_memory.messages:
            role = "User" if msg.type == "human" else "Bot"
            print(f"{role}: {msg.content}")
        print("\n--- End of chat history ---\n")

    # Save chat history
    with open("chat_history.txt", "w") as f:
        for msg in memory.chat_memory.messages:
            role = "User" if msg.type == "human" else "Bot"
            f.write(f"{role}: {msg.content}\n")
    print("\nChat history saved to chat_history.txt")

    # Debug info
    print("\n--- Chain Info ---")
    print(qa_chain)
    print("\n--- Memory Info ---")
    print(memory)
    print("\n--- Vectorstore Info ---")
    print(vs)
    print(f"Vectorstore contains {vs.index.ntotal} chunks.")
    print("You can now ask questions about the loaded documents. Type 'exit' or 'quit' to end the chat.")
    print(f"[Config] USE_MMR={USE_MMR}, SCORE_THRESHOLD={SCORE_THRESHOLD}, TOP_K={TOP_K}")

