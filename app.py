# app.py — Flask app for Risk Auditor Agent (context-only GPT + concise sources)

from flask import Flask, render_template, request, session
from chains.chat_chain import get_chain
from utils.query_filter import extract_page_number_from_query
from utils.score_responses import score_response_gpt
from utils.snippet_utils import summarize_sources_for_display
import re
import openai
import os
from dotenv import load_dotenv
import logging

# ──────────────────────────────
# Load environment variables
# ──────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ──────────────────────────────
# Logging configuration
# ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ──────────────────────────────
# App + chain setup
# ──────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")
chain = get_chain()

# ──────────────────────────────
# Topic continuity check (via GPT)
# ──────────────────────────────
def is_topic_related(new_question: str, previous_question: str, threshold: float = 0.75) -> bool:
    prompt = f"""
Determine how topically related these two questions are on a scale from 0 to 1 (0 = unrelated, 1 = very related):

Q1: "{previous_question}"
Q2: "{new_question}"

Just return a single number between 0 and 1.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        score_str = response.choices[0].message["content"].strip()
        score = float(score_str)
        logger.info(f"Topic similarity score: {score:.2f}")
        return score >= threshold
    except Exception as e:
        logger.warning(f"Failed to evaluate topic similarity: {e}")
        return True  # Fail open

# ──────────────────────────────
# Optional truncation helper
# ──────────────────────────────
def truncate_at_sentence(text, limit=300):
    if len(text) <= limit:
        return text
    cutoff = text[:limit]
    sentences = re.findall(r'(.+?[.!?])(?=\s|$)', cutoff)
    return sentences[-1] if sentences else cutoff + "..."

# ──────────────────────────────
# Routes
# ──────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form["question"]
    reset = request.form.get("reset") == "true"

    chat_history = [] if reset or "chat_history" not in session else session["chat_history"]
    previous_question = session.get("previous_question", "")
    suggest_reset = False

    if previous_question and not reset:
        suggest_reset = not is_topic_related(question, previous_question)
        logger.info(f"Suggest reset: {suggest_reset}")

   
    # Run LLM chain
    # Invoke chain normally; context will be injected via retrieval
    result = chain.invoke({"question": question, "chat_history": chat_history})


    answer = result["answer"]

    # Source summarisation
    source_summaries = summarize_sources_for_display(result["source_documents"], max_sentences=3)
    logger.info("Source Pages Passed: %s", [doc.metadata.get("page") for doc in result["source_documents"]])

    # Evaluate (optional)
    scores = score_response_gpt(question, answer, source_summaries)

    # Update session
    session["chat_history"] = chat_history + [(question, answer)]
    session["previous_question"] = question

    return render_template(
        "partials/answer_block.html",
        answer=answer,
        sources=source_summaries,
        scores=scores,
        suggest_reset=suggest_reset,
        question=question
    )

# ──────────────────────────────
# Run the app
# ──────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
# To run the app, use:
# python app.py
# Then visit http://localhost:5001 in your browser.

