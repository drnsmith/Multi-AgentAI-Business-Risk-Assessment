# utils/score_responses.py

from openai import OpenAI
from typing import List, Dict

client = OpenAI()

def score_response_gpt(question: str, answer: str, sources: List[Dict]) -> Dict[str, float]:
    """Score the response using GPT for relevance, completeness, and faithfulness."""
    context_snippets = "\n\n".join(
        f"Source: {src.get('file', 'unknown')} (page {src.get('page', '?')}):\n{src.get('snippet', '')}"
        for src in sources
    )

    prompt = f"""
You are an expert evaluator assessing the quality of an AI-generated answer to a user query.

Question:
{question}

Answer:
{answer}

Relevant Source Excerpts:
{context_snippets}

Rate the answer on a scale of 0 to 1 for each of the following:

- Relevance: Does the answer directly address the question?
- Completeness: Does it include all major relevant points?
- Faithfulness: Is the answer consistent with the provided sources?

Respond in this JSON format:
{{
  "relevance": ...,
  "completeness": ...,
  "faithfulness": ...
}}
    """

    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a strict evaluator of AI-generated content quality."},
            {"role": "user", "content": prompt}
        ]
    )

    try:
        parsed = eval(response.choices[0].message.content.strip())  # ⚠️ Use a real parser in prod
        return parsed
    except Exception:
        return {"relevance": 0.0, "completeness": 0.0, "faithfulness": 0.0}

