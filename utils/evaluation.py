# utils/evaluation.py

from openai import OpenAI
from openai import RateLimitError
import time

client = OpenAI()

def score_response_gpt(question, answer, sources, retries=2):
    prompt = f"""
You are an expert evaluator. Given a user question, an AI answer, and source documents, rate the response on the following criteria (from 0 to 1):

1. Relevance: Does the answer address the user's question directly?
2. Completeness: Does the answer cover all critical aspects of the question?
3. Faithfulness: Is the answer accurate and consistent with the provided sources?

Question:
{question}

Answer:
{answer}

Sources:
{sources}

Return JSON:
{{"relevance": x, "completeness": y, "faithfulness": z}}
"""

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a meticulous evaluator."},
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.choices[0].message.content.strip()
            return eval(result)
        except RateLimitError:
            print("⚠️ Rate limit hit, retrying...")
            time.sleep(2)
        except Exception as e:
            print(f"❌ GPT Evaluation failed: {e}")
            return {"relevance": 0.0, "completeness": 0.0, "faithfulness": 0.0}
    print("❗ Max retries reached, returning default scores.")
    return {"relevance": 0.0, "completeness": 0.0, "faithfulness": 0.0} 
# This function uses OpenAI's GPT to evaluate the quality of an AI-generated answer
# based on relevance, completeness, and faithfulness to the provided sources.
