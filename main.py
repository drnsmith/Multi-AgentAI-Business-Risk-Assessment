# main.py

from agents.retriever import retriever_agent
from agents.analyst import analyst_agent
from agents.synthesiser import synthesiser_agent


def run_audit(query: str):
    docs = retriever_agent(query)

    # Debug preview of retrieved evidence
    for d in docs:
        print(d.metadata.get("source"), d.metadata.get("page"), d.page_content[:300], "...\n")

    # Analyst step
    answer = analyst_agent(input_documents=docs, query=query)

    print("\n========================\nFinal Answer:\n")
    print(answer)
    print("\n========================\n")

    # Synthesiser step (executive summary)
    summary = synthesiser_agent(answer, docs)
    print("\n[Synthesiser] Executive Summary\n")
    print(summary)


if __name__ == "__main__":
    query = input("Enter your audit query: ")
    run_audit(query)


