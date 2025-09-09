# cli.py
import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Now use absolute imports
from chains.chat_chain import get_chain
from utils.query_filter import extract_page_number_from_query
from config import FILTER_BY_PAGE

def run_cli():
    query = input("Ask a question: ")

    # Optionally extract page info
    if FILTER_BY_PAGE:
        page = extract_page_number_from_query(query)
        if page:
            print(f" Page filter detected: filtering for page {page}")

    chain = get_chain()
    response = chain.invoke({
        "question": query,
        "chat_history": []
    })

    print("\n📢 Answer:")
    print(response["answer"])

    if "source_documents" in response:
        print("\n📄 Sources:")
        for doc in response["source_documents"]:
            print(f"- {doc.metadata.get('source')} (page {doc.metadata.get('page')})")

if __name__ == "__main__":
    run_cli()

