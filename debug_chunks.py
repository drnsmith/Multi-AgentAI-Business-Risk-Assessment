# debug_chunks.py

from utils.vectorstore import load_vectorstore

def debug_chunk_for_phrase(phrase):
    vs = load_vectorstore()
    results = vs.similarity_search(phrase, k=1)
    
    if not results:
        print("No results found.")
        return
    
    doc = results[0]
    print(" Chunk Content:\n")
    print(doc.page_content)
    print("\n Metadata:", doc.metadata)

if __name__ == "__main__":
    phrase = "FDI to the 45 LDCs increased"
    debug_chunk_for_phrase(phrase)
# This script is used to debug and inspect the content of chunks in the vector store.
# It loads the vector store, performs a similarity search for a given phrase,
# and prints the content and metadata of the retrieved chunk.
# You can run this script to check if the chunks contain the expected information.
# Make sure to adjust the phrase variable to test different queries.
# This is useful for verifying that the vector store is populated correctly     
# and that the chunks contain relevant data for your application.
# before running the main Flask application.

