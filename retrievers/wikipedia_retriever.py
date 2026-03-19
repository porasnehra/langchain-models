import os
from dotenv import load_dotenv
from langchain_community.retrievers import WikiRetriever
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

print("=== Wikipedia Retriever ===")
print("Fetches documents directly from Wikipedia API.")

# Initialize Wikipedia retriever
retriever = WikiRetriever()

# Example query
query = "Python programming language"
docs = retriever.invoke(query)

print(f"\nQuery: {query}")
print(f"Retrieved {len(docs)} docs:")
for i, doc in enumerate(docs, 1):
    print(f"{i}. {doc.page_content[:200]}...")

print("\nWikipedia retriever retrieves relevant Wikipedia page content.")
