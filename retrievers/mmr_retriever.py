import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

print("=== MMR Retriever (Maximal Marginal Relevance) ===")
print("Retrieves diverse results using MMR search_type.")

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France",
    "London is the capital of England",
    "Washington DC is the capital of USA",
]

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", task_type="retrieval_query"
)

# Create/load vectorstore
vectorstore = Chroma.from_texts(
    texts=documents, embedding=embedding, persist_directory="./retrievers/chroma_db"
)

# MMR retriever: fetch_k=6 (candidates), k=2 (final), lambda_mult=0.5 (balance relevance/diversity)
retriever = vectorstore.as_retriever(
    search_type="mmr", search_kwargs={"fetch_k": 6, "lambda_mult": 0.5}
)

query = "capital Europe"
docs = retriever.invoke(query)

print(f"\nQuery: {query}")
print(f"Retrieved {len(docs)} diverse docs:")
for i, doc in enumerate(docs, 1):
    print(f"{i}. {doc.page_content}")

print("\nMMR balances relevance & diversity, avoiding similar results.")
