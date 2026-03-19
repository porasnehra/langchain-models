import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

print("=== Vector Store Retriever ===")
print("Uses Chroma vectorstore with similarity search retriever.")

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

# Create vectorstore (persists to ./retrievers/chroma_db)
vectorstore = Chroma.from_texts(
    texts=documents, embedding=embedding, persist_directory="./retrievers/chroma_db"
)

# Basic retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "capital of India"
docs = retriever.invoke(query)

print(f"\nQuery: {query}")
print(f"Retrieved {len(docs)} docs:")
for i, doc in enumerate(docs, 1):
    print(f"{i}. {doc.page_content}")

print("\nVectorStoreRetriever returns top-k similar docs.")
