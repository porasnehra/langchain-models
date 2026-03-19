import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

print("=== MultiQuery Retriever ===")
print("LLM rewrites query into multiple variants for better retrieval.")

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

vectorstore = Chroma.from_texts(
    texts=documents, embedding=embedding, persist_directory="./retrievers/chroma_db"
)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

query = "What is the capital of India?"
docs = retriever.invoke(query)

print(f"\nQuery: {query}")
print("Generated queries (printed internally by retriever)")
print(f"Retrieved {len(docs)} docs:")
for i, doc in enumerate(docs, 1):
    print(f"{i}. {doc.page_content}")

print("\nMultiQuery improves recall with query variants.")
