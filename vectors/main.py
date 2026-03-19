import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# embedding = GoogleGenerativeAIEmbeddings(
#     model="models/text-embedding-004", task_type="retrieval_document"
# )
embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", task_type="retrieval_query"
)


documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France",
]

# Initialize Vectorstore
vectorstore = Chroma.from_texts(
    texts=documents, embedding=embedding, persist_directory="./chroma_db"
)

# Query
query = "What is the capital of India?"
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# Use invoke instead of get_relevant_documents
docs = retriever.invoke(query)

for doc in docs:
    print(f"Result: {doc.page_content}")
