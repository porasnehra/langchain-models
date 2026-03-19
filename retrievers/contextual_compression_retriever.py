import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

print("=== Contextual Compression Retriever ===")
print("Retrieves docs then compresses to relevant excerpts using LLM.")

documents = [
    "Delhi is the capital of India and known for historical monuments like Red Fort.",
    "Kolkata is the capital of West Bengal, famous for Durga Puja and Howrah Bridge.",
    "Paris is the capital of France, home to Eiffel Tower and Louvre Museum.",
    "London is the capital of England, with Big Ben and British Museum.",
    "Washington DC is the capital of USA, featuring White House and Capitol.",
]

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", task_type="retrieval_query"
)

vectorstore = Chroma.from_texts(
    texts=documents, embedding=embedding, persist_directory="./retrievers/chroma_db"
)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

compressor_prompt = ChatPromptTemplate.from_template(
    """Given a question and document, extract the relevant parts.
    Question: {{question}}
    Document: {{text}}
    Relevant parts:"""
)

compressor = LLMChainExtractor.from_llm(llm, compressor_prompt)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

query = "What is the capital of India?"
docs = retriever.invoke(query)

print(f"\nQuery: {query}")
print(f"Retrieved & compressed {len(docs)} docs:")
for i, doc in enumerate(docs, 1):
    print(f"{i}. {doc.page_content}")

print("\nCompression reduces noise, keeps relevant info.")
