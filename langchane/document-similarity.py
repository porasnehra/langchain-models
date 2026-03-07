import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Updated import
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="retrieval_query"
)
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about bumrah'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(f"Query: {query}")
print(f"Retrieved: {documents[index]}")
print(f"Similarity score: {score:.4f}")