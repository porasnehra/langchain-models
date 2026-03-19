# Different Types of Retrievers

This folder demonstrates various LangChain retrievers using Gemini embeddings and sample documents.

**Sample Docs:**
```
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal", 
    "Paris is the capital of France",
    "London is the capital of England",
    "Washington DC is the capital of USA"
]
```

## Setup
1. `pip install -r ../requirements.txt`
2. Set GOOGLE_API_KEY in `.env`
3. Run each: `python [filename].py`

## Retrievers:
- **wikipedia_retriever.py**: Fetches from Wikipedia.
- **vectorstore_retriever.py**: Basic similarity search.
- **mmr_retriever.py**: Maximal Marginal Relevance (diversity).
- **multi_query_retriever.py**: Generates multiple queries for better coverage.
- **contextual_compression_retriever.py**: Compresses docs post-retrieval.

Chroma persists to `./chroma_db`. Run examples to see outputs!
