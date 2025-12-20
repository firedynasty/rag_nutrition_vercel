"""
Example: How to use the retrieval system
"""

from src.retrieval import get_knowledge_base, get_context
from src.constants import get_rag_config

# Load config
config = get_rag_config()

# Get the knowledge base table
k_base = get_knowledge_base()

# Your question
query = "What foods help reduce inflammation?"

# Full retrieval with hybrid search + reranking
context = get_context(
    k_base=k_base,
    query_text=query,
    n_retrieve=config["retriever"]["n_retrieve"],
    n_titles=config["retriever"]["n_titles"],
    enrich_first=config["retriever"]["enrich_first"],
    reranker=config["retriever"]["reranker"]
)

print(f"Query: {query}\n")
print("Retrieved Context:")
print("-" * 50)
print(context)
