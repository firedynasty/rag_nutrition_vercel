#!/usr/bin/env python3
"""
Query the novel RAG knowledge base.

Usage:
    python query_novel.py "Who is Shoko Sekine?"
    python query_novel.py "What happens in chapter 5?"
    python query_novel.py -i  # Interactive mode
"""

import argparse
import os
import sys

# Add current dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.constants import get_rag_config
from src.retrieval import get_knowledge_base, get_context


def query_rag(query: str, verbose: bool = False) -> str:
    """
    Query the novel knowledge base.

    Args:
        query: The question to ask
        verbose: Whether to print additional info

    Returns:
        Context string with relevant information
    """
    config = get_rag_config()

    if verbose:
        print(f"Loading knowledge base...")

    k_base = get_knowledge_base()

    if verbose:
        print(f"Searching for: {query}")
        print("-" * 50)

    context = get_context(
        k_base=k_base,
        query_text=query,
        n_retrieve=config["retriever"]["n_retrieve"],
        n_chapters=config["retriever"]["n_titles"],
        enrich_first=config["retriever"]["enrich_first"],
        reranker=config["retriever"]["reranker"]
    )

    return context


def interactive_mode():
    """Run in interactive query mode."""
    print("=" * 60)
    print("  All She Was Worth - RAG Query System")
    print("=" * 60)
    print("\nExample queries:")
    print("  - Who is Shoko Sekine?")
    print("  - What happens in chapter 10?")
    print("  - Who is Kyoko Shinjo?")
    print("  - What is the connection between debt and identity?")
    print("\nType 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break

        print()
        result = query_rag(query, verbose=True)
        print(result)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Query the novel RAG knowledge base"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Question to ask about the novel"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    if args.interactive or not args.query:
        interactive_mode()
    else:
        result = query_rag(args.query, verbose=args.verbose)
        print(result)


if __name__ == "__main__":
    main()
