#!/usr/bin/env python3
"""
Ingest novel summary text into LanceDB for RAG queries.

Usage:
    python ingest_novel.py -i ../temp_extract/all_she_was_worth.txt

This script:
1. Reads the extracted text file
2. Parses it into chapters
3. Chunks the text with overlap
4. Creates embeddings and stores in LanceDB
"""

import argparse
import hashlib
import os
import re
import sys
from typing import Any

import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.constants import LANCEDB_URI, get_rag_config


def parse_chapters(text: str) -> list[dict[str, Any]]:
    """
    Parse the text into chapters based on 'Chapter X Summary' markers.

    Args:
        text: Full text content

    Returns:
        List of chapter dictionaries with metadata
    """
    chapters = []

    # Split by chapter markers
    # Pattern matches "Chapter 1 Summary :", "Chapter 2 Summary:", etc.
    chapter_pattern = r'(Chapter\s+(\d+)\s+Summary\s*:?)'

    # Find all chapter markers with their positions
    matches = list(re.finditer(chapter_pattern, text, re.IGNORECASE))

    if not matches:
        # No chapters found, treat entire text as one section
        return [{
            "chapter": "Full Text",
            "chapter_num": 0,
            "section": "complete",
            "content": text.strip()
        }]

    # Extract intro/about section (before first chapter)
    intro_end = matches[0].start()
    intro_text = text[:intro_end].strip()
    if intro_text:
        # Check for specific sections in intro
        if "About the book" in intro_text:
            chapters.append({
                "chapter": "About the Book",
                "chapter_num": 0,
                "section": "introduction",
                "content": intro_text
            })

    # Extract each chapter
    for i, match in enumerate(matches):
        chapter_num = int(match.group(2))
        chapter_start = match.end()

        # End is either next chapter or end of text
        if i + 1 < len(matches):
            chapter_end = matches[i + 1].start()
        else:
            # For last chapter, find where quotes/quiz sections start
            remaining = text[chapter_start:]
            quiz_match = re.search(r'Best Quotes from|Chapter \d+ \| Quiz', remaining)
            if quiz_match:
                chapter_end = chapter_start + quiz_match.start()
            else:
                chapter_end = len(text)

        content = text[chapter_start:chapter_end].strip()

        # Clean up page markers
        content = re.sub(r'={50}\nPAGE \d+\n={50}\n?', '\n', content)
        content = re.sub(r'\n{3,}', '\n\n', content)

        # Detect section type
        section = "summary"
        if "Critical Thinking" in content:
            section = "summary_with_analysis"

        chapters.append({
            "chapter": f"Chapter {chapter_num}",
            "chapter_num": chapter_num,
            "section": section,
            "content": content.strip()
        })

    # Extract quotes section if present
    quotes_match = re.search(r'Best Quotes from.*?(?=Chapter \d+ \| Quiz|$)', text, re.DOTALL)
    if quotes_match:
        quotes_text = quotes_match.group(0).strip()
        quotes_text = re.sub(r'={50}\nPAGE \d+\n={50}\n?', '\n', quotes_text)
        chapters.append({
            "chapter": "Quotes Collection",
            "chapter_num": 99,
            "section": "quotes",
            "content": quotes_text
        })

    return chapters


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 150) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk
        overlap: Characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end within last 200 chars
            search_start = max(end - 200, start)
            last_period = text.rfind('. ', search_start, end)
            if last_period > start:
                end = last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start with overlap
        start = end - overlap if end < len(text) else end

    return chunks


def create_documents(chapters: list[dict[str, Any]], config: dict) -> list[dict[str, Any]]:
    """
    Create document records for LanceDB.

    Args:
        chapters: Parsed chapter data
        config: RAG configuration

    Returns:
        List of document records
    """
    max_chars = config["embeddings"]["n_char_max"]
    overlap = config["embeddings"]["overlap"]

    documents = []

    for chapter_data in chapters:
        chapter = chapter_data["chapter"]
        chapter_num = chapter_data["chapter_num"]
        section = chapter_data["section"]
        content = chapter_data["content"]

        # Chunk the content
        chunks = chunk_text(content, max_chars, overlap)
        n_docs = len(chunks)

        # Create hash for chapter
        hash_title = hashlib.md5(chapter.encode()).hexdigest()[:16]

        for rank, chunk in enumerate(chunks):
            # Create unique hash for this chunk
            hash_doc = hashlib.md5(f"{chapter}_{rank}_{chunk[:50]}".encode()).hexdigest()[:16]

            documents.append({
                "text": chunk,
                "chapter": chapter,
                "chapter_num": chapter_num,
                "section": section,
                "hash_title": hash_title,
                "hash_doc": hash_doc,
                "rank_abs": rank,
                "rank_rel": rank / max(n_docs, 1),
                "n_docs": n_docs,
            })

    return documents


def ingest_to_lancedb(documents: list[dict[str, Any]], config: dict):
    """
    Ingest documents into LanceDB with embeddings.

    Args:
        documents: List of document records
        config: RAG configuration
    """
    model_name = config["embeddings"]["model_name"]
    n_dim = config["embeddings"]["n_dim_vec"]

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Generate embeddings for all texts
    print("Generating embeddings...")
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True)

    # Add vectors to documents
    for doc, vec in zip(documents, embeddings):
        doc["vector"] = vec.tolist()

    # Connect to LanceDB
    table_name = config["knowledge_base"]["table_name"]
    print(f"Connecting to LanceDB at: {LANCEDB_URI}")

    db = lancedb.connect(LANCEDB_URI)

    # Drop existing table if present
    try:
        db.drop_table(table_name)
        print(f"Dropped existing table: {table_name}")
    except Exception:
        pass

    # Create table with documents
    print(f"Creating table '{table_name}' with {len(documents)} chunks...")
    table = db.create_table(table_name, data=documents)

    # Create FTS index for hybrid search
    print("Creating full-text search index...")
    try:
        table.create_fts_index("text", replace=True)
        print("FTS index created successfully")
    except Exception as e:
        print(f"Warning: Could not create FTS index: {e}")
        print("Vector search will still work, but hybrid search won't be available")

    print(f"\nIngestion complete! Table '{table_name}' has {table.count_rows()} rows")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest novel summary text into LanceDB for RAG"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the extracted text file"
    )
    args = parser.parse_args()

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Load config
    config = get_rag_config()

    # Read text file
    print(f"Reading: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"  Read {len(text):,} characters")

    # Parse into chapters
    print("Parsing chapters...")
    chapters = parse_chapters(text)
    print(f"  Found {len(chapters)} sections:")
    for ch in chapters:
        print(f"    - {ch['chapter']} ({len(ch['content']):,} chars)")

    # Create document chunks
    print("Creating document chunks...")
    documents = create_documents(chapters, config)
    print(f"  Created {len(documents)} chunks")

    # Ingest to LanceDB
    ingest_to_lancedb(documents, config)


if __name__ == "__main__":
    main()
