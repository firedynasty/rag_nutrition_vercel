#!/usr/bin/env python3
"""
Migrate LanceDB data to Pinecone with OpenAI embeddings.

Requirements:
  pip install pinecone openai lancedb pandas

Environment variables:
  OPENAI_API_KEY - Your OpenAI API key
  PINECONE_API_KEY - Your Pinecone API key
"""

import os
import time
import re
from openai import OpenAI
from pinecone import Pinecone
import lancedb
import pandas as pd

# Configuration
LANCEDB_PATH = "rag_nutrition/databases/my_lancedb"
OLD_TABLE_NAME = "table_simple05"
PINECONE_INDEX_NAME = "nutrition-rag"  # Change this to your index name
BATCH_SIZE = 100  # Pinecone recommends smaller batches for upsert
EMBEDDING_DIMENSIONS = 1536  # OpenAI text-embedding-3-small default


def sanitize_text(text: str) -> str:
    """Remove or replace problematic Unicode characters."""
    if not text:
        return ""
    # Replace common problematic characters
    text = text.replace('\u2019', "'")  # curly apostrophe
    text = text.replace('\u2018', "'")  # left single quote
    text = text.replace('\u201c', '"')  # left double quote
    text = text.replace('\u201d', '"')  # right double quote
    text = text.replace('\u2013', '-')  # en dash
    text = text.replace('\u2014', '-')  # em dash
    text = text.replace('\u2026', '...')  # ellipsis
    # Remove any remaining non-ASCII characters that might cause issues
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text


def get_embeddings_batch(client: OpenAI, texts: list) -> list:
    """Get embeddings for a batch of texts from OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
        dimensions=EMBEDDING_DIMENSIONS  # Reduce dimensions to match Pinecone index
    )
    return [item.embedding for item in response.data]


def main():
    # Check for API keys
    openai_key = os.environ.get("OPENAI_API_KEY")
    pinecone_key = os.environ.get("PINECONE_API_KEY")

    if not openai_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    if not pinecone_key:
        print("Error: PINECONE_API_KEY environment variable not set")
        return

    openai_client = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)

    # Connect to LanceDB and load data
    print(f"Loading data from LanceDB: {LANCEDB_PATH}/{OLD_TABLE_NAME}")
    db = lancedb.connect(LANCEDB_PATH)
    old_table = db.open_table(OLD_TABLE_NAME)
    df = old_table.to_pandas()
    print(f"Found {len(df)} documents")

    # Connect to Pinecone index
    print(f"\nConnecting to Pinecone index: {PINECONE_INDEX_NAME}")
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        print(f"Index stats: {stats}")

        # Check dimensions
        index_dims = stats.get('dimension', 0)
        if index_dims != EMBEDDING_DIMENSIONS:
            print(f"\n*** ERROR: Dimension mismatch! ***")
            print(f"Your index has {index_dims} dimensions")
            print(f"Script configured for {EMBEDDING_DIMENSIONS} dimensions")
            print(f"\nEither update EMBEDDING_DIMENSIONS in this script, or recreate your Pinecone index")
            return

        print(f"Using {EMBEDDING_DIMENSIONS}-dimension embeddings (matches your index)")


    except Exception as e:
        print(f"Error: Could not connect to index '{PINECONE_INDEX_NAME}'")
        print(f"Make sure to create an index with {EMBEDDING_DIMENSIONS} dimensions in Pinecone dashboard")
        print(f"Error details: {e}")
        return

    # Process and upload in batches
    print(f"\nMigrating {len(df)} documents...")
    start_time = time.time()

    texts = df['text'].tolist()

    for i in range(0, len(df), BATCH_SIZE):
        batch_df = df.iloc[i:i + BATCH_SIZE]
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"  Batch {batch_num}/{total_batches} ({i} to {min(i + BATCH_SIZE, len(df))})")

        # Get embeddings from OpenAI
        try:
            embeddings = get_embeddings_batch(openai_client, batch_texts)
        except Exception as e:
            print(f"  OpenAI error: {e}, retrying in 5s...")
            time.sleep(5)
            embeddings = get_embeddings_batch(openai_client, batch_texts)

        # Prepare vectors for Pinecone
        vectors = []
        for j, (idx, row) in enumerate(batch_df.iterrows()):
            # Use hash_doc as ID (or create one)
            doc_id = str(row.get('hash_doc', f'doc_{i + j}'))

            # Metadata (Pinecone has 40KB limit per vector)
            # Sanitize all text to avoid Unicode encoding issues
            metadata = {
                'text': sanitize_text(str(row['text']))[:8000],  # Truncate long text
                'title': sanitize_text(str(row.get('title', ''))),
                'url': str(row.get('url', '')),
                'tags_doc': sanitize_text(str(row.get('tags_doc', '')))[:500],
            }

            vectors.append({
                'id': doc_id,
                'values': embeddings[j],
                'metadata': metadata
            })

        # Upsert to Pinecone
        try:
            index.upsert(vectors=vectors)
        except Exception as e:
            print(f"  Pinecone error: {e}, retrying in 5s...")
            time.sleep(5)
            index.upsert(vectors=vectors)

        # Small delay to avoid rate limits
        time.sleep(0.5)

    elapsed = time.time() - start_time
    print(f"\nMigration complete in {elapsed:.1f} seconds")
    print(f"Uploaded {len(df)} vectors to Pinecone index '{PINECONE_INDEX_NAME}'")

    # Verify
    stats = index.describe_index_stats()
    print(f"\nFinal index stats: {stats}")


if __name__ == "__main__":
    main()
