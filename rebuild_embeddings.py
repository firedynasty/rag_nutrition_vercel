#!/usr/bin/env python3
"""
Rebuild LanceDB embeddings using OpenAI text-embedding-3-small.

Migrates from sentence-transformers (384 dims) to OpenAI embeddings (1536 dims).
Uses batch embedding for speed (up to 2048 texts per request).
"""

import os
from openai import OpenAI
import lancedb
import pandas as pd
import time

# Configuration
LANCEDB_PATH = "rag_nutrition/databases/my_lancedb"
OLD_TABLE_NAME = "table_simple05"
NEW_TABLE_NAME = "nutrition_openai"
BATCH_SIZE = 1000  # OpenAI allows up to 2048, but 1000 is safer


def get_embeddings_batch(client: OpenAI, texts: list) -> list:
    """Get embeddings for a batch of texts."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    # Return embeddings in the same order as input
    return [item.embedding for item in response.data]


def main():
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Run: export OPENAI_API_KEY='your-key-here'")
        return

    client = OpenAI(api_key=api_key)

    # Connect to LanceDB
    print(f"Connecting to LanceDB at: {LANCEDB_PATH}")
    db = lancedb.connect(LANCEDB_PATH)

    # Load existing data
    print(f"Loading data from table: {OLD_TABLE_NAME}")
    old_table = db.open_table(OLD_TABLE_NAME)
    df = old_table.to_pandas()

    print(f"Found {len(df)} documents to re-embed")
    print(f"Using batch size: {BATCH_SIZE}")

    # Get all texts
    texts = df['text'].tolist()
    all_embeddings = []

    # Process in batches
    print("\nRe-embedding documents with OpenAI text-embedding-3-small...")
    start_time = time.time()

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"  Batch {batch_num}/{total_batches} ({i} to {min(i + BATCH_SIZE, len(texts))} of {len(texts)})")

        try:
            embeddings = get_embeddings_batch(client, batch)
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"  Error on batch {batch_num}: {e}")
            print("  Retrying in 5 seconds...")
            time.sleep(5)
            embeddings = get_embeddings_batch(client, batch)
            all_embeddings.extend(embeddings)

    elapsed = time.time() - start_time
    print(f"\nEmbedding complete in {elapsed:.1f} seconds")

    # Update dataframe
    df['vector'] = all_embeddings

    # Create new table
    print(f"\nCreating new table: {NEW_TABLE_NAME}")
    db.drop_table(NEW_TABLE_NAME, ignore_missing=True)
    db.create_table(NEW_TABLE_NAME, df)

    print(f"\nSuccess! Re-embedded {len(df)} documents")
    print(f"\nNext step: Update TABLE_NAME in api/rag.py to \"{NEW_TABLE_NAME}\"")


if __name__ == "__main__":
    main()
