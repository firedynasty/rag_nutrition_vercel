#!/usr/bin/env python3
"""
Upload a book's LanceDB data to Pinecone with OpenAI embeddings.

Usage:
  python upload_book_to_pinecone.py --rag-folder ../streamlit_apps/rag_romeo_and_juliet --index rag-romeo-and-juliet

Requirements:
  pip install pinecone openai lancedb pandas

Environment variables:
  OPENAI_API_KEY - Your OpenAI API key
  PINECONE_API_KEY - Your Pinecone API key
"""

import argparse
import os
import time
import toml
from openai import OpenAI
from pinecone import Pinecone
import lancedb

BATCH_SIZE = 100
EMBEDDING_DIMENSIONS = 1536  # OpenAI text-embedding-3-small


def sanitize_text(text: str) -> str:
    """Remove or replace problematic Unicode characters."""
    if not text:
        return "[empty]"
    text = str(text)
    text = text.replace('\u2019', "'")
    text = text.replace('\u2018', "'")
    text = text.replace('\u201c', '"')
    text = text.replace('\u201d', '"')
    text = text.replace('\u2013', '-')
    text = text.replace('\u2014', '-')
    text = text.replace('\u2026', '...')
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.strip()
    # OpenAI rejects empty strings
    if not text:
        return "[empty]"
    return text


def get_embeddings_batch(client: OpenAI, texts: list) -> list:
    """Get embeddings for a batch of texts from OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
        dimensions=EMBEDDING_DIMENSIONS
    )
    return [item.embedding for item in response.data]


def main():
    parser = argparse.ArgumentParser(description="Upload book RAG to Pinecone")
    parser.add_argument("--rag-folder", required=True, help="Path to rag_* folder (e.g., ../streamlit_apps/rag_romeo_and_juliet)")
    parser.add_argument("--index", required=True, help="Pinecone index name (e.g., rag-romeo-and-juliet)")
    args = parser.parse_args()

    # Check for API keys
    openai_key = os.environ.get("OPENAI_API_KEY")
    pinecone_key = os.environ.get("PINECONE_API_KEY")

    if not openai_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    if not pinecone_key:
        print("Error: PINECONE_API_KEY environment variable not set")
        return

    # Read rag_config.toml to get database path and table name
    config_path = os.path.join(args.rag_folder, "rag_config.toml")
    if not os.path.exists(config_path):
        print(f"Error: Config not found: {config_path}")
        return

    config = toml.load(config_path)
    db_uri = config["knowledge_base"]["uri"]
    table_name = config["knowledge_base"]["table_name"]

    # Build full path to database
    lancedb_path = os.path.join(args.rag_folder, db_uri)
    print(f"LanceDB path: {lancedb_path}")
    print(f"Table name: {table_name}")

    openai_client = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)

    # Connect to LanceDB and load data
    print(f"\nLoading data from LanceDB...")
    db = lancedb.connect(lancedb_path)
    old_table = db.open_table(table_name)
    df = old_table.to_pandas()
    print(f"Found {len(df)} documents")

    # Preview columns
    print(f"Columns: {list(df.columns)}")

    # Connect to Pinecone index
    print(f"\nConnecting to Pinecone index: {args.index}")
    try:
        index = pc.Index(args.index)
        stats = index.describe_index_stats()
        print(f"Index stats: {stats}")
    except Exception as e:
        print(f"Error: Could not connect to index '{args.index}'")
        print(f"Make sure to create an index with {EMBEDDING_DIMENSIONS} dimensions in Pinecone dashboard")
        print(f"Error details: {e}")
        return

    # Process and upload in batches
    print(f"\nUploading {len(df)} documents...")
    start_time = time.time()

    texts = df['text'].tolist()

    for i in range(0, len(df), BATCH_SIZE):
        batch_df = df.iloc[i:i + BATCH_SIZE]
        batch_texts = [sanitize_text(t) for t in texts[i:i + BATCH_SIZE]]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"  Batch {batch_num}/{total_batches} ({i} to {min(i + BATCH_SIZE, len(df))})")

        # Get embeddings from OpenAI
        try:
            embeddings = get_embeddings_batch(openai_client, batch_texts)
        except Exception as e:
            print(f"  OpenAI error: {e}")
            # Debug: show problematic texts
            for idx, t in enumerate(batch_texts):
                if not t or len(t) < 2:
                    print(f"    Problem text at index {idx}: '{t[:50] if t else 'EMPTY'}'")
            print("  Retrying in 5s...")
            time.sleep(5)
            embeddings = get_embeddings_batch(openai_client, batch_texts)

        # Prepare vectors for Pinecone
        vectors = []
        for j, (idx, row) in enumerate(batch_df.iterrows()):
            doc_id = str(row.get('hash_doc', f'doc_{i + j}'))

            # Build metadata - handle different schema for books vs nutrition
            metadata = {
                'text': sanitize_text(str(row['text']))[:8000],
            }

            # Add chapter info if present (books)
            if 'chapter' in row:
                metadata['title'] = sanitize_text(str(row.get('chapter', '')))
                metadata['chapter_num'] = int(row.get('chapter_num', 0))
                metadata['section'] = str(row.get('section', ''))

            # Add title/url if present (nutrition)
            if 'title' in row:
                metadata['title'] = sanitize_text(str(row.get('title', '')))
            if 'url' in row:
                metadata['url'] = str(row.get('url', ''))

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

        time.sleep(0.5)

    elapsed = time.time() - start_time
    print(f"\nUpload complete in {elapsed:.1f} seconds")
    print(f"Uploaded {len(df)} vectors to Pinecone index '{args.index}'")

    # Verify
    stats = index.describe_index_stats()
    print(f"\nFinal index stats: {stats}")

    # Print the host for this index (needed for env var)
    print(f"\n=== IMPORTANT ===")
    print(f"Add this environment variable to Vercel:")
    env_key = f"PINECONE_HOST_{args.index.upper().replace('-', '_')}"
    print(f"  {env_key}=<your-pinecone-host>")
    print(f"\nGet the host from Pinecone dashboard > Indexes > {args.index}")


if __name__ == "__main__":
    main()
