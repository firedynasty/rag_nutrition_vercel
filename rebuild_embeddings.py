#!/usr/bin/env python3
"""
Rebuild LanceDB embeddings using OpenAI text-embedding-3-small.

Migrates from sentence-transformers (384 dims) to OpenAI embeddings (1536 dims).
Reads rag_config.toml to find the database and source table automatically.

Usage:
  python rebuild_embeddings.py --rag-folder rag_nutrition
  python rebuild_embeddings.py --rag-folder ../streamlit_rags/rag_chess
  python rebuild_embeddings.py --rag-folder rag_nutrition --source nutrition_openai  (re-use existing)

Requirements:
  pip install openai lancedb pandas toml

Environment variables:
  OPENAI_API_KEY - Your OpenAI API key
"""

import argparse
import os
import time
from pathlib import Path

import toml
from openai import OpenAI
import lancedb

BATCH_SIZE = 1000  # OpenAI allows up to 2048, but 1000 is safer


def get_embeddings_batch(client: OpenAI, texts: list) -> list:
    """Get embeddings for a batch of texts."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]


def main():
    parser = argparse.ArgumentParser(description="Re-embed a LanceDB RAG folder with OpenAI embeddings")
    parser.add_argument("--rag-folder", required=True, help="Path to rag_* folder (e.g. rag_nutrition or ../streamlit_rags/rag_chess)")
    parser.add_argument("--source", help="Source table name override (default: read from rag_config.toml)")
    parser.add_argument("--output", help="Output table name (default: <source>_openai)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    rag_folder = Path(args.rag_folder)
    if not rag_folder.exists():
        print(f"Error: folder not found: {rag_folder}")
        return

    config_path = rag_folder / "rag_config.toml"
    if not config_path.exists():
        print(f"Error: rag_config.toml not found in {rag_folder}")
        return

    config = toml.load(config_path)
    db_uri = config["knowledge_base"]["uri"]
    source_table = args.source or config["knowledge_base"]["table_name"]
    output_table = args.output or f"{source_table}_openai"
    lancedb_path = rag_folder / db_uri

    print(f"RAG folder:    {rag_folder}")
    print(f"LanceDB:       {lancedb_path}")
    print(f"Source table:  {source_table}")
    print(f"Output table:  {output_table}")

    client = OpenAI(api_key=api_key)

    db = lancedb.connect(str(lancedb_path))

    available = db.table_names() if hasattr(db, 'table_names') else db.list_tables()
    if source_table not in available:
        print(f"Error: table '{source_table}' not found. Available: {list(available)}")
        return

    print(f"\nLoading data from '{source_table}'...")
    df = db.open_table(source_table).to_pandas()
    print(f"Found {len(df)} documents")

    # Check current dimensions
    sample_vec = df["vector"].iloc[0]
    current_dims = len(sample_vec.tolist() if hasattr(sample_vec, "tolist") else list(sample_vec))
    print(f"Current embedding dims: {current_dims}")

    if current_dims == 1536:
        print("Already 1536-dim (OpenAI). Use --output to save to a different table name if needed.")
        return

    texts = df['text'].tolist()
    all_embeddings = []

    print(f"\nRe-embedding {len(texts)} documents with text-embedding-3-small (1536-dim)...")
    start_time = time.time()

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches} ({i} to {min(i + BATCH_SIZE, len(texts))})")

        try:
            embeddings = get_embeddings_batch(client, batch)
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"  Error: {e}, retrying in 5s...")
            time.sleep(5)
            all_embeddings.extend(get_embeddings_batch(client, batch))

    elapsed = time.time() - start_time
    print(f"\nEmbedding complete in {elapsed:.1f}s")

    df['vector'] = all_embeddings

    print(f"Saving to table '{output_table}'...")
    db.drop_table(output_table, ignore_missing=True)
    db.create_table(output_table, df)

    print(f"\nDone! {len(df)} documents saved to '{output_table}'")
    print(f"\nNext step:")
    print(f"  python migrate_to_qdrant.py --rag-folder {rag_folder} --collection <name> --table {output_table}")


if __name__ == "__main__":
    main()
