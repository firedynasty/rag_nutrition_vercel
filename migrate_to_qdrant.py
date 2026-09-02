#!/usr/bin/env python3
"""
Migrate any LanceDB RAG folder to a Qdrant collection.

Reads rag_config.toml from the rag folder to find the database and table.
No re-embedding needed — vectors are uploaded as-is.

Usage:
  python migrate_to_qdrant.py --rag-folder rag_nutrition --collection nutrition-rag
  python migrate_to_qdrant.py --rag-folder ../streamlit_rags/rag_chess --collection rag-chess
  python migrate_to_qdrant.py --rag-folder ../streamlit_rags/rag_romans --collection rag-romans
  python migrate_to_qdrant.py --rag-folder rag_nutrition --collection nutrition-rag --table nutrition_openai

Requirements:
  pip install qdrant-client lancedb pandas toml

Environment variables:
  QDRANT_URL     - Your Qdrant cluster URL (e.g. https://xyz.us-west-1-0.aws.cloud.qdrant.io)
  QDRANT_API_KEY - Your Qdrant API key
"""

import argparse
import os
import time
from pathlib import Path

import lancedb
import toml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

BATCH_SIZE = 100


def main():
    parser = argparse.ArgumentParser(description="Migrate a LanceDB RAG folder to Qdrant")
    parser.add_argument("--rag-folder", required=True, help="Path to rag_* folder (e.g. rag_nutrition or ../streamlit_rags/rag_chess)")
    parser.add_argument("--collection", required=True, help="Qdrant collection name (e.g. nutrition-rag, rag-chess)")
    parser.add_argument("--table", help="LanceDB table name override (default: read from rag_config.toml)")
    args = parser.parse_args()

    qdrant_url = os.environ.get("QDRANT_URL", "").rstrip("/")
    qdrant_key = os.environ.get("QDRANT_API_KEY")

    if not qdrant_url:
        print("Error: QDRANT_URL environment variable not set")
        return
    if not qdrant_key:
        print("Error: QDRANT_API_KEY environment variable not set")
        return

    rag_folder = Path(args.rag_folder)
    if not rag_folder.exists():
        print(f"Error: folder not found: {rag_folder}")
        return

    # Read rag_config.toml to find db path and table name
    config_path = rag_folder / "rag_config.toml"
    if not config_path.exists():
        print(f"Error: rag_config.toml not found in {rag_folder}")
        return

    config = toml.load(config_path)
    db_uri = config["knowledge_base"]["uri"]
    table_name = args.table or config["knowledge_base"]["table_name"]
    lancedb_path = rag_folder / db_uri

    print(f"RAG folder:  {rag_folder}")
    print(f"LanceDB:     {lancedb_path}")
    print(f"Table:       {table_name}")
    print(f"Collection:  {args.collection}")

    # Load data from LanceDB
    print(f"\nLoading data...")
    db = lancedb.connect(str(lancedb_path))

    available_tables = db.table_names() if hasattr(db, 'table_names') else db.list_tables()
    if table_name not in available_tables:
        print(f"Error: table '{table_name}' not found. Available: {list(available_tables)}")
        return

    table = db.open_table(table_name)
    df = table.to_pandas()

    # Detect dimensions from first vector
    sample_vec = df["vector"].iloc[0]
    dims = len(sample_vec.tolist() if hasattr(sample_vec, "tolist") else list(sample_vec))
    print(f"Found {len(df)} vectors ({dims}-dim)")

    # Connect to Qdrant
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

    # Create or confirm collection
    existing = [c.name for c in client.get_collections().collections]
    if args.collection in existing:
        info = client.get_collection(args.collection)
        existing_dims = info.config.params.vectors.size
        if existing_dims != dims:
            print(f"Error: collection '{args.collection}' exists with {existing_dims} dims but data has {dims} dims")
            return
        print(f"\nCollection '{args.collection}' already exists ({info.points_count} points) — will upsert")
    else:
        print(f"\nCreating collection '{args.collection}' ({dims}-dim, Cosine)...")
        client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=dims, distance=Distance.COSINE)
        )
        print("Collection created.")

    # Upload in batches
    print(f"\nUploading {len(df)} vectors...")
    start_time = time.time()
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{total_batches} ({i} to {min(i + BATCH_SIZE, len(df))})")

        points = []
        for j, (_, row) in enumerate(batch.iterrows()):
            vector = row["vector"].tolist() if hasattr(row["vector"], "tolist") else list(row["vector"])

            # Build payload from whatever columns exist
            payload = {"text": str(row.get("text", ""))}
            for col in ["title", "url", "section", "chapter", "hash_doc", "tags_doc", "tags_all"]:
                if col in row and row[col] is not None:
                    payload[col] = str(row[col])

            points.append(PointStruct(id=i + j, vector=vector, payload=payload))

        client.upsert(collection_name=args.collection, points=points)
        time.sleep(0.2)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s — {len(df)} vectors uploaded to '{args.collection}'")

    info = client.get_collection(args.collection)
    print(f"Collection stats: {info.points_count} points, status: {info.status}")


if __name__ == "__main__":
    main()
