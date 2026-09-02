#!/usr/bin/env python3
"""
Migrate LanceDB nutrition data to Qdrant.

Reads from the nutrition_openai table which already has 1536-dim OpenAI embeddings
so NO re-embedding is needed — vectors are uploaded as-is.

Requirements:
  pip install qdrant-client lancedb pandas

Environment variables:
  QDRANT_URL     - Your Qdrant cluster URL (e.g. https://xyz.us-east-1-0.aws.cloud.qdrant.io)
  QDRANT_API_KEY - Your Qdrant API key
"""

import os
import time
import lancedb
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Configuration
LANCEDB_PATH = "rag_nutrition/databases/my_lancedb"
LANCEDB_TABLE = "nutrition_openai"   # already has 1536-dim OpenAI embeddings
COLLECTION_NAME = "nutrition-rag"    # Qdrant collection name (matches DEFAULT_INDEX in api/rag.py)
EMBEDDING_DIMENSIONS = 1536
BATCH_SIZE = 100


def main():
    qdrant_url = os.environ.get("QDRANT_URL", "").rstrip("/")
    qdrant_key = os.environ.get("QDRANT_API_KEY")

    if not qdrant_url:
        print("Error: QDRANT_URL environment variable not set")
        return
    if not qdrant_key:
        print("Error: QDRANT_API_KEY environment variable not set")
        return

    # Connect to Qdrant
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

    # Load data from LanceDB
    print(f"Loading data from LanceDB: {LANCEDB_PATH}/{LANCEDB_TABLE}")
    db = lancedb.connect(LANCEDB_PATH)
    table = db.open_table(LANCEDB_TABLE)
    df = table.to_pandas()
    print(f"Found {len(df)} vectors ({EMBEDDING_DIMENSIONS}-dim, no re-embedding needed)")

    # Create collection if it doesn't exist
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"\nCollection '{COLLECTION_NAME}' already exists — will upsert into it")
    else:
        print(f"\nCreating collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIMENSIONS, distance=Distance.COSINE)
        )
        print("Collection created.")

    # Upload in batches
    print(f"\nUploading {len(df)} vectors to Qdrant...")
    start_time = time.time()
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{total_batches} ({i} to {min(i + BATCH_SIZE, len(df))})")

        points = []
        for j, (_, row) in enumerate(batch.iterrows()):
            point_id = i + j  # Qdrant requires integer or UUID ids
            vector = row["vector"].tolist() if hasattr(row["vector"], "tolist") else list(row["vector"])

            payload = {
                "text": str(row.get("text", "")),
                "title": str(row.get("title", "")),
                "url": str(row.get("url", "")),
                "tags_doc": str(row.get("tags_doc", "")),
                "hash_doc": str(row.get("hash_doc", "")),
            }

            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        time.sleep(0.2)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s — {len(df)} vectors uploaded to '{COLLECTION_NAME}'")

    # Verify
    info = client.get_collection(COLLECTION_NAME)
    print(f"Collection stats: {info.points_count} points, status: {info.status}")


if __name__ == "__main__":
    main()
