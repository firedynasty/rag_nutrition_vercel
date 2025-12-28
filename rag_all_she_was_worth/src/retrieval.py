from typing import Any

import lancedb
import numpy as np
import pandas as pd
from lancedb.db import DBConnection
from lancedb.table import Table
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.constants import LANCEDB_URI, get_rag_config

# Cache models at module level
_embedding_model = None
_reranker_model = None


def get_embedding_model():
    """Get or create the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        config = get_rag_config()
        model_name = config["embeddings"]["model_name"]
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def get_reranker_model(model_name: str):
    """Get or create the reranker model."""
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(model_name)
    return _reranker_model


def connect_to_lancedb_table(uri: str, table_name: str) -> Table:
    """Connect to a LanceDB table."""
    db: DBConnection = lancedb.connect(uri=uri)
    return db.open_table(table_name)


def get_knowledge_base(table_name: str | None = None) -> Table:
    """Get the novel knowledge base table."""
    db: DBConnection = lancedb.connect(uri=LANCEDB_URI)
    _table_name: str = table_name or get_rag_config()["knowledge_base"]["table_name"]
    return db.open_table(_table_name)


def retrieve_context(
    k_base: Table,
    query_text: str,
    reranker: dict,
    n_retrieve: int = 10,
) -> list[dict]:
    """
    Retrieve most relevant text chunks using vector search and reranking.

    Args:
        k_base: The knowledge base table
        query_text: The search query
        reranker: Configuration for the cross-encoder reranker
        n_retrieve: Number of chunks to retrieve

    Returns:
        List of retrieved chunks with metadata
    """
    # Get query embedding
    embedding_model = get_embedding_model()
    query_vector = embedding_model.encode(query_text)

    # Retrieve more candidates for reranking
    n_candidates = min(n_retrieve * 3, 50)

    # Vector search
    results = (
        k_base.search(query_vector)
        .limit(n_candidates)
        .to_list()
    )

    if not results:
        return []

    # Rerank with cross-encoder
    rr_model_name: str = reranker["model_name"]
    reranker_model = get_reranker_model(rr_model_name)

    # Prepare pairs for reranking
    pairs = [(query_text, r["text"]) for r in results]
    scores = reranker_model.predict(pairs)

    # Add reranking scores and sort
    for r, score in zip(results, scores):
        r["_relevance_score"] = float(score)

    results.sort(key=lambda x: x["_relevance_score"], reverse=True)

    return results[:n_retrieve]


def group_chunks_by_chapter(resp: list[dict], n_chapters: int = 5) -> list[dict]:
    """
    Group retrieved text chunks by chapter.

    Args:
        resp: List of retrieved chunks
        n_chapters: Number of top chapters to return

    Returns:
        List of chapter dictionaries with aggregated chunks
    """
    if not resp:
        return []

    return (
        pd.DataFrame(resp)
        .drop_duplicates(subset="hash_doc")
        .groupby("chapter")
        .agg(
            chapter_num=("chapter_num", "first"),
            section=("section", "first"),
            n_docs=("n_docs", "first"),
            chunks=("text", list),
            rank_abs=("rank_abs", list),
            score_sum=("_relevance_score", "sum"),
            n_chunks=("text", "count"),
        )
        .reset_index()
        .sort_values(by="score_sum", ascending=False)
        .iloc[:n_chapters]
        .to_dict("records")
    )


def format_context(resp: list[dict]) -> str:
    """
    Format the context into a readable string.

    Args:
        resp: List of chapter dictionaries

    Returns:
        Formatted string with chapter summaries
    """
    if not resp:
        return "No context found."

    output_lines: list[str] = []
    overlap: int = get_rag_config()["embeddings"]["overlap"]

    for i, row in enumerate(resp):
        chapter: str = row.get("chapter", "Unknown")
        section: str = row.get("section", "")

        header = f"{i + 1}. {chapter}"
        if section:
            header += f" - {section}"
        output_lines.append(header + ":")

        chunks: list[str] = row.get("chunks", [])
        rank_abs: list[int] = row.get("rank_abs", [])

        if not chunks:
            output_lines.append("\t- No content available.")
        else:
            previous_rank: int = -1
            for r_current, chunk in zip(rank_abs, chunks):
                if r_current - previous_rank > 1:
                    output_lines.append(f"\t- {chunk}")
                elif len(chunk) > overlap:
                    output_lines[-1] += chunk[overlap:]
                previous_rank = r_current

    return "\n".join(output_lines)


def enrich_text_chunks(k_base: Table, chunks_of_chapter: dict[str, Any], window_size: int = 1) -> dict[str, Any]:
    """
    Sentence-window retrieval: fetch surrounding chunks for context.

    Args:
        k_base: Knowledge base table
        chunks_of_chapter: Dictionary with chapter chunks
        window_size: Number of chunks before/after to fetch

    Returns:
        Enriched dictionary with additional context
    """
    original_ranks: list[int] = chunks_of_chapter["rank_abs"]
    chapter: str = chunks_of_chapter["chapter"]

    new_ranks: set[int] = set()
    for r in original_ranks:
        for step in range(1, window_size + 1):
            new_ranks.add(r - step)
            new_ranks.add(r + step)
    new_ranks.difference_update(original_ranks)

    if not new_ranks:
        return chunks_of_chapter

    new_ranks_str: str = ",".join(map(str, sorted(new_ranks)))
    query_text: str = f"chapter = '{chapter}' AND rank_abs IN ({new_ranks_str})"

    fields: list[str] = ["rank_abs", "text", "n_docs"]
    try:
        new_chunks: pd.DataFrame = k_base.search().where(query_text).to_pandas()[fields]
    except Exception:
        return chunks_of_chapter

    if new_chunks.empty:
        return chunks_of_chapter

    text_dict: dict[int, str] = dict(zip(original_ranks, chunks_of_chapter["chunks"]))
    text_dict.update({c["rank_abs"]: c["text"] for c in new_chunks.to_dict("records")})

    enriched: dict[str, Any] = chunks_of_chapter.copy()
    updated_ranks: list[int] = sorted(text_dict)
    enriched["rank_abs"] = updated_ranks
    enriched["chunks"] = [text_dict[r] for r in updated_ranks]
    enriched["enriched"] = True
    if "n_chunks" in enriched:
        enriched["n_chunks"] = len(updated_ranks)

    return enriched


def get_context(
    k_base: Table,
    query_text: str,
    n_titles: int = 5,
    n_retrieve: int = 10,
    enrich_first: bool = False,
    **kwargs
) -> str:
    """
    Retrieve and format context based on query.

    Args:
        k_base: Knowledge base table
        query_text: Search query
        n_titles: Number of chapters/sections to return
        n_retrieve: Number of chunks to retrieve
        enrich_first: Whether to enrich first result with context

    Returns:
        Formatted context string
    """
    cxt_raw: list[dict] = retrieve_context(
        k_base=k_base, query_text=query_text, n_retrieve=n_retrieve, **kwargs
    )

    cxt_grouped: list[dict] = group_chunks_by_chapter(cxt_raw, n_chapters=n_titles)

    if enrich_first and cxt_grouped:
        cxt_grouped[0] = enrich_text_chunks(k_base=k_base, chunks_of_chapter=cxt_grouped[0])

    return format_context(cxt_grouped)
