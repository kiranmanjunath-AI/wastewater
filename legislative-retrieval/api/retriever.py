"""
retriever.py
Retrieval logic for the Legislative Retrieval API.

Pipeline:
  1. Embed query with "query: " + query (E5 instruction prefix)
  2. Build Qdrant filter: language + temporal validity window
  3. Cosine similarity search, top_k results
  4. Return ranked results with full citation + text + metadata
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "legislation_chunks")


@lru_cache(maxsize=1)
def _get_model():
    """Load the multilingual-e5-small model once and cache it."""
    from sentence_transformers import SentenceTransformer
    print("[retriever] Loading embedding model intfloat/multilingual-e5-small ...")
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    print("[retriever]   Model ready.")
    return model


@lru_cache(maxsize=1)
def _get_qdrant_client():
    """Create and cache the Qdrant client."""
    from qdrant_client import QdrantClient
    return QdrantClient(url=QDRANT_URL, timeout=15)


def embed_query(query: str) -> list[float]:
    """
    Embed a query string using the E5 query instruction prefix.
    Returns a normalised 384-dim vector.
    """
    model = _get_model()
    vector = model.encode(f"query: {query}", normalize_embeddings=True)
    return vector.tolist()


def _build_filter(language: str, as_of: str) -> dict:
    """
    Build the Qdrant filter dict for a retrieve request.

    Conditions:
      - language == req.language
      - valid_from <= as_of  (stored as ISO string, YYYY-MM-DD)
      - valid_to > as_of OR valid_to is null (still valid / no expiry)

    Because Qdrant string comparisons are lexicographic and ISO dates
    sort correctly as strings, we use Range conditions.
    """
    return {
        "must": [
            {"key": "language", "match": {"value": language}},
            {"key": "valid_from", "range": {"lte": as_of}},
        ],
        "should": [
            {"key": "valid_to", "is_null": {}},
            {"key": "valid_to", "range": {"gt": as_of}},
        ],
        "minimum_should": 1,
    }


def search(
    query: str,
    top_k: int = 5,
    language: str = "en",
    as_of: Optional[str] = None,
    act: Optional[str] = "ITA",
) -> list[dict]:
    """
    Perform a cosine similarity search over legislation_chunks.

    Returns a list of dicts, each containing:
      citation, text, score, language, valid_from, valid_to,
      amending_act, section, subsection, paragraph, part, division
    """
    client = _get_qdrant_client()

    if as_of is None:
        as_of = date.today().isoformat()

    query_vector = embed_query(query)
    qdrant_filter = _build_filter(language, as_of)

    # If an act filter is provided, add it to the must conditions
    if act:
        qdrant_filter["must"].append({"key": "act", "match": {"value": act}})

    raw_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    results = []
    for hit in raw_results:
        p = hit.payload or {}
        results.append({
            "citation": p.get("citation", ""),
            "text": p.get("text", ""),
            "score": round(float(hit.score), 6),
            "language": p.get("language", language),
            "valid_from": p.get("valid_from", ""),
            "valid_to": p.get("valid_to"),
            "amending_act": p.get("amending_act"),
            "section": p.get("section"),
            "subsection": p.get("subsection"),
            "paragraph": p.get("paragraph"),
            "part": p.get("part"),
            "division": p.get("division"),
        })

    return results


def get_consolidated_as_of(act: str = "ITA", language: str = "en") -> Optional[str]:
    """
    Return the consolidated_as_of date of the most recently ingested version,
    by querying the SQLite version index.
    """
    try:
        from db.version_index import get_latest_version
        latest = get_latest_version(act=act, language=language)
        if latest:
            return latest.get("consolidated_as_of")
    except Exception:
        pass
    return None


def health_check() -> dict:
    """
    Check Qdrant connectivity and collection status.
    Returns a dict with status, chunk_count, and latest_version.
    """
    try:
        client = _get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            return {
                "qdrant": "ok",
                "collection": "missing",
                "chunk_count": 0,
                "latest_version": None,
            }
        count = client.count(collection_name=COLLECTION_NAME).count
        return {
            "qdrant": "ok",
            "collection": "ok",
            "chunk_count": count,
            "latest_version": get_consolidated_as_of(),
        }
    except Exception as exc:
        return {
            "qdrant": f"error: {exc}",
            "collection": "unknown",
            "chunk_count": 0,
            "latest_version": None,
        }
