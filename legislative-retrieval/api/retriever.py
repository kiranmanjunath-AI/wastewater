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
QDRANT_PATH = os.getenv("QDRANT_PATH", "")  # If set, use embedded local storage (no Docker)
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
    """Create and cache the Qdrant client (embedded or remote)."""
    from qdrant_client import QdrantClient
    from pathlib import Path
    if QDRANT_PATH:
        Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=QDRANT_PATH)
    return QdrantClient(url=QDRANT_URL, timeout=15)


def embed_query(query: str) -> list[float]:
    """
    Embed a query string using the E5 query instruction prefix.
    Returns a normalised 384-dim vector.
    """
    model = _get_model()
    vector = model.encode(f"query: {query}", normalize_embeddings=True)
    return vector.tolist()


@lru_cache(maxsize=4)
def _get_vectorizer(language: str):
    """Load the pre-fitted TF-IDF vectorizer for sparse BM25 query encoding."""
    import pickle
    from pathlib import Path

    vec_path = Path(os.getenv("DATA_DIR", "./data")) / f"tfidf_{language}.pkl"
    if vec_path.exists():
        print(f"[retriever] Loading TF-IDF vectorizer for '{language}' ...")
        with open(vec_path, "rb") as f:
            return pickle.load(f)
    return None


def embed_query_sparse(query: str, language: str) -> Optional[tuple[list[int], list[float]]]:
    """
    Compute a TF-IDF sparse vector for the query.
    Returns (indices, values) or None if no vectorizer is available.
    """
    vectorizer = _get_vectorizer(language)
    if vectorizer is None:
        return None
    row = vectorizer.transform([query])
    cx = row.tocoo()
    if cx.nnz == 0:
        return None
    return cx.col.tolist(), cx.data.tolist()


def _build_filter(language: str, as_of: str, act: Optional[str] = None):
    """
    Build the Qdrant filter for a retrieve request.

    V1 prototype: single consolidated version, all chunks have valid_to=None.
    Filter by language and act only.  Temporal range filtering (valid_from/valid_to)
    requires storing dates as numeric timestamps; deferred to V1-production.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    must = [FieldCondition(key="language", match=MatchValue(value=language))]
    if act:
        must.append(FieldCondition(key="act", match=MatchValue(value=act)))
    return Filter(must=must)


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
    qdrant_filter = _build_filter(language, as_of, act)
    sparse = embed_query_sparse(query, language)

    if sparse is not None:
        from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

        sparse_indices, sparse_values = sparse
        raw_results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=query_vector,
                    using="dense",
                    filter=qdrant_filter,
                    limit=top_k * 5,
                ),
                Prefetch(
                    query=SparseVector(indices=sparse_indices, values=sparse_values),
                    using="bm25",
                    filter=qdrant_filter,
                    limit=top_k * 5,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        ).points
    else:
        raw_results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            using="dense",
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        ).points

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
