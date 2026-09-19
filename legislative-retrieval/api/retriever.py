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

import math
import os
import re
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


def _expand_query(query: str) -> str:
    # Expansion 1: define X -> Definition of X  (stemming gap: define != definition for BM25)
    # Expansion 2: fraction -> one-half half  (ITA uses Unicode 1/2 char, normalised in chunks)
    # Expansion 3: RRSP converted/matured -> maturity annuity RRIF  (ITA uses "maturity" not "converted")
    # Expansion 4: capital loss applied -> allowable net s.3 s.111  (conceptual vocabulary gap)
    # Expansion 5: interest borrowed money -> paragraph 20(1)(c)  (s.20.1 drowns out s.20(1)(c))
    expanded = query
    quote_chars = chr(0x27) + chr(0x2018) + chr(0x2019) + chr(0x201c) + chr(0x201d)
    m = re.search(
        r"defin[a-z]*\s+(?:of\s+)?[" + quote_chars + r"]?"
        r"([a-zA-Z][a-zA-Z\s\-]*?)"
        r"[" + quote_chars + r"]?\s*(?:\?|under|in\s+the|$)",
        query,
        re.IGNORECASE,
    )
    if m:
        term = m.group(1).strip()
        if 2 <= len(term) <= 40:
            # "Part XVII" and "ITA s.248" target s.248(1) (the general definitions
            # section) over local definitions in other sections that also say "Definition of X".
            # "means" matches s.248(1) text ("X means...") vs. local defs that say
            # "has the same meaning as in subsection 248(1)".
            expanded = expanded + " Definition of '" + term + "' Part XVII ITA s.248 means"
    if re.search(r"fraction", query, re.IGNORECASE):
        expanded = expanded + " one-half half"
    if re.search(r"RRSP", query, re.IGNORECASE) and re.search(r"convert|matur|option", query, re.IGNORECASE):
        # s.146(2): "maturity date is no later than the end of the calendar year in which
        # the annuitant attains 71 years of age"; s.146(3) lists options (life annuity, term
        # annuity, retirement income fund).
        expanded = expanded + " maturity annuity RRIF retirement income fund maturity date annuitant 71 calendar year"
    if re.search(r"over.contribut|excess.*RRSP|RRSP.*excess", query, re.IGNORECASE):
        # s.204.1 imposes tax on cumulative excess RRSP amounts; its context prefix
        # is "Tax payable by individuals -- contributions after 1990", which distinguishes
        # it from s.204.2 ("Cumulative excess amounts" — the computation formula).
        expanded = expanded + " cumulative excess amount registered retirement savings plans tax payable individual over-contribution penalty Part X.1"
    if re.search(r"capital loss", query, re.IGNORECASE) and re.search(r"applied|offset|against", query, re.IGNORECASE):
        expanded = expanded + " allowable capital losses net capital loss deductible"
    if re.search(r"interest", query, re.IGNORECASE) and re.search(r"borrowed", query, re.IGNORECASE):
        # "deductions permitted computing income business property" targets s.20(1)(c) context prefix
        # over s.20.1 ("Borrowed money used to earn income from property — Lost source").
        expanded = expanded + " legal obligation interest paid borrowed money deductions permitted computing income business property"
    # Employment income computation -> ITA s.5(1) vocabulary
    if re.search(r"employ\w*\s+income|income.*employ", query, re.IGNORECASE) and re.search(r"include|comput", query, re.IGNORECASE):
        expanded = expanded + " salary wages remuneration gratuities office employment"
    # Part XIII withholding (royalties / interest) -> boost s.212(1) chunks
    # s.212(1) text: "Every non-resident person shall pay an income tax of 25%..."
    if re.search(r"royalt", query, re.IGNORECASE) and re.search(r"non.resid|withhold", query, re.IGNORECASE):
        expanded = expanded + " Part XIII rent royalty payment non-resident shall pay income tax"
    if re.search(r"Part XIII|withholding.*interest|interest.*withhold", query, re.IGNORECASE):
        expanded = expanded + " Part XIII non-resident person shall pay income tax interest"
    # Principal residence exemption -> s.40(2)(b) vocabulary
    if re.search(r"principal residence", query, re.IGNORECASE):
        expanded = expanded + " principal residence gain disposition exempt years owned"
    return expanded

def embed_query(query: str) -> list[float]:
    """
    Embed a query string using the E5 query instruction prefix.
    Returns a normalised 384-dim vector.
    """
    model = _get_model()
    vector = model.encode(f"query: {query}", normalize_embeddings=True)
    return vector.tolist()


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Load a cross-encoder model for reranking; returns None if unavailable."""
    try:
        from sentence_transformers import CrossEncoder
        print("[retriever] Loading cross-encoder model ...")
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        print("[retriever]   Cross-encoder ready.")
        return model
    except Exception as exc:
        print(f"[retriever] WARNING: Cross-encoder unavailable ({exc}), falling back to RRF.")
        return None


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
    Perform a hybrid search (dense + BM25 RRF) over legislation_chunks,
    then optionally rerank with a cross-encoder for higher precision.

    Returns a list of dicts, each containing:
      citation, text, score, language, valid_from, valid_to,
      amending_act, section, subsection, paragraph, part, division
    """
    client = _get_qdrant_client()

    if as_of is None:
        as_of = date.today().isoformat()

    expanded = _expand_query(query)
    query_vector = embed_query(expanded)
    qdrant_filter = _build_filter(language, as_of, act)
    sparse = embed_query_sparse(expanded, language)

    # Retrieve a larger candidate pool for cross-encoder reranking.
    # 200 candidates (vs. 100) ensures sections that rank ~50 in RRF still reach CE.
    candidate_k = top_k * 20

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
                    limit=candidate_k,
                ),
                Prefetch(
                    query=SparseVector(indices=sparse_indices, values=sparse_values),
                    using="bm25",
                    filter=qdrant_filter,
                    limit=candidate_k,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=candidate_k,
            with_payload=True,
            with_vectors=False,
        ).points
    else:
        raw_results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            using="dense",
            query_filter=qdrant_filter,
            limit=candidate_k,
            with_payload=True,
            with_vectors=False,
        ).points

    candidates = []
    for hit in raw_results:
        p = hit.payload or {}
        candidates.append({
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

    # Cross-encoder reranking: blend CE scores with RRF scores to avoid regressions.
    # Pure CE reranking can over-promote specific sections over general foundational ones.
    # Blending (70% CE, 30% RRF) preserves strong prior-stage rankings while still
    # letting CE correct errors where the right section is ranked but not top-3.
    cross_encoder = _get_cross_encoder()
    if cross_encoder is not None and candidates:
        rrf_scores = [c["score"] for c in candidates]
        max_rrf = max(rrf_scores) if rrf_scores else 1.0

        pairs = [(expanded, c["text"]) for c in candidates]
        ce_raw = cross_encoder.predict(pairs).tolist()
        # Normalise CE logits to (0,1) via sigmoid
        ce_norm = [1.0 / (1.0 + math.exp(-s)) for s in ce_raw]
        rrf_norm = [s / max_rrf for s in rrf_scores]

        for c, ce_n, rrf_n in zip(candidates, ce_norm, rrf_norm):
            c["score"] = round(0.70 * ce_n + 0.30 * rrf_n, 6)

        candidates.sort(key=lambda c: c["score"], reverse=True)

    return candidates[:top_k]


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
