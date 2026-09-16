"""
Build and manage the vector + BM25 index.

ChromaDB (cosine similarity) stores dense embeddings from Voyage Finance-2.
BM25Okapi provides sparse retrieval; both indexes are used together in Phase 3
via Reciprocal Rank Fusion.

The build is incremental by default: chunks already in ChromaDB are not
re-embedded. BM25 is always rebuilt from the full corpus to keep it consistent.
"""

import logging
import os
import pickle
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import voyageai

from ..ingest.base import DATA_DIR
from .chunker import Chunk

log = logging.getLogger(__name__)

CHROMA_DIR   = DATA_DIR / "chroma"
BM25_FILE    = DATA_DIR / "bm25.pkl"
COLLECTION   = "central_bank_docs"
VOYAGE_MODEL = "voyage-finance-2"
EMBED_BATCH  = 64    # ~25K tokens/batch
_RATE_LIMIT_PAUSE = 1    # small gap; retry logic handles any remaining throttling


# ── ChromaDB helpers ───────────────────────────────────────────────────────────

def _chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


# ── Embedding ──────────────────────────────────────────────────────────────────

def _embed(texts: list[str], vo: voyageai.Client, attempt: int = 0) -> list[list[float]]:
    """Embed with automatic retry on rate-limit errors (exponential backoff)."""
    try:
        result = vo.embed(texts, model=VOYAGE_MODEL, input_type="document")
        return result.embeddings
    except Exception as exc:
        if attempt >= 5:
            raise
        # Any rate-limit or server error: back off and retry
        wait = _RATE_LIMIT_PAUSE * (2 ** attempt)
        log.warning("Embed error (%s) — waiting %ds before retry %d/5", exc.__class__.__name__, wait, attempt + 1)
        time.sleep(wait)
        return _embed(texts, vo, attempt + 1)


# ── Main build ─────────────────────────────────────────────────────────────────

def build_index(chunks: list[Chunk], incremental: bool = True) -> None:
    """
    Embed chunks and persist to ChromaDB + BM25.
    incremental=True skips chunks whose IDs already exist in ChromaDB.
    """
    if not chunks:
        log.warning("No chunks to index")
        return

    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY not set in environment / .env")

    vo = voyageai.Client(api_key=api_key)
    collection = _chroma_collection()

    # -- Figure out which chunks are new --
    if incremental:
        existing: set[str] = set(collection.get(include=[])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing]
    else:
        new_chunks = chunks

    log.info(
        "Embedding %d new chunks (%d already indexed)",
        len(new_chunks), len(chunks) - len(new_chunks),
    )

    # -- Embed and store new chunks in batches --
    n_batches = (len(new_chunks) + EMBED_BATCH - 1) // EMBED_BATCH
    eta_min = round(n_batches * _RATE_LIMIT_PAUSE / 60)
    if eta_min > 2:
        log.info(
            "Free-tier rate limits detected — throttling to ~3 req/min. "
            "Add a payment method at dashboard.voyageai.com to speed this up. "
            "Estimated time: ~%d min.", eta_min,
        )

    for i in tqdm(range(0, len(new_chunks), EMBED_BATCH), desc="Embedding", unit="batch"):
        batch      = new_chunks[i : i + EMBED_BATCH]
        texts      = [c.text for c in batch]
        embeddings = _embed(texts, vo)
        # Pause between batches to stay within free-tier rate limits.
        # If the account has a payment method this sleep is the only overhead (~1s wasted).
        if i + EMBED_BATCH < len(new_chunks):
            time.sleep(_RATE_LIMIT_PAUSE)

        collection.add(
            ids        = [c.chunk_id  for c in batch],
            embeddings = embeddings,
            documents  = texts,
            metadatas  = [
                {
                    "doc_id":      c.doc_id,
                    "institution": c.institution,
                    "doc_type":    c.doc_type,
                    "date":        c.date,
                    "title":       c.title,
                    "url":         c.url,
                    "page_num":    c.page_num,
                    "chunk_idx":   c.chunk_idx,
                }
                for c in batch
            ],
        )

    # -- Rebuild BM25 over full corpus (always, to stay consistent) --
    log.info("Rebuilding BM25 index over %d chunks…", len(chunks))
    tokenized = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_FILE, "wb") as f:
        pickle.dump(
            {"bm25": bm25, "chunk_ids": [c.chunk_id for c in chunks], "chunks": chunks},
            f,
        )

    log.info(
        "Index complete — %d vectors in ChromaDB, BM25 at %s",
        collection.count(), BM25_FILE.name,
    )


# ── Load helpers (used by retriever in Phase 3) ────────────────────────────────

def load_chroma() -> chromadb.Collection:
    return _chroma_collection()


def load_bm25() -> tuple[BM25Okapi, list[str], list["Chunk"]]:
    """Returns (bm25_index, chunk_id_list, chunk_list)."""
    with open(BM25_FILE, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunk_ids"], data["chunks"]
