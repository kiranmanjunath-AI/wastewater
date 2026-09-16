"""
Hybrid retriever: dense (ChromaDB + Voyage Finance-2) + sparse (BM25).
Results merged with Reciprocal Rank Fusion (RRF).

Designed to be instantiated once and reused across queries (Streamlit cache).
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import voyageai

from ..embed.chunker import Chunk
from ..embed.indexer import load_chroma, load_bm25

log = logging.getLogger(__name__)

VOYAGE_MODEL  = "voyage-finance-2"
TOP_K_DENSE   = 20   # candidates pulled from ChromaDB
TOP_K_SPARSE  = 20   # candidates pulled from BM25
RRF_K         = 60   # RRF constant (standard value)
DEFAULT_TOP_N = 6    # final chunks passed to the generator


@dataclass
class RetrievalResult:
    chunk:        Chunk
    rrf_score:    float
    dense_rank:   Optional[int]   # None if not in dense top-K
    sparse_rank:  Optional[int]   # None if not in sparse top-K


def _rrf(rank: Optional[int]) -> float:
    return 0.0 if rank is None else 1.0 / (rank + RRF_K)


class Retriever:
    """Load indexes once; query many times."""

    def __init__(self) -> None:
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY not set — check your .env file")
        self._vo         = voyageai.Client(api_key=api_key)
        self._collection = load_chroma()
        self._bm25, self._chunk_ids, self._chunks = load_bm25()
        self._chunk_map  = {c.chunk_id: c for c in self._chunks}
        log.info("Retriever ready: %d chunks in index", len(self._chunks))

    # ── Filtering ──────────────────────────────────────────────────────────────

    def _chroma_where(
        self,
        institution: Optional[str],
        doc_type:    Optional[str],
        date_from:   Optional[str],
        date_to:     Optional[str],
    ) -> Optional[dict]:
        # ChromaDB $gte/$lte don't support string comparisons — date filtering
        # is done client-side after retrieval via _chunk_matches.
        filters = []
        if institution:
            filters.append({"institution": {"$eq": institution}})
        if doc_type:
            filters.append({"doc_type": {"$eq": doc_type}})
        if not filters:
            return None
        return filters[0] if len(filters) == 1 else {"$and": filters}

    def _chunk_matches(
        self,
        c:           Chunk,
        institution: Optional[str],
        doc_type:    Optional[str],
        date_from:   Optional[str],
        date_to:     Optional[str],
    ) -> bool:
        if institution and c.institution != institution:
            return False
        if doc_type and c.doc_type != doc_type:
            return False
        if date_from and c.date < date_from:
            return False
        if date_to and c.date > date_to:
            return False
        return True

    # ── Main query ─────────────────────────────────────────────────────────────

    def query(
        self,
        query_text:  str,
        *,
        top_n:       int           = DEFAULT_TOP_N,
        institution: Optional[str] = None,
        doc_type:    Optional[str] = None,
        date_from:   Optional[str] = None,
        date_to:     Optional[str] = None,
    ) -> list[RetrievalResult]:
        # 1. Embed the query
        q_vec = self._vo.embed(
            [query_text], model=VOYAGE_MODEL, input_type="query"
        ).embeddings[0]

        # 2. Dense retrieval (ChromaDB)
        where = self._chroma_where(institution, doc_type, date_from, date_to)
        kwargs = dict(
            query_embeddings=[q_vec],
            n_results=TOP_K_DENSE,
            include=["metadatas", "documents"],
        )
        if where:
            kwargs["where"] = where
        dense_res = self._collection.query(**kwargs)
        # Post-filter dense results by date (ISO string comparison is lexicographic-safe)
        raw_dense_ids   = dense_res["ids"][0]
        raw_dense_metas = dense_res["metadatas"][0]
        dense_ids: list[str] = [
            cid for cid, meta in zip(raw_dense_ids, raw_dense_metas)
            if (not date_from or meta.get("date", "") >= date_from)
            and (not date_to   or meta.get("date", "") <= date_to)
        ]

        # 3. Sparse retrieval (BM25)
        tokenized = query_text.lower().split()
        scores    = self._bm25.get_scores(tokenized)

        filtered_sparse = [
            (self._chunk_ids[i], scores[i])
            for i, c in enumerate(self._chunks)
            if scores[i] > 0
            and self._chunk_matches(c, institution, doc_type, date_from, date_to)
        ]
        filtered_sparse.sort(key=lambda x: x[1], reverse=True)
        sparse_ids = [cid for cid, _ in filtered_sparse[:TOP_K_SPARSE]]

        # 4. Reciprocal Rank Fusion
        dense_rank_map:  dict[str, int]   = {cid: r + 1 for r, cid in enumerate(dense_ids)}
        sparse_rank_map: dict[str, int]   = {cid: r + 1 for r, cid in enumerate(sparse_ids)}
        all_ids = set(dense_ids) | set(sparse_ids)

        rrf_scores: dict[str, float] = {
            cid: _rrf(dense_rank_map.get(cid)) + _rrf(sparse_rank_map.get(cid))
            for cid in all_ids
        }

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        results = []
        for cid, score in ranked:
            chunk = self._chunk_map.get(cid)
            if chunk:
                results.append(RetrievalResult(
                    chunk=chunk,
                    rrf_score=score,
                    dense_rank=dense_rank_map.get(cid),
                    sparse_rank=sparse_rank_map.get(cid),
                ))

        log.debug(
            "Query '%s...' -> %d results (dense=%d sparse=%d)",
            query_text[:40], len(results), len(dense_ids), len(sparse_ids),
        )
        return results
