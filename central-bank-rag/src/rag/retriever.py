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
TOP_K_DENSE   = 50   # candidates pulled from ChromaDB
TOP_K_SPARSE  = 30   # candidates pulled from BM25
BM25_PER_DOC  = 2    # max BM25 chunks per source document (prevents flooding)
RRF_K         = 60   # RRF constant (standard value)
DEFAULT_TOP_N = 6    # final chunks passed to the generator

# Primary announcement documents get a score boost so they surface
# over longer contextual documents (minutes, deliberations) when both
# are semantically close to the query.
ANNOUNCEMENT_TYPES = frozenset({"FOMC Statement", "Rate Decision Statement"})
ANNOUNCEMENT_BOOST = 2.0


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

        # 3. Sparse retrieval (BM25) — capped at BM25_PER_DOC chunks per
        #    source document so that verbose docs (e.g. Beige Books) cannot
        #    monopolise the entire sparse pool.
        tokenized = query_text.lower().split()
        scores    = self._bm25.get_scores(tokenized)

        filtered_sparse = [
            (self._chunk_ids[i], scores[i], self._chunks[i])
            for i, c in enumerate(self._chunks)
            if scores[i] > 0
            and self._chunk_matches(c, institution, doc_type, date_from, date_to)
        ]
        filtered_sparse.sort(key=lambda x: x[1], reverse=True)

        from collections import defaultdict
        doc_chunk_count: dict[str, int] = defaultdict(int)
        sparse_ids: list[str] = []
        for cid, sc, c in filtered_sparse:
            if doc_chunk_count[c.doc_id] < BM25_PER_DOC:
                sparse_ids.append(cid)
                doc_chunk_count[c.doc_id] += 1
            if len(sparse_ids) >= TOP_K_SPARSE:
                break

        # 4. Reciprocal Rank Fusion with boost for primary announcement types
        dense_rank_map:  dict[str, int]   = {cid: r + 1 for r, cid in enumerate(dense_ids)}
        sparse_rank_map: dict[str, int]   = {cid: r + 1 for r, cid in enumerate(sparse_ids)}
        all_ids = set(dense_ids) | set(sparse_ids)

        rrf_scores: dict[str, float] = {}
        for cid in all_ids:
            score = _rrf(dense_rank_map.get(cid)) + _rrf(sparse_rank_map.get(cid))
            chunk = self._chunk_map.get(cid)
            if chunk and chunk.doc_type in ANNOUNCEMENT_TYPES:
                score *= ANNOUNCEMENT_BOOST
            rrf_scores[cid] = score

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Ensure each primary announcement type is represented — if missing from
        # top_n, inject the most recent qualifying chunk from the dense candidate
        # pool, replacing the lowest-scoring result.
        if not doc_type:  # skip when caller already filtered to one doc_type
            ranked_set = {cid for cid, _ in ranked}
            for a_type in ANNOUNCEMENT_TYPES:
                if any(
                    self._chunk_map.get(cid) and self._chunk_map[cid].doc_type == a_type
                    for cid in ranked_set
                ):
                    continue
                # Find the most recent chunk of this type in the dense pool that
                # also passes the active filters, sorted by date descending.
                # Always pick the most recent chunk of this type from the
                # full index (not just the dense pool) so the latest
                # announcement is always visible regardless of query wording.
                candidates = [
                    (c.date, cid)
                    for cid, c in self._chunk_map.items()
                    if cid not in ranked_set
                    and c.doc_type == a_type
                    and self._chunk_matches(
                        c, institution, doc_type, date_from, date_to
                    )
                ]
                if candidates:
                    _, best_cid = max(candidates)
                    ranked[-1] = (best_cid, rrf_scores.get(best_cid, 0.0))
                    ranked_set.add(best_cid)

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
