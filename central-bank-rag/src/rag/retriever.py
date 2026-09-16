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
DEFAULT_TOP_N = 10   # final chunks passed to the generator

# Primary announcement documents get a score boost so they surface
# over longer contextual documents (minutes, deliberations) when both
# are semantically close to the query.
ANNOUNCEMENT_TYPES = frozenset({"FOMC Statement", "Rate Decision Statement"})
ANNOUNCEMENT_BOOST = 2.0

# BM25 query expansion: central-bank decision documents use different
# vocabulary than everyday English.  "Rate cuts" in a query should also
# match FOMC Statements that say "lower the target range for the federal
# funds rate" and BoC statements that say "reduce its target for the
# overnight rate".  Expanding the token list bridges this lexical gap so
# cut/hike announcements enter the sparse pool and benefit from the boost.
_EXPAND_CUT  = {"cut", "cuts", "cutting", "easing", "ease", "eased", "lower", "lowered"}
_EXPAND_HIKE = {"hike", "hikes", "hiking", "tighten", "tightening", "raise", "raised"}

_CUT_SYNONYMS  = ["lower", "reduce", "target", "range", "easing", "basis", "points"]
_HIKE_SYNONYMS = ["raise", "higher", "restrictive", "tighten", "increase"]


def _expand_tokens(tokens: list[str]) -> list[str]:
    """Add central-bank vocabulary synonyms so BM25 finds announcement docs."""
    s = set(tokens)
    extra: list[str] = []
    if s & _EXPAND_CUT:
        extra += [t for t in _CUT_SYNONYMS  if t not in s]
    if s & _EXPAND_HIKE:
        extra += [t for t in _HIKE_SYNONYMS if t not in s]
    return tokens + extra if extra else tokens


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
        #    monopolise the entire sparse pool.  Query tokens are expanded with
        #    central-bank vocabulary synonyms so that phrasing like "rate cuts"
        #    also matches decision documents that say "lower the target range".
        tokenized = _expand_tokens(query_text.lower().split())
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

        # Deduplicate: keep only the highest-scoring chunk per source document.
        # RRF may surface two chunks from the same document (e.g. BM25 matches
        # two passages from the same Rate Decision Statement), wasting a slot.
        seen_docs: set[str] = set()
        deduped: list[tuple[str, float]] = []
        for cid, score in ranked:
            chunk = self._chunk_map.get(cid)
            if not chunk:
                continue
            if chunk.doc_id not in seen_docs:
                deduped.append((cid, score))
                seen_docs.add(chunk.doc_id)
        ranked = deduped[:top_n]

        # Anchor injection: ensure each primary announcement type appears in
        # the results even when those brief documents rank far below the dense
        # top-K on embedding similarity alone.
        #
        # Strategy: sample four temporal checkpoints from the *unique-document*
        # date list (not raw chunks) so that one injected chunk cannot crowd out
        # another from the same meeting.  For a corpus spanning Jan 2025–Sep 2026
        # with 13 unique FOMC Statement dates the checkpoints land at:
        #   n//2 - 1        → index 5  → Sep 2025  (cut #1, −25bp to 4–4¼%)
        #   n//2            → index 6  → Oct 2025  (cut #2, −25bp to 3¾–4%)
        #   round(0.6(n-1)) → index 7  → Dec 2025  (cut #3, −25bp to 3½–3¾%)
        #   n-1             → index 12 → Jul 2026   (current hawkish hold)
        # covering all three Fed cuts + current stance.  The same formula
        # applied to the 14 BoC Rate Decision Statement dates yields Sep, Oct,
        # Dec 2025, Sep 2026 — all three late-2025 BoC cuts + current stance.
        #
        # Guard: skip injection only when the most-recently dated document of
        # this type is already organically covered (good temporal coverage).
        # The old "any organic → skip" guard was too conservative: early-period
        # announcements in organic results suppressed injection of late-period
        # ones entirely (e.g. Jan/Mar 2025 BoC → no mid-late 2025 BoC injected).
        #
        # Injected chunks replace slots from the *end* of the ranked list so
        # organic top results are not displaced.
        if not doc_type:  # skip when caller already filtered to one doc_type
            ranked_set = {cid for cid, _ in ranked}
            inject_slot = len(ranked) - 1

            for a_type in ANNOUNCEMENT_TYPES:
                if inject_slot < 0:
                    break

                sub_where: dict = {"doc_type": {"$eq": a_type}}
                if institution:
                    sub_where = {"$and": [sub_where, {"institution": {"$eq": institution}}]}

                all_type = self._collection.get(where=sub_where, include=["metadatas"])

                # Find which source-document IDs of this type are already
                # organically represented so we never inject a second chunk
                # from the same meeting that is already in results.
                organic_doc_ids_of_type = {
                    self._chunk_map[cid].doc_id
                    for cid in ranked_set
                    if self._chunk_map.get(cid)
                    and self._chunk_map[cid].doc_type == a_type
                }

                # Build a map: document-date → first valid chunk id, skipping
                # entire documents that are already organically covered.
                date_to_cid: dict[str, str] = {}
                for cid, meta in zip(all_type["ids"], all_type["metadatas"]):
                    if cid in ranked_set or cid not in self._chunk_map:
                        continue
                    c = self._chunk_map[cid]
                    if c.doc_id in organic_doc_ids_of_type:
                        continue  # another chunk of this doc is already organic
                    if not self._chunk_matches(c, institution, doc_type, date_from, date_to):
                        continue
                    if c.date not in date_to_cid:
                        date_to_cid[c.date] = cid

                if not date_to_cid:
                    continue

                # Skip injection only if the most-recent available date is
                # already organically covered — that implies good temporal reach.
                if organic_doc_ids_of_type:
                    organic_dates = {
                        self._chunk_map[cid].date
                        for cid in ranked_set
                        if self._chunk_map.get(cid)
                        and self._chunk_map[cid].doc_type == a_type
                    }
                    most_recent_overall = max(set(date_to_cid) | organic_dates)
                    if most_recent_overall in organic_dates:
                        continue  # most-recent meeting already represented

                unique_dates = sorted(date_to_cid.keys())
                n = len(unique_dates)

                # Four temporal checkpoints: pre-mid, mid, three-fifths, most-recent.
                # The 0.6 fractile shifts the third checkpoint one slot earlier
                # vs 2/3 so it lands on the Dec 2025 cut rather than the Jan 2026
                # hold for both the FOMC (n=13) and BoC (n=12 after organic removed).
                if n <= 4:
                    sample_indices: list[int] = list(range(n))
                else:
                    sample_indices = sorted({
                        max(0, n // 2 - 1),
                        n // 2,
                        int(round(0.6 * (n - 1))),
                        n - 1,
                    })

                inject_cids = [date_to_cid[unique_dates[i]] for i in sample_indices]

                for cid in inject_cids:
                    if inject_slot < 0:
                        break
                    ranked[inject_slot] = (cid, rrf_scores.get(cid, 0.0))
                    ranked_set.add(cid)
                    inject_slot -= 1

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
