"""
main.py
FastAPI application for the Legislative Retrieval Platform.

Start with:
    uvicorn api.main:app --reload
from the legislative-retrieval/ directory.

Endpoints:
  POST /retrieve    — semantic search over ITA chunks
  GET  /versions    — list all ingested versions
  GET  /health      — liveness + readiness check
  POST /diff        — (stub) compare two ingested versions
  GET  /usage       — (stub) usage statistics
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from api.models import (
    DiffRequest,
    DiffResponse,
    HealthResponse,
    RetrieveRequest,
    RetrieveResponse,
    RetrievedChunk,
    UsageResponse,
    VersionChunkDelta,
    VersionRecord,
    VersionsResponse,
)

app = FastAPI(
    title="Legislative Retrieval API",
    description=(
        "Retrieval-only REST API surfacing exact, cited passages from the Canadian Income Tax Act (ITA). "
        "No LLM inference — pure semantic search. Embed in your own LLM workflows."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# POST /retrieve
# ---------------------------------------------------------------------------

@app.post("/retrieve", response_model=RetrieveResponse, summary="Semantic search over ITA provisions")
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    """
    Search for ITA provisions matching the query.

    - Embeds the query using multilingual-e5-small with the required 'query: ' prefix.
    - Filters by language and temporal validity window (valid on `as_of` date).
    - Returns top_k ranked results with full citation and passage text.

    The `text` field in each result already includes the section heading context prefix,
    so it is interpretable in isolation — suitable for direct insertion into LLM prompts.
    """
    from api.retriever import search, get_consolidated_as_of

    as_of = request.as_of or date.today().isoformat()

    try:
        raw_results = search(
            query=request.query,
            top_k=request.top_k,
            language=request.language,
            as_of=as_of,
            act=request.act,
        )
    except Exception as exc:
        # Provide a clear error rather than a 500 with a traceback
        raise HTTPException(
            status_code=503,
            detail=(
                f"Search failed. Ensure Qdrant is running and the collection has been populated. "
                f"Error: {exc}"
            ),
        )

    chunks = [RetrievedChunk(**r) for r in raw_results]

    consolidated_as_of = get_consolidated_as_of(act=request.act or "ITA", language=request.language)

    return RetrieveResponse(
        query=request.query,
        as_of=as_of,
        consolidated_as_of=consolidated_as_of,
        results=chunks,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# GET /versions
# ---------------------------------------------------------------------------

@app.get("/versions", response_model=VersionsResponse, summary="List all ingested versions")
async def list_versions(
    act: Optional[str] = Query(default=None, description="Filter by act (e.g. 'ITA')"),
    language: Optional[str] = Query(default=None, description="Filter by language ('en' or 'fr')"),
) -> VersionsResponse:
    """
    Return all ingestion runs recorded in the version index, newest first.
    """
    from db.version_index import get_all_versions, get_latest_version

    try:
        all_versions = get_all_versions(act=act, language=language)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not read version index. Error: {exc}",
        )

    if not all_versions:
        return VersionsResponse(versions=[])

    # The most recently ingested version for each (act, language) pair is "current"
    # We build a set of the latest version_id per combination
    latest_by_key: dict[tuple[str, str], str] = {}
    for v in all_versions:
        key = (v["act"], v["language"])
        if key not in latest_by_key:
            latest_by_key[key] = v["version_id"]

    records = []
    for v in all_versions:
        key = (v["act"], v["language"])
        delta = VersionChunkDelta(
            added=v.get("chunks_added") or 0,
            removed=v.get("chunks_removed") or 0,
            unchanged=(v.get("chunk_count") or 0) - (v.get("chunks_added") or 0),
        )
        records.append(
            VersionRecord(
                version_id=v["version_id"],
                act=v["act"],
                language=v["language"],
                consolidated_as_of=v.get("consolidated_as_of"),
                ingested_at=v["ingested_at"],
                amending_act=v.get("amending_act"),
                chunk_count=v.get("chunk_count") or 0,
                chunk_delta=delta,
                is_current=(latest_by_key.get(key) == v["version_id"]),
            )
        )

    return VersionsResponse(versions=records)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """
    Check the health of the service.
    - `status` is 'ok' when Qdrant is reachable and the collection exists.
    - `status` is 'degraded' if Qdrant is unreachable or the collection is missing/empty.
    """
    from api.retriever import health_check

    info = health_check()

    qdrant_ok = info["qdrant"] == "ok"
    collection_ok = info["collection"] == "ok"
    chunk_count = info["chunk_count"]

    if qdrant_ok and collection_ok and chunk_count > 0:
        status = "ok"
    else:
        status = "degraded"

    return HealthResponse(
        status=status,
        qdrant=info["qdrant"],
        collection=info["collection"],
        chunk_count=chunk_count,
        latest_version=info.get("latest_version"),
    )


# ---------------------------------------------------------------------------
# POST /diff  (stub)
# ---------------------------------------------------------------------------

@app.post("/diff", response_model=DiffResponse, summary="[Stub] Compare two ingested versions")
async def diff(request: DiffRequest) -> DiffResponse:
    """
    Stub endpoint. Will return a structural diff between two ingested versions,
    listing added/removed/changed provisions with full citation context.
    Not yet implemented.
    """
    return DiffResponse(
        message="Diff endpoint not yet implemented.",
        version_id_a=request.version_id_a,
        version_id_b=request.version_id_b,
    )


# ---------------------------------------------------------------------------
# GET /usage  (stub)
# ---------------------------------------------------------------------------

@app.get("/usage", response_model=UsageResponse, summary="[Stub] Usage statistics")
async def usage() -> UsageResponse:
    """
    Stub endpoint. Will return per-endpoint query counts and latency percentiles.
    Not yet implemented.
    """
    return UsageResponse(message="Usage tracking not yet implemented.")
