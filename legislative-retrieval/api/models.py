"""
models.py
Pydantic request and response models for the Legislative Retrieval API.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# /retrieve
# ---------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Free-text search query")
    as_of: Optional[str] = Field(
        default=None,
        description="ISO date (YYYY-MM-DD). Only return provisions valid on this date. Defaults to today.",
        examples=["2024-01-01"],
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    act: Optional[str] = Field(default="ITA", description="Legislation identifier (currently only 'ITA')")
    language: str = Field(default="en", pattern="^(en|fr)$", description="Result language: 'en' or 'fr'")

    @field_validator("as_of", mode="before")
    @classmethod
    def validate_as_of(cls, v):
        if v is None:
            return None
        # Accept both date strings and date objects
        if isinstance(v, date):
            return v.isoformat()
        try:
            date.fromisoformat(str(v))
        except ValueError:
            raise ValueError(f"as_of must be an ISO date string (YYYY-MM-DD), got: {v!r}")
        return str(v)


class RetrievedChunk(BaseModel):
    citation: str = Field(..., description="ITA citation, e.g. 'ITA s.20(1)(c)'")
    text: str = Field(..., description="Full text of the retrieved chunk (includes context prefix)")
    score: float = Field(..., description="Cosine similarity score (0-1)")
    language: str
    valid_from: str
    valid_to: Optional[str] = None
    amending_act: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    paragraph: Optional[str] = None
    part: Optional[str] = None
    division: Optional[str] = None


class RetrieveResponse(BaseModel):
    query: str
    as_of: str
    consolidated_as_of: Optional[str] = Field(
        default=None,
        description="Date of the most recently ingested version in the collection",
    )
    results: list[RetrievedChunk]
    retrieved_at: str = Field(..., description="ISO timestamp of the retrieval")


# ---------------------------------------------------------------------------
# /versions
# ---------------------------------------------------------------------------

class VersionChunkDelta(BaseModel):
    added: int
    removed: int
    unchanged: int


class VersionRecord(BaseModel):
    version_id: str
    act: str
    language: str
    consolidated_as_of: Optional[str] = None
    ingested_at: str
    amending_act: Optional[str] = None
    chunk_count: int
    chunk_delta: VersionChunkDelta
    is_current: bool


class VersionsResponse(BaseModel):
    versions: list[VersionRecord]


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' or 'degraded'")
    qdrant: str = Field(..., description="Qdrant connection status")
    collection: str = Field(..., description="Collection status")
    chunk_count: int = Field(..., description="Total chunks in the collection")
    latest_version: Optional[str] = Field(default=None, description="Most recent consolidated_as_of date")


# ---------------------------------------------------------------------------
# /diff (stub)
# ---------------------------------------------------------------------------

class DiffRequest(BaseModel):
    version_id_a: str
    version_id_b: str
    act: str = "ITA"
    language: str = "en"


class DiffResponse(BaseModel):
    message: str = "Diff endpoint not yet implemented."
    version_id_a: str
    version_id_b: str


# ---------------------------------------------------------------------------
# /usage (stub)
# ---------------------------------------------------------------------------

class UsageResponse(BaseModel):
    message: str = "Usage tracking not yet implemented."
