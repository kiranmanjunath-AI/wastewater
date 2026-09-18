"""
version_index.py
SQLite-backed version tracking for legislative ingestion runs.

Table: ingestion_runs
  version_id        TEXT PRIMARY KEY (UUID)
  act               TEXT
  language          TEXT
  consolidated_as_of TEXT (date ISO string)
  ingested_at       TEXT (ISO timestamp)
  source_url        TEXT
  version_hash      TEXT
  chunk_count       INTEGER
  chunks_added      INTEGER
  chunks_removed    INTEGER
  amending_act      TEXT
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(os.getenv("DB_PATH", "./data/version_index.db"))

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ingestion_runs (
    version_id          TEXT PRIMARY KEY,
    act                 TEXT NOT NULL,
    language            TEXT NOT NULL,
    consolidated_as_of  TEXT,
    ingested_at         TEXT NOT NULL,
    source_url          TEXT,
    version_hash        TEXT,
    chunk_count         INTEGER,
    chunks_added        INTEGER,
    chunks_removed      INTEGER,
    amending_act        TEXT
);
"""


def _get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open (and if necessary initialise) the SQLite database."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()
    return conn


def record_ingestion_run(
    version_id: str,
    act: str,
    language: str,
    consolidated_as_of: str,
    ingested_at: str,
    source_url: str,
    version_hash: str,
    chunk_count: int,
    chunks_added: int,
    chunks_removed: int,
    amending_act: str,
    db_path: Optional[Path] = None,
) -> None:
    """Insert a new ingestion run record."""
    conn = _get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO ingestion_runs
              (version_id, act, language, consolidated_as_of, ingested_at,
               source_url, version_hash, chunk_count, chunks_added, chunks_removed, amending_act)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                act,
                language,
                consolidated_as_of,
                ingested_at,
                source_url,
                version_hash,
                chunk_count,
                chunks_added,
                chunks_removed,
                amending_act,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_all_versions(
    act: Optional[str] = None,
    language: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> list[dict]:
    """
    Return all ingestion run records, newest first.
    Optionally filter by act and/or language.
    """
    conn = _get_connection(db_path)
    try:
        where_clauses = []
        params = []
        if act:
            where_clauses.append("act = ?")
            params.append(act)
        if language:
            where_clauses.append("language = ?")
            params.append(language)
        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        sql = f"SELECT * FROM ingestion_runs {where} ORDER BY ingested_at DESC"
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_latest_version(
    act: str = "ITA",
    language: str = "en",
    db_path: Optional[Path] = None,
) -> Optional[dict]:
    """Return the most recently ingested version for a given act/language, or None."""
    versions = get_all_versions(act=act, language=language, db_path=db_path)
    return versions[0] if versions else None


def version_exists(
    version_hash: str,
    act: str = "ITA",
    language: str = "en",
    db_path: Optional[Path] = None,
) -> bool:
    """Return True if this version_hash has already been ingested."""
    conn = _get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM ingestion_runs WHERE version_hash = ? AND act = ? AND language = ?",
            (version_hash, act, language),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def list_version_ids(db_path: Optional[Path] = None) -> list[str]:
    """Return all version IDs, newest first."""
    conn = _get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT version_id FROM ingestion_runs ORDER BY ingested_at DESC"
        ).fetchall()
        return [row["version_id"] for row in rows]
    finally:
        conn.close()
