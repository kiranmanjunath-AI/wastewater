"""
load_qdrant.py
Embed chunks and load them into the Qdrant vector database.

Embedding model: intfloat/multilingual-e5-small (384-dim)
  - Passage prefix: "passage: " + text  (at index time)
  - Query prefix:   "query: " + text    (at query time)

Also updates the SQLite version index on successful load.
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "legislation_chunks")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DB_PATH = Path(os.getenv("DB_PATH", "./data/version_index.db"))

VECTOR_SIZE = 384
BATCH_SIZE = 64  # number of chunks per Qdrant upsert batch


def load_embedding_model():
    """Load and return the multilingual-e5-small model."""
    from sentence_transformers import SentenceTransformer

    print("[embed] Loading intfloat/multilingual-e5-small ...")
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    print("[embed]   Model loaded.")
    return model


def embed_passages(model, texts: list[str]) -> list[list[float]]:
    """Embed a list of passages with the required 'passage: ' prefix."""
    prefixed = [f"passage: {t}" for t in texts]
    embeddings = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def ensure_collection(client) -> None:
    """Create the Qdrant collection if it does not already exist."""
    from qdrant_client.models import Distance, VectorParams

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"[qdrant] Collection '{COLLECTION_NAME}' already exists.")
        return

    print(f"[qdrant] Creating collection '{COLLECTION_NAME}' (size={VECTOR_SIZE}, cosine) ...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    # Create payload indexes for filterable fields
    from qdrant_client.models import PayloadSchemaType
    for field in ["language", "valid_from", "valid_to", "act", "section", "citation"]:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )
    print(f"[qdrant] Collection created with payload indexes.")


def upsert_chunks(client, model, chunks: list[dict], start_id: int = 0) -> int:
    """
    Embed and upsert chunks into Qdrant.
    Returns the number of successfully upserted points.
    """
    from qdrant_client.models import PointStruct

    total = len(chunks)
    upserted = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        vectors = embed_passages(model, texts)

        points = []
        for i, (chunk, vector) in enumerate(zip(batch, vectors)):
            point_id = start_id + batch_start + i
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=chunk,
                )
            )

        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        upserted += len(points)

        if (batch_start // BATCH_SIZE + 1) % 10 == 0 or batch_start + BATCH_SIZE >= total:
            print(f"[qdrant]   Upserted {upserted}/{total} chunks ...")

    return upserted


def compute_delta(
    client,
    new_hashes: set[str],
    language: str,
    act: str = "ITA",
) -> tuple[int, int, int]:
    """
    Compute delta against existing collection for the same language.
    Returns (added, removed, unchanged).
    Simplified approach: compare version_hash sets.
    """
    existing_hashes: set[str] = set()
    try:
        scroll_result = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter={
                "must": [
                    {"key": "act", "match": {"value": act}},
                    {"key": "language", "match": {"value": language}},
                ]
            },
            limit=10000,
            with_payload=["version_hash"],
            with_vectors=False,
        )
        for point in scroll_result[0]:
            if point.payload and "version_hash" in point.payload:
                existing_hashes.add(point.payload["version_hash"])
    except Exception as exc:
        print(f"[qdrant] WARNING: Could not scroll existing hashes: {exc}")

    added = len(new_hashes - existing_hashes)
    removed = len(existing_hashes - new_hashes)
    unchanged = len(new_hashes & existing_hashes)
    return added, removed, unchanged


def run_ingestion(
    chunks: list[dict],
    language: str = "en",
    act: str = "ITA",
    consolidated_as_of: str = "",
    amending_act: str = "",
    source_url: str = "",
    xml_version_hash: str = "",
) -> dict:
    """
    Full ingestion pipeline: connect to Qdrant, embed, upsert, record version.

    Returns a summary dict with ingestion run metadata.
    """
    from qdrant_client import QdrantClient

    print(f"[ingest] Connecting to Qdrant at {QDRANT_URL} ...")
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=30)
        # Test connection
        client.get_collections()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot connect to Qdrant at {QDRANT_URL}. "
            f"Is the docker-compose stack running? Error: {exc}"
        ) from exc

    ensure_collection(client)
    model = load_embedding_model()

    new_hashes = {c["version_hash"] for c in chunks}
    added, removed, unchanged = compute_delta(client, new_hashes, language, act)

    print(f"[ingest] Delta: +{added} added, -{removed} removed, {unchanged} unchanged")

    # Determine current max ID to avoid collisions
    try:
        count_result = client.count(collection_name=COLLECTION_NAME)
        start_id = count_result.count
    except Exception:
        start_id = 0

    upserted = upsert_chunks(client, model, chunks, start_id=start_id)
    print(f"[ingest] Upserted {upserted} chunks total.")

    version_id = str(uuid.uuid4())
    ingested_at = datetime.now(timezone.utc).isoformat()

    # Record to SQLite
    try:
        from db.version_index import record_ingestion_run
        record_ingestion_run(
            version_id=version_id,
            act=act,
            language=language,
            consolidated_as_of=consolidated_as_of,
            ingested_at=ingested_at,
            source_url=source_url,
            version_hash=xml_version_hash,
            chunk_count=len(chunks),
            chunks_added=added,
            chunks_removed=removed,
            amending_act=amending_act,
        )
        print(f"[ingest] Version recorded: {version_id}")
    except Exception as exc:
        print(f"[ingest] WARNING: Could not record version to SQLite: {exc}", file=sys.stderr)

    return {
        "version_id": version_id,
        "act": act,
        "language": language,
        "consolidated_as_of": consolidated_as_of,
        "ingested_at": ingested_at,
        "chunk_count": len(chunks),
        "chunks_added": added,
        "chunks_removed": removed,
        "chunks_unchanged": unchanged,
        "amending_act": amending_act,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed ITA chunks and load into Qdrant")
    parser.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="Path to ITA XML (auto-fetches if not provided)",
    )
    parser.add_argument("--language", default="en", choices=["en", "fr"])
    parser.add_argument("--valid-from", default="2024-01-01")
    parser.add_argument("--amending-act", default="S.C. 2024, c. 15")
    parser.add_argument("--consolidated-as-of", default="2024-01-01")
    args = parser.parse_args()

    xml_path = args.xml or (DATA_DIR / f"ita_{args.language}.xml")

    if not xml_path.exists():
        print(f"[load] XML not found at {xml_path}. Fetching from Justice Laws ...")
        from ingestion.fetch_ita import fetch_all
        try:
            results = fetch_all([args.language])
            xml_path = results[args.language]["path"]
            xml_hash = results[args.language]["version_hash"]
            source_url = results[args.language]["url"]
        except RuntimeError as exc:
            print(f"[load] FATAL: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        from ingestion.fetch_ita import compute_hash
        xml_hash = compute_hash(xml_path.read_bytes())
        source_url = ""

    print(f"[load] Parsing XML: {xml_path} ...")
    from ingestion.chunk_statute import parse_ita_xml
    chunks = parse_ita_xml(
        xml_path=xml_path,
        language=args.language,
        valid_from=args.valid_from,
        amending_act=args.amending_act,
    )

    summary = run_ingestion(
        chunks=chunks,
        language=args.language,
        act="ITA",
        consolidated_as_of=args.consolidated_as_of,
        amending_act=args.amending_act,
        source_url=source_url,
        xml_version_hash=xml_hash,
    )

    print("\n=== Ingestion complete ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
