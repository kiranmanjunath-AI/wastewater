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
import pickle as _pickle
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_PATH = os.getenv("QDRANT_PATH", "")  # If set, use embedded local storage (no Docker)
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


def build_tfidf_sparse_vectors(chunks: list[dict], language: str):
    """
    Fit a TF-IDF vectorizer on all chunk texts and return (vectorizer, sparse_matrix).
    sparse_matrix is a scipy CSR matrix with shape (n_chunks, vocab_size).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    print(f"[embed] Building TF-IDF sparse vectors for {len(chunks)} chunks ...")
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=100_000,
        sublinear_tf=True,
        min_df=2,
        norm="l2",
    )
    X = vectorizer.fit_transform(texts)
    print(f"[embed]   Vocabulary size: {len(vectorizer.vocabulary_)}")
    return vectorizer, X


def save_vectorizer(vectorizer, language: str) -> None:
    """Persist the fitted TF-IDF vectorizer to disk for use at query time."""
    vec_path = DATA_DIR / f"tfidf_{language}.pkl"
    with open(vec_path, "wb") as f:
        _pickle.dump(vectorizer, f)
    print(f"[embed]   Saved TF-IDF vectorizer to {vec_path}")


def ensure_collection(client) -> None:
    """
    Create the Qdrant collection with named dense + BM25 sparse vectors.
    Deletes and recreates the collection if the old single-vector schema is detected.
    """
    from qdrant_client.models import (
        Distance, PayloadSchemaType, SparseIndexParams,
        SparseVectorParams, VectorParams,
    )

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        info = client.get_collection(COLLECTION_NAME)
        vectors_cfg = info.config.params.vectors
        sparse_cfg = info.config.params.sparse_vectors
        # Old single-vector schema or missing sparse → recreate
        if isinstance(vectors_cfg, VectorParams) or not sparse_cfg or "bm25" not in (sparse_cfg or {}):
            print(f"[qdrant] Schema migration needed. Deleting collection '{COLLECTION_NAME}' ...")
            client.delete_collection(COLLECTION_NAME)
        else:
            print(f"[qdrant] Collection '{COLLECTION_NAME}' already exists.")
            return

    print(f"[qdrant] Creating collection '{COLLECTION_NAME}' (dense + BM25 sparse) ...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)},
        sparse_vectors_config={
            "bm25": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
    )
    for field in ["language", "valid_from", "valid_to", "act", "section", "citation"]:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )
    print(f"[qdrant] Collection created with payload indexes.")


_CHUNK_NS = uuid.UUID("c0a1b2c3-d4e5-f6a7-b8c9-d0e1f2a3b4c5")


def _chunk_point_id(chunk: dict) -> str:
    """
    Deterministic UUID for a chunk derived from its content hash.
    Using uuid5 ensures the same chunk always maps to the same ID across
    re-ingestion runs, making upsert idempotent and eliminating ID collisions
    between EN and FR (which have different version_hash values).
    """
    return str(uuid.uuid5(_CHUNK_NS, chunk["version_hash"]))


def upsert_chunks(client, model, chunks: list[dict], sparse_matrix=None) -> int:
    """
    Embed and upsert chunks into Qdrant with named dense + optional BM25 sparse vectors.
    Point IDs are deterministic UUIDs derived from each chunk's content hash.
    Returns the number of successfully upserted points.
    """
    from qdrant_client.models import PointStruct, SparseVector

    total = len(chunks)
    upserted = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        dense_vectors = embed_passages(model, texts)

        points = []
        for i, (chunk, dense_vec) in enumerate(zip(batch, dense_vectors)):
            abs_idx = batch_start + i
            point_id = _chunk_point_id(chunk)

            vector_dict: dict = {"dense": dense_vec}
            if sparse_matrix is not None:
                row = sparse_matrix[abs_idx]
                cx = row.tocoo()
                if cx.nnz > 0:
                    vector_dict["bm25"] = SparseVector(
                        indices=cx.col.tolist(),
                        values=cx.data.tolist(),
                    )

            points.append(PointStruct(id=point_id, vector=vector_dict, payload=chunk))

        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        upserted += len(points)

        if (batch_start // BATCH_SIZE + 1) % 10 == 0 or batch_start + BATCH_SIZE >= total:
            print(f"[qdrant]   Upserted {upserted}/{total} chunks ...")

    return upserted


def purge_language_chunks(client, language: str, act: str = "ITA") -> int:
    """
    Delete all existing points for a given language/act.
    Returns the number of deleted points.

    Required because embedded Qdrant does not fully clear data on delete_collection+recreate
    with the same collection name.  Call this before upserting to prevent duplicates.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue, PointIdsList

    f = Filter(must=[
        FieldCondition(key="act", match=MatchValue(value=act)),
        FieldCondition(key="language", match=MatchValue(value=language)),
    ])

    all_ids: list[int] = []
    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=f,
            limit=1000,
            with_payload=False,
            with_vectors=False,
            offset=offset,
        )
        all_ids.extend([p.id for p in result])
        if next_offset is None:
            break
        offset = next_offset

    if all_ids:
        # Delete in batches to stay within request size limits
        for i in range(0, len(all_ids), 1000):
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=PointIdsList(points=all_ids[i : i + 1000]),
                wait=True,
            )
        print(f"[ingest] Purged {len(all_ids)} existing {language} chunks.")

    return len(all_ids)


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
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        scroll_result = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key="act", match=MatchValue(value=act)),
                FieldCondition(key="language", match=MatchValue(value=language)),
            ]),
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

    if QDRANT_PATH:
        print(f"[ingest] Using embedded Qdrant at path: {QDRANT_PATH}")
        Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
        try:
            client = QdrantClient(path=QDRANT_PATH)
            client.get_collections()
        except Exception as exc:
            raise RuntimeError(f"Cannot open embedded Qdrant at {QDRANT_PATH}. Error: {exc}") from exc
    else:
        print(f"[ingest] Connecting to Qdrant at {QDRANT_URL} ...")
        try:
            client = QdrantClient(url=QDRANT_URL, timeout=30)
            client.get_collections()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot connect to Qdrant at {QDRANT_URL}. "
                f"Set QDRANT_PATH=./data/qdrant_storage to use embedded mode (no Docker). Error: {exc}"
            ) from exc

    ensure_collection(client)

    # Purge stale points for this language before upserting (handles embedded Qdrant
    # same-name recreate bug and prevents duplicates on re-ingestion).
    purge_language_chunks(client, language=language, act=act)

    model = load_embedding_model()

    # Build TF-IDF sparse vectors and persist vectorizer for query-time use
    vectorizer, sparse_matrix = build_tfidf_sparse_vectors(chunks, language)
    save_vectorizer(vectorizer, language)

    new_hashes = {c["version_hash"] for c in chunks}
    added, removed, unchanged = compute_delta(client, new_hashes, language, act)

    print(f"[ingest] Delta: +{added} added, -{removed} removed, {unchanged} unchanged")

    upserted = upsert_chunks(client, model, chunks, sparse_matrix=sparse_matrix)
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
    parser.add_argument("--valid-from", default=None, help="ISO date for valid_from (default: from XML)")
    parser.add_argument("--amending-act", default="")
    parser.add_argument("--consolidated-as-of", default=None, help="ISO date (default: from XML)")
    args = parser.parse_args()

    xml_path = args.xml or (DATA_DIR / f"ita_{args.language}.xml")
    consolidated_as_of_from_xml = ""

    if not xml_path.exists():
        print(f"[load] XML not found at {xml_path}. Fetching from Justice Laws ...")
        from ingestion.fetch_ita import fetch_all
        try:
            results = fetch_all([args.language])
            xml_path = results[args.language]["path"]
            xml_hash = results[args.language]["version_hash"]
            source_url = results[args.language]["url"]
            consolidated_as_of_from_xml = results[args.language].get("consolidated_as_of", "")
        except RuntimeError as exc:
            print(f"[load] FATAL: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        from ingestion.fetch_ita import compute_hash, extract_consolidation_date
        raw = xml_path.read_bytes()
        xml_hash = compute_hash(raw)
        source_url = ""
        consolidated_as_of_from_xml = extract_consolidation_date(raw)

    # Use explicit arg if given, fall back to XML date, then today
    from datetime import date as _date
    consolidated_as_of = (
        args.consolidated_as_of
        or consolidated_as_of_from_xml
        or str(_date.today())
    )
    valid_from = args.valid_from or consolidated_as_of

    print(f"[load] Parsing XML: {xml_path} ...")
    from ingestion.chunk_statute import parse_ita_xml
    chunks = parse_ita_xml(
        xml_path=xml_path,
        language=args.language,
        valid_from=valid_from,
        amending_act=args.amending_act,
    )

    summary = run_ingestion(
        chunks=chunks,
        language=args.language,
        act="ITA",
        consolidated_as_of=consolidated_as_of,
        amending_act=args.amending_act,
        source_url=source_url,
        xml_version_hash=xml_hash,
    )

    print("\n=== Ingestion complete ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
