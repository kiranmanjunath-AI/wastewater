"""
Build the embedding index from downloaded documents.

Usage:
    python scripts/build_index.py              # incremental update
    python scripts/build_index.py --rebuild    # drop existing and rebuild

Requires VOYAGE_API_KEY in the environment or in a .env file at the project root.
Run download_docs.py first to populate data/raw/ and data/metadata.json.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env before importing modules that read env vars
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.ingest.base import load_metadata
from src.embed.chunker import chunk_all
from src.embed.indexer import build_index, CHROMA_DIR, BM25_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build central-bank RAG index")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Delete existing ChromaDB store and BM25 file, then rebuild from scratch",
    )
    args = parser.parse_args()

    if args.rebuild:
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
            log.info("Removed existing ChromaDB store")
        if BM25_FILE.exists():
            BM25_FILE.unlink()
            log.info("Removed existing BM25 index")

    meta = load_metadata()
    if not meta:
        log.error("metadata.json is empty — run python scripts/download_docs.py first")
        sys.exit(1)
    log.info("Loaded %d documents from metadata.json", len(meta))

    log.info("Extracting and chunking text from %d documents…", len(meta))
    chunks = chunk_all(meta)
    if not chunks:
        log.error("No text extracted — check that data/raw/ files exist")
        sys.exit(1)

    build_index(chunks, incremental=not args.rebuild)


if __name__ == "__main__":
    main()
