"""
Download all central bank documents.

Usage:
    python scripts/download_docs.py              # download everything
    python scripts/download_docs.py --fed-only   # Federal Reserve only
    python scripts/download_docs.py --boc-only   # Bank of Canada only
    python scripts/download_docs.py --dry-run    # show what would be downloaded

Documents are saved under data/raw/{fed,boc}/.
Progress is persisted to data/metadata.json after every scraper run,
so it is safe to Ctrl-C and resume at any time.
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

# Make sure the project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest.base import load_metadata, save_metadata, BASE_DIR
from src.ingest import fed_scraper, boc_scraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def summarise(meta: dict) -> None:
    """Print a breakdown of what's in the metadata store."""
    counts: Counter = Counter()
    for rec in meta.values():
        counts[(rec["institution"], rec["doc_type"])] += 1

    print("\n-- Document inventory -----------------------------------------")
    for (inst, dtype), n in sorted(counts.items()):
        print(f"  {inst:20s}  {dtype:40s}  {n:3d}")
    print(f"  {'TOTAL':64s}  {sum(counts.values()):3d}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download central bank documents")
    parser.add_argument("--fed-only",  action="store_true", help="Only download Fed documents")
    parser.add_argument("--boc-only",  action="store_true", help="Only download BoC documents")
    parser.add_argument("--dry-run",   action="store_true", help="Show what would be fetched without downloading")
    parser.add_argument("--from-date", default="2025-01-01", help="Earliest date to collect (YYYY-MM-DD)")
    parser.add_argument("--to-date",   default=None,         help="Latest date to collect (YYYY-MM-DD, default: today)")
    args = parser.parse_args()

    # Override date range if requested
    if args.from_date or args.to_date:
        import src.ingest.base as base_mod
        from datetime import date, datetime
        if args.from_date:
            base_mod.DATE_FROM = datetime.strptime(args.from_date, "%Y-%m-%d").date()
        if args.to_date:
            base_mod.DATE_TO = datetime.strptime(args.to_date, "%Y-%m-%d").date()

    if args.dry_run:
        log.info("DRY RUN — no files will be downloaded")
        # Patch download functions to no-ops
        import src.ingest.base as base_mod
        base_mod.download_pdf      = lambda url, dest, **kw: log.info("  [dry-run] PDF  %s", dest.name) or False
        base_mod.save_html_content = lambda url, dest, **kw: log.info("  [dry-run] HTML %s", dest.name) or ""

    meta = load_metadata()
    log.info("Loaded %d existing records from metadata.json", len(meta))

    all_new = []

    if not args.boc_only:
        new_recs = fed_scraper.run(meta)
        save_metadata(meta)   # checkpoint after Fed
        all_new.extend(new_recs)
        log.info("Fed: %d new documents collected", len(new_recs))

    if not args.fed_only:
        new_recs = boc_scraper.run(meta)
        save_metadata(meta)   # checkpoint after BoC
        all_new.extend(new_recs)
        log.info("BoC: %d new documents collected", len(new_recs))

    log.info("Done. %d new documents total.", len(all_new))
    summarise(meta)

    # Verify all files actually exist on disk
    missing = [
        rec for rec in meta.values()
        if not (BASE_DIR / rec["local_path"]).exists()
    ]
    if missing:
        log.warning("%d files listed in metadata but missing from disk:", len(missing))
        for m in missing:
            log.warning("  %s  %s", m["doc_id"], m["local_path"])
    else:
        log.info("All %d files verified on disk ✓", len(meta))


if __name__ == "__main__":
    main()
