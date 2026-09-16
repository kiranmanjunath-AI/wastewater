"""
Federal Reserve document scraper.

Collects (all as PDF):
  - FOMC Statements
  - FOMC Minutes
  - Summary of Economic Projections (SEP / Projection Materials)
  - Beige Book
  - Monetary Policy Report (Semi-Annual)

All downloads are idempotent: existing files are never re-fetched.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from .base import (
    BASE_DIR, RAW_DIR,
    DocumentRecord,
    fetch_page, download_pdf,
    extract_date_from_href,
    is_in_range, parse_iso,
)

log = logging.getLogger(__name__)

FED_BASE = "https://www.federalreserve.gov"
FED_DIR = RAW_DIR / "fed"

CALENDAR_URL      = f"{FED_BASE}/monetarypolicy/fomccalendars.htm"
BEIGE_BOOK_URL    = f"{FED_BASE}/monetarypolicy/beige-book-default.htm"
BEIGE_BOOK_ARCHIVE = f"{FED_BASE}/monetarypolicy/beige-book-archive.htm"
MPR_URL           = f"{FED_BASE}/monetarypolicy/mpr_default.htm"


def _abs(href: str) -> str:
    """Resolve relative hrefs against FED_BASE."""
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return FED_BASE + href
    return href


def _fmt_month(iso: str) -> str:
    return datetime.strptime(iso, "%Y-%m-%d").strftime("%B %Y")


# ── Individual scrapers ───────────────────────────────────────────────────────

def scrape_fomc_calendar(seen: set) -> list[DocumentRecord]:
    """
    Parse the FOMC calendar page.  Returns DocumentRecords for:
      • FOMC Statements  (monetary{YYYYMMDD}a1.pdf)
      • FOMC Minutes     (fomcminutes{YYYYMMDD}.pdf)
      • SEP              (fomcprojtabl{YYYYMMDD}.pdf)
    """
    records = []
    soup = fetch_page(CALENDAR_URL)
    if not soup:
        log.error("Could not fetch FOMC calendar — %s", CALENDAR_URL)
        return records

    # Patterns map: regex → (doc_type, id_prefix, title_template)
    patterns = [
        (
            re.compile(r'/monetarypolicy/files/monetary(\d{8})a1\.pdf$'),
            "FOMC Statement", "fed_fomc_statement_",
            lambda iso: f"FOMC Statement — {_fmt_month(iso)}",
        ),
        (
            re.compile(r'/monetarypolicy/files/fomcminutes(\d{8})\.pdf$'),
            "FOMC Minutes", "fed_fomc_minutes_",
            lambda iso: f"FOMC Minutes — {_fmt_month(iso)}",
        ),
        (
            re.compile(r'/monetarypolicy/files/fomcprojtabl(\d{8})\.pdf$'),
            "Summary of Economic Projections", "fed_fomc_sep_",
            lambda iso: f"FOMC SEP — {_fmt_month(iso)}",
        ),
    ]

    for a in soup.find_all("a", href=True):
        url = _abs(a["href"])
        for pattern, doc_type, id_prefix, title_fn in patterns:
            m = pattern.search(url)
            if not m:
                continue
            from .base import parse_yyyymmdd, is_in_range
            d = parse_yyyymmdd(m.group(1))
            if not d or not is_in_range(d):
                continue
            iso = d.isoformat()
            doc_id = id_prefix + iso.replace("-", "")
            if doc_id in seen:
                continue
            dest = FED_DIR / f"{doc_id}.pdf"
            rec = DocumentRecord(
                doc_id=doc_id,
                institution="Federal Reserve",
                doc_type=doc_type,
                date=iso,
                title=title_fn(iso),
                url=url,
                local_path=str(dest.relative_to(BASE_DIR)),
                format="pdf",
            )
            download_pdf(url, dest)
            records.append(rec)
            seen.add(doc_id)

    return records


def _collect_beige_books_from_page(soup, seen: set) -> list[DocumentRecord]:
    """Extract Beige Book PDF records from a parsed page."""
    from .base import parse_yyyymmdd
    records = []
    pattern = re.compile(r'BeigeBook_(\d{8})\.pdf$', re.IGNORECASE)
    for a in soup.find_all("a", href=True):
        url = _abs(a["href"])
        m = pattern.search(url)
        if not m:
            continue
        d = parse_yyyymmdd(m.group(1))
        if not d or not is_in_range(d):
            continue
        iso = d.isoformat()
        doc_id = f"fed_beige_book_{iso.replace('-','')}"
        if doc_id in seen:
            continue
        dest = FED_DIR / f"{doc_id}.pdf"
        dt = datetime.strptime(iso, "%Y-%m-%d")
        rec = DocumentRecord(
            doc_id=doc_id,
            institution="Federal Reserve",
            doc_type="Beige Book",
            date=iso,
            title=f"Beige Book — {dt.strftime('%B %d, %Y')}",
            url=url,
            local_path=str(dest.relative_to(BASE_DIR)),
            format="pdf",
        )
        download_pdf(url, dest)
        records.append(rec)
        seen.add(doc_id)
    return records


def scrape_beige_book(seen: set) -> list[DocumentRecord]:
    """
    Collect Beige Book PDFs from the current listing page plus per-year
    archive pages covering every year in DATE_FROM..DATE_TO.
    """
    from .base import DATE_FROM, DATE_TO
    records = []
    # Current-year listing first (has 2026+ books)
    urls = [BEIGE_BOOK_URL]
    # Add per-year archive pages for every year we care about
    for year in range(DATE_FROM.year, DATE_TO.year + 1):
        urls.append(f"{FED_BASE}/monetarypolicy/beigebook{year}.htm")

    for url in urls:
        soup = fetch_page(url)
        if not soup:
            log.warning("Could not fetch Beige Book page — %s", url)
            continue
        recs = _collect_beige_books_from_page(soup, seen)
        if recs:
            log.info("  %d Beige Books from %s", len(recs), url)
        records.extend(recs)
    return records


def scrape_mpr(seen: set) -> list[DocumentRecord]:
    """
    Parse the Fed MPR landing page and collect Monetary Policy Report PDFs.
    Typical filename: {YYYYMMDD}_mprfullreport.pdf
    """
    records = []
    soup = fetch_page(MPR_URL)
    if not soup:
        log.error("Could not fetch Fed MPR page — %s", MPR_URL)
        return records

    # Match any PDF link that has both a date and "mpr" (case-insensitive)
    pattern = re.compile(r'(\d{8}).*mpr.*\.pdf|mpr.*(\d{8}).*\.pdf', re.IGNORECASE)
    from .base import parse_yyyymmdd

    for a in soup.find_all("a", href=True):
        href = a["href"]
        url = _abs(href)
        if not url.lower().endswith(".pdf"):
            continue
        m = pattern.search(url)
        if not m:
            continue
        date_str = m.group(1) or m.group(2)
        if not date_str:
            continue
        d = parse_yyyymmdd(date_str)
        if not d or not is_in_range(d):
            continue
        iso = d.isoformat()
        doc_id = f"fed_mpr_{iso.replace('-','')}"
        if doc_id in seen:
            continue
        dest = FED_DIR / f"{doc_id}.pdf"
        rec = DocumentRecord(
            doc_id=doc_id,
            institution="Federal Reserve",
            doc_type="Monetary Policy Report",
            date=iso,
            title=f"Fed Monetary Policy Report — {_fmt_month(iso)}",
            url=url,
            local_path=str(dest.relative_to(BASE_DIR)),
            format="pdf",
        )
        download_pdf(url, dest)
        records.append(rec)
        seen.add(doc_id)

    return records


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run(meta: dict) -> list[DocumentRecord]:
    """
    Run all Fed scrapers.  `meta` is the shared metadata dict; new records
    are added in-place.  Returns the full list of collected records.
    """
    FED_DIR.mkdir(parents=True, exist_ok=True)
    seen = set(meta.keys())  # skip already-recorded docs
    all_records: list[DocumentRecord] = []

    log.info("── Federal Reserve ─────────────────────────────────────────")

    log.info("Scraping FOMC calendar (statements, minutes, SEP)…")
    recs = scrape_fomc_calendar(seen)
    log.info("  %d new FOMC documents", len(recs))
    all_records.extend(recs)

    log.info("Scraping Beige Book…")
    recs = scrape_beige_book(seen)
    log.info("  %d new Beige Books", len(recs))
    all_records.extend(recs)

    log.info("Scraping Fed Monetary Policy Report…")
    recs = scrape_mpr(seen)
    log.info("  %d new Fed MPRs", len(recs))
    all_records.extend(recs)

    # Persist new records to metadata
    for rec in all_records:
        meta[rec.doc_id] = {
            "doc_id": rec.doc_id,
            "institution": rec.institution,
            "doc_type": rec.doc_type,
            "date": rec.date,
            "title": rec.title,
            "url": rec.url,
            "local_path": rec.local_path,
            "format": rec.format,
        }

    return all_records
