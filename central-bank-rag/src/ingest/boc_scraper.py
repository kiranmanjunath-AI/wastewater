"""
Bank of Canada document scraper.

Collects:
  - Rate Decision Statements   (HTML saved to disk)
  - Monetary Policy Reports    (PDF)
  - Governing Council Deliberations (HTML saved to disk)

All downloads are idempotent: existing files are never re-fetched.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .base import (
    BASE_DIR, RAW_DIR, DATE_FROM,
    DocumentRecord,
    fetch_page, download_pdf, save_html_content,
    is_in_range, parse_iso, parse_yyyymmdd,
)

log = logging.getLogger(__name__)

BOC_BASE = "https://www.bankofcanada.ca"
BOC_DIR  = RAW_DIR / "boc"

MPR_LISTING_URL   = f"{BOC_BASE}/publications/mpr/"
DELIBERATIONS_URL = f"{BOC_BASE}/search/?content_type[]=summary-of-deliberations&per_page=50"

# All BoC Fixed Announcement Dates from Jan 2025 onward.
# These are published in advance on bankofcanada.ca — no scraping needed.
BOC_FAD_DATES = [
    "2025-01-29", "2025-03-12", "2025-04-16", "2025-06-04",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-10",
    "2026-07-15", "2026-09-02",
]


def _fmt_month(iso: str) -> str:
    return datetime.strptime(iso, "%Y-%m-%d").strftime("%B %Y")


def _abs(href: str) -> str:
    if href.startswith("http"):
        return href
    return urljoin(BOC_BASE, href)


# ── Rate Decisions ────────────────────────────────────────────────────────────

def scrape_rate_decisions(seen: set) -> list[DocumentRecord]:
    """
    Download each BoC rate decision press release page.

    URLs follow: bankofcanada.ca/{YYYY}/{MM}/fad-press-release-{YYYY}-{MM}-{DD}/
    We construct them directly from BOC_FAD_DATES — no pagination needed.
    Content is HTML (no PDF available); the full page is saved to disk.
    """
    records = []

    for iso in BOC_FAD_DATES:
        d = parse_iso(iso)
        if not d or not is_in_range(d):
            continue

        doc_id = f"boc_rate_decision_{iso.replace('-','')}"
        if doc_id in seen:
            continue

        yyyy, mm, dd = iso[:4], iso[5:7], iso[8:10]
        url = f"{BOC_BASE}/{yyyy}/{mm}/fad-press-release-{iso}/"
        dest = BOC_DIR / f"{doc_id}.html"

        content = save_html_content(url, dest)
        if content is not None:
            # Extract the headline from the saved HTML
            soup = BeautifulSoup(content, "html.parser")
            h1 = soup.find("h1")
            title = h1.get_text(strip=True) if h1 else f"BoC Rate Decision — {_fmt_month(iso)}"
            rec = DocumentRecord(
                doc_id=doc_id,
                institution="Bank of Canada",
                doc_type="Rate Decision Statement",
                date=iso,
                title=title,
                url=url,
                local_path=str(dest.relative_to(BASE_DIR)),
                format="html",
            )
            records.append(rec)
            seen.add(doc_id)
        else:
            log.warning("Could not fetch BoC rate decision for %s — URL may be wrong: %s", iso, url)

    return records


# ── Monetary Policy Reports ───────────────────────────────────────────────────

def scrape_mpr(seen: set) -> list[DocumentRecord]:
    """
    1. Scrape the MPR listing page for hub links (mpr-YYYY-MM-DD).
    2. For each hub page, find the PDF download link.
    3. Download the PDF.
    """
    records = []
    soup = fetch_page(MPR_LISTING_URL)
    if not soup:
        log.error("Could not fetch BoC MPR listing — %s", MPR_LISTING_URL)
        return records

    hub_pattern = re.compile(r'/publications/mpr/mpr-(\d{4}-\d{2}-\d{2})/')

    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = hub_pattern.search(href)
        if not m:
            continue
        iso = m.group(1)
        d = parse_iso(iso)
        if not d or not is_in_range(d):
            continue

        doc_id = f"boc_mpr_{iso.replace('-','')}"
        if doc_id in seen:
            continue

        hub_url = _abs(href)
        hub_soup = fetch_page(hub_url)
        if not hub_soup:
            log.warning("Could not fetch BoC MPR hub: %s", hub_url)
            continue

        # Find the PDF link on the hub page
        pdf_url = _find_mpr_pdf(hub_soup, iso)
        if not pdf_url:
            log.warning("No PDF found on BoC MPR hub: %s", hub_url)
            continue

        dest = BOC_DIR / f"{doc_id}.pdf"
        link_text = a.get_text(strip=True)
        title = link_text if link_text else f"BoC Monetary Policy Report — {_fmt_month(iso)}"

        download_pdf(pdf_url, dest)
        rec = DocumentRecord(
            doc_id=doc_id,
            institution="Bank of Canada",
            doc_type="Monetary Policy Report",
            date=iso,
            title=title,
            url=pdf_url,
            local_path=str(dest.relative_to(BASE_DIR)),
            format="pdf",
        )
        records.append(rec)
        seen.add(doc_id)

    return records


def _find_mpr_pdf(soup: BeautifulSoup, iso: str) -> Optional[str]:
    """Return the PDF URL from an MPR hub page."""
    # Primary: look for an <a> whose text is exactly "PDF"
    for a in soup.find_all("a", href=True):
        if a.get_text(strip=True).upper() == "PDF" and ".pdf" in a["href"].lower():
            return _abs(a["href"])

    # Fallback: any link with mpr-{date}.pdf in the path
    slug = f"mpr-{iso}"
    for a in soup.find_all("a", href=True):
        if slug in a["href"] and a["href"].lower().endswith(".pdf"):
            return _abs(a["href"])

    # Last resort: any PDF link on the page
    for a in soup.find_all("a", href=True):
        if a["href"].lower().endswith(".pdf") and "/wp-content/" in a["href"]:
            return _abs(a["href"])

    return None


# ── Governing Council Deliberations ──────────────────────────────────────────

def scrape_deliberations(seen: set) -> list[DocumentRecord]:
    """
    Scrape BoC Governing Council deliberations from the search index.

    Uses per_page=50 to fetch all 29 entries on a single page — no pagination.
    Content is HTML; the full page is saved to disk.
    """
    records = []
    soup = fetch_page(DELIBERATIONS_URL)
    if not soup:
        log.error("Could not fetch BoC deliberations search page")
        return records

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "deliberation" not in href.lower():
            continue

        title_text = a.get_text(strip=True)
        iso = _parse_delib_date(href, title_text)
        if not iso:
            continue
        d = parse_iso(iso)
        if not d or not is_in_range(d):
            continue

        doc_id = f"boc_deliberations_{iso.replace('-','')}"
        if doc_id in seen:
            continue

        full_url = _abs(href)
        dest = BOC_DIR / f"{doc_id}.html"
        title = title_text if title_text else f"BoC Governing Council Deliberations — {_fmt_month(iso)}"

        content = save_html_content(full_url, dest)
        if content is not None:
            rec = DocumentRecord(
                doc_id=doc_id,
                institution="Bank of Canada",
                doc_type="Governing Council Deliberations",
                date=iso,
                title=title,
                url=full_url,
                local_path=str(dest.relative_to(BASE_DIR)),
                format="html",
            )
            records.append(rec)
            seen.add(doc_id)

    return records


# Month name → zero-padded month number
_MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}


def _parse_delib_date(href: str, title: str) -> Optional[str]:
    """
    Extract the FAD date (YYYY-MM-DD) from a deliberations URL or title.

    URLs look like:
      …/summary-of-governing-council-deliberations-fixed-announcement-date-of-july-15-2026/
      …/summary-governing-council-deliberations-fixed-announcement-date-of-march-18-2026/

    Title looks like:
      "Summary of Governing Council deliberations: Fixed announcement date of July 15, 2026"
    """
    # Try URL slug first
    # Pattern: -{month}-{day}-{year}/ at end of slug
    m = re.search(
        r'-of-(\w+)-(\d{1,2})-(\d{4})/?$',
        href.rstrip("/").split("?")[0],
    )
    if m:
        month_name, day, year = m.group(1).lower(), m.group(2), m.group(3)
        mm = _MONTHS.get(month_name)
        if mm:
            return f"{year}-{mm}-{int(day):02d}"

    # Fall back to parsing the title string
    m = re.search(
        r'of\s+(\w+)\s+(\d{1,2}),?\s+(\d{4})',
        title,
        re.IGNORECASE,
    )
    if m:
        month_name, day, year = m.group(1).lower(), m.group(2), m.group(3)
        mm = _MONTHS.get(month_name)
        if mm:
            return f"{year}-{mm}-{int(day):02d}"

    return None


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run(meta: dict) -> list[DocumentRecord]:
    """
    Run all BoC scrapers. `meta` is the shared metadata dict; new records
    are added in-place. Returns the full list of collected records.
    """
    BOC_DIR.mkdir(parents=True, exist_ok=True)
    seen = set(meta.keys())
    all_records: list[DocumentRecord] = []

    log.info("── Bank of Canada ───────────────────────────────────────────")

    log.info("Scraping rate decision statements…")
    recs = scrape_rate_decisions(seen)
    log.info("  %d new rate decisions", len(recs))
    all_records.extend(recs)

    log.info("Scraping Monetary Policy Reports…")
    recs = scrape_mpr(seen)
    log.info("  %d new MPRs", len(recs))
    all_records.extend(recs)

    log.info("Scraping Governing Council deliberations…")
    recs = scrape_deliberations(seen)
    log.info("  %d new deliberations", len(recs))
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
