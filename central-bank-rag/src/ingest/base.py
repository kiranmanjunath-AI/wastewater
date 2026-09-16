"""Shared utilities for the central bank RAG ingestion pipeline."""

import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_FILE = DATA_DIR / "metadata.json"

# Capture roughly 20 months of documents
DATE_FROM = date(2025, 1, 1)
DATE_TO = date.today()

_SESSION: Optional[requests.Session] = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({
            "User-Agent": "central-bank-rag/1.0 (academic research; public documents only)",
            "Accept": "text/html,application/pdf,*/*",
        })
    return _SESSION


# ── Metadata store ────────────────────────────────────────────────────────────

@dataclass
class DocumentRecord:
    doc_id: str
    institution: str   # "Federal Reserve" | "Bank of Canada"
    doc_type: str      # e.g. "FOMC Statement", "Beige Book", "MPR", …
    date: str          # ISO YYYY-MM-DD
    title: str
    url: str
    local_path: str    # relative to BASE_DIR
    format: str        # "pdf" | "html"


def load_metadata() -> dict:
    if METADATA_FILE.exists():
        return json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    return {}


def save_metadata(meta: dict) -> None:
    METADATA_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ── Date helpers ──────────────────────────────────────────────────────────────

def is_in_range(d: date) -> bool:
    return DATE_FROM <= d <= DATE_TO


def parse_yyyymmdd(s: str) -> Optional[date]:
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except ValueError:
        return None


def parse_iso(s: str) -> Optional[date]:
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def extract_date_from_href(href: str) -> Optional[date]:
    """Pull the first 8-digit run out of a URL and return a date if in range."""
    m = re.search(r'(\d{8})', href)
    if m:
        d = parse_yyyymmdd(m.group(1))
        if d and is_in_range(d):
            return d
    return None


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def fetch_page(url: str, retries: int = 3, pause: float = 1.5) -> Optional[BeautifulSoup]:
    """GET a page and return BeautifulSoup, or None on repeated failure."""
    sess = _session()
    for attempt in range(retries):
        try:
            r = sess.get(url, timeout=30)
            r.raise_for_status()
            time.sleep(pause)
            return BeautifulSoup(r.text, "html.parser")
        except requests.RequestException as e:
            log.warning("fetch_page attempt %d/%d failed for %s — %s", attempt + 1, retries, url, e)
            if attempt < retries - 1:
                time.sleep(pause * 2)
    return None


def download_pdf(url: str, dest: Path, pause: float = 1.0) -> bool:
    """
    Download a PDF to dest. Returns True if a new download was made.
    Skips silently if the file already exists (idempotent).
    """
    if dest.exists():
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    sess = _session()

    for attempt in range(3):
        try:
            r = sess.get(url, timeout=60, stream=True)
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=65_536):
                    fh.write(chunk)
            log.info("  ✓ PDF  %s", dest.name)
            time.sleep(pause)
            return True
        except requests.RequestException as e:
            log.warning("  ✗ PDF attempt %d: %s — %s", attempt + 1, dest.name, e)
            if attempt < 2:
                time.sleep(3)

    return False


def save_html_content(url: str, dest: Path, pause: float = 1.5) -> Optional[str]:
    """
    Fetch a URL and save the response body as UTF-8 HTML.
    Returns the text on success, None on failure.
    Skips the fetch if the file already exists.
    """
    if dest.exists():
        return dest.read_text(encoding="utf-8")

    dest.parent.mkdir(parents=True, exist_ok=True)
    sess = _session()

    try:
        r = sess.get(url, timeout=30)
        r.raise_for_status()
        text = r.text
        dest.write_text(text, encoding="utf-8")
        log.info("  ✓ HTML %s", dest.name)
        time.sleep(pause)
        return text
    except requests.RequestException as e:
        log.warning("  ✗ HTML %s — %s", url, e)
        return None
