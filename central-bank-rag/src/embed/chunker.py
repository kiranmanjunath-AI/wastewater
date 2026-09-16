"""
Extract text from raw documents and split into overlapping chunks.

PDFs: PyMuPDF block-level extraction + find_tables() for structured tables.
HTML: BeautifulSoup block-element extraction including <table> elements.
Both strip navigation/boilerplate. Chunks target ~400 tokens with ~50-token
sentence overlap at boundaries.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz  # PyMuPDF
from bs4 import BeautifulSoup

from ..ingest.base import BASE_DIR

log = logging.getLogger(__name__)

TARGET_CHARS   = 1600  # ~400 tokens at 4 chars/token
OVERLAP_CHARS  = 200   # ~50 tokens
MIN_PARA_CHARS = 20    # discard noise-level text blocks


@dataclass
class Chunk:
    chunk_id:    str
    doc_id:      str
    institution: str
    doc_type:    str
    date:        str
    title:       str
    url:         str
    text:        str
    page_num:    int    # 1-indexed for PDF, 0 for HTML
    chunk_idx:   int


# ── Table helpers ──────────────────────────────────────────────────────────────

def _pdf_table_to_text(rows: list[list[str | None]]) -> str:
    """Convert a PyMuPDF table (list of rows) to pipe-delimited text."""
    lines = []
    for row in rows:
        cells = [(c.strip().replace("\n", " ") if c else "") for c in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _bboxes_overlap(a: tuple, b: tuple) -> bool:
    """True if two (x0, y0, x1, y1) rectangles intersect."""
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _html_table_to_text(table) -> str:
    """Convert a BeautifulSoup <table> to pipe-delimited text rows."""
    lines = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["th", "td"])]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_pdf(path: Path) -> list[tuple[int, str]]:
    """Return [(page_num, para_text), ...] from a PDF.

    Tables are detected with find_tables() and rendered as pipe-delimited rows.
    Text blocks that fall inside a detected table bbox are skipped to avoid
    double-extraction.
    """
    paragraphs: list[tuple[int, str]] = []
    try:
        doc = fitz.open(str(path))
        for page_num, page in enumerate(doc, start=1):
            # Detect tables first so we can exclude their bboxes from block extraction
            try:
                finder = page.find_tables()
                tables = finder.tables
            except Exception:
                tables = []

            table_bboxes = [t.bbox for t in tables]

            # Add each table as a structured text block
            for table in tables:
                text = _pdf_table_to_text(table.extract())
                text = re.sub(r" {2,}", " ", text)
                if len(text) >= MIN_PARA_CHARS:
                    paragraphs.append((page_num, text))

            # Extract prose blocks, skipping any that overlap a table region
            for block in page.get_text("blocks"):
                if block[6] != 0:   # skip image blocks
                    continue
                block_bbox = block[:4]
                if any(_bboxes_overlap(block_bbox, tb) for tb in table_bboxes):
                    continue
                text = block[4].strip().replace("\n", " ")
                text = re.sub(r" {2,}", " ", text)
                if len(text) >= MIN_PARA_CHARS:
                    paragraphs.append((page_num, text))

        doc.close()
    except Exception as exc:
        log.error("PDF extraction failed %s: %s", path.name, exc)
    return paragraphs


def extract_html(path: Path) -> list[tuple[int, str]]:
    """Return [(0, para_text), ...] from a saved HTML file.

    Tables are extracted first and then decomposed from the tree so their
    cell text doesn't also appear inside <p> or <li> siblings.
    """
    paragraphs: list[tuple[int, str]] = []
    try:
        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")

        for tag in soup(["nav", "header", "footer", "script", "style", "aside", "noscript"]):
            tag.decompose()

        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id=re.compile(r"main|content", re.I))
            or soup.find(class_=re.compile(r"main|content|body", re.I))
            or soup.find("body")
        )
        if not main:
            return paragraphs

        # Extract tables first, then remove them so they don't pollute prose extraction
        for table in main.find_all("table"):
            text = _html_table_to_text(table)
            text = re.sub(r" {2,}", " ", text)
            if len(text) >= MIN_PARA_CHARS:
                paragraphs.append((0, text))
            table.decompose()

        # Extract remaining prose block elements
        for elem in main.find_all(["p", "h1", "h2", "h3", "h4", "li"]):
            text = elem.get_text(separator=" ", strip=True)
            text = re.sub(r" {2,}", " ", text)
            if len(text) >= MIN_PARA_CHARS:
                paragraphs.append((0, text))

    except Exception as exc:
        log.error("HTML extraction failed %s: %s", path.name, exc)
    return paragraphs


# ── Chunking ───────────────────────────────────────────────────────────────────

def _first_sentence(text: str) -> str:
    """Up to first sentence boundary, capped at OVERLAP_CHARS."""
    m = re.search(r"[.!?](?:\s|$)", text)
    if m:
        candidate = text[: m.end()].strip()
        if len(candidate) <= OVERLAP_CHARS:
            return candidate
    return text[: OVERLAP_CHARS]


def _make_chunks(paragraphs: list[tuple[int, str]], rec: dict) -> list[Chunk]:
    """Merge paragraphs into TARGET_CHARS chunks with sentence-level overlap."""
    chunks: list[Chunk] = []
    buf_texts: list[str] = []
    buf_pages: list[int] = []
    buf_len = 0

    def flush(extra: str = "") -> None:
        nonlocal buf_texts, buf_pages, buf_len
        if not buf_texts:
            return
        text = " ".join(buf_texts)
        if extra:
            text = text + " " + extra
        chunks.append(Chunk(
            chunk_id=f"{rec['doc_id']}_{len(chunks):04d}",
            doc_id=rec["doc_id"],
            institution=rec["institution"],
            doc_type=rec["doc_type"],
            date=rec["date"],
            title=rec["title"],
            url=rec["url"],
            text=text.strip(),
            page_num=buf_pages[0],
            chunk_idx=len(chunks),
        ))
        buf_texts, buf_pages, buf_len = [], [], 0

    for page_num, text in paragraphs:
        if buf_len + len(text) > TARGET_CHARS and buf_texts:
            # First sentence of current para becomes overlap tail of previous chunk
            flush(_first_sentence(text))
        buf_texts.append(text)
        buf_pages.append(page_num)
        buf_len += len(text)

    flush()
    return chunks


# ── Public API ─────────────────────────────────────────────────────────────────

def chunk_document(rec: dict) -> list[Chunk]:
    path = BASE_DIR / rec["local_path"]
    if not path.exists():
        log.warning("Missing file: %s", rec["local_path"])
        return []
    paragraphs = extract_pdf(path) if rec["format"] == "pdf" else extract_html(path)
    chunks = _make_chunks(paragraphs, rec)
    log.debug("  %-40s %d chunks", rec["doc_id"], len(chunks))
    return chunks


def chunk_all(meta: dict) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for rec in meta.values():
        all_chunks.extend(chunk_document(rec))
    log.info("Chunked %d documents -> %d chunks", len(meta), len(all_chunks))
    return all_chunks
