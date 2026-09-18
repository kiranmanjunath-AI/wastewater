"""
fetch_ita.py
Download the Canadian Income Tax Act XML from Justice Laws (both EN and FR).
Saves raw XML to DATA_DIR and returns local file paths.
"""

import os
import sys
import hashlib
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ITA_URLS = {
    "en": "https://laws-justice.gc.ca/eng/acts/I-3.3/FullText.xml",
    "fr": "https://laws-lois.justice.gc.ca/fra/lois/I-3.3/TexteComplet.xml",
}

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))


def compute_hash(content: bytes) -> str:
    """Return sha256:<hex> hash of raw bytes."""
    digest = hashlib.sha256(content).hexdigest()
    return f"sha256:{digest}"


def fetch_xml(language: str, url: str, timeout: int = 60) -> tuple[bytes, str]:
    """
    Fetch XML from the given URL.

    Returns (content_bytes, version_hash).
    Raises RuntimeError with a descriptive message on failure.
    """
    print(f"[fetch] Downloading ITA ({language.upper()}) from: {url}")
    try:
        response = requests.get(url, timeout=timeout, headers={"Accept": "application/xml, text/xml, */*"})
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not connect to Justice Laws server for language '{language}'. "
            f"Check your internet connection or try again later. URL: {url}\n  Cause: {exc}"
        ) from exc
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Request timed out after {timeout}s fetching '{language}' ITA. "
            f"Try increasing the timeout or check your connection. URL: {url}"
        )
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"HTTP {response.status_code} error fetching '{language}' ITA. "
            f"The URL may have changed. URL: {url}\n  Cause: {exc}"
        ) from exc

    content = response.content

    # Sanity-check: confirm we got XML, not an HTML error page
    sample = content[:500].lower()
    if b"<?xml" not in sample and b"<statute" not in sample and b"<loi" not in sample:
        # Try to detect HTML error page
        if b"<html" in sample or b"<!doctype" in sample:
            raise RuntimeError(
                f"Expected XML from Justice Laws but got an HTML page for language '{language}'. "
                f"The URL may have moved. URL: {url}\n"
                f"  First 200 chars of response: {content[:200]}"
            )
        print(
            f"[fetch] WARNING: Response for '{language}' does not start with expected XML markers. "
            f"Proceeding anyway, but parsing may fail.\n"
            f"  First 200 bytes: {content[:200]}"
        )

    version_hash = compute_hash(content)
    print(f"[fetch]   Size: {len(content):,} bytes  Hash: {version_hash}")
    return content, version_hash


def save_xml(language: str, content: bytes) -> Path:
    """Save raw XML to DATA_DIR/ita_<language>.xml and return the path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"ita_{language}.xml"
    out_path.write_bytes(content)
    print(f"[fetch]   Saved to: {out_path}")
    return out_path


def fetch_all(languages: list[str] | None = None) -> dict[str, dict]:
    """
    Fetch ITA XML for each requested language.

    Returns a dict keyed by language with keys:
      path        Path – local file path
      version_hash str – sha256:... of raw content
      url         str – source URL

    Raises RuntimeError for any language that fails.
    """
    if languages is None:
        languages = list(ITA_URLS.keys())

    results: dict[str, dict] = {}
    errors: list[str] = []

    for lang in languages:
        if lang not in ITA_URLS:
            print(f"[fetch] WARNING: Unknown language '{lang}', skipping.")
            continue
        url = ITA_URLS[lang]
        try:
            content, version_hash = fetch_xml(lang, url)
            path = save_xml(lang, content)
            results[lang] = {"path": path, "version_hash": version_hash, "url": url}
        except RuntimeError as exc:
            print(f"[fetch] ERROR ({lang}): {exc}", file=sys.stderr)
            errors.append(str(exc))

    if errors and not results:
        raise RuntimeError(
            "All language downloads failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    if errors:
        print(f"[fetch] WARNING: {len(errors)} language(s) failed; continuing with partial results.")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download ITA XML from Justice Laws")
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=["en", "fr"],
        default=["en", "fr"],
        help="Languages to fetch (default: en fr)",
    )
    args = parser.parse_args()

    try:
        results = fetch_all(args.languages)
        for lang, info in results.items():
            print(f"\n[fetch] {lang.upper()} ready: {info['path']}")
    except RuntimeError as exc:
        print(f"\n[fetch] FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
