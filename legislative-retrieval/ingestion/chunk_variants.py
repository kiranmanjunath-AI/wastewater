"""
chunk_variants.py
Three chunking strategies for evaluation comparison.
All strategies consume the same parsed section data and return a uniform
list-of-dicts matching the Qdrant payload schema.

Strategies:
  1. strict_subsection  — one chunk per subsection, no overlap
  2. sliding_window     — subsections with 1-paragraph overlap with adjacent subsection
  3. hierarchical       — subsection as unit + parent section heading as context prefix (default)

All functions accept the same signature:
    fn(sections: list[dict]) -> list[dict]

where each item in `sections` is the internal parsed-section format produced by
_parse_to_sections(), which is a lightweight re-parse of the XML structure into:
    {
        "part": str,
        "division": str,
        "section": str,
        "marginal_note": str,
        "subsections": [
            {
                "label": str,
                "intro": str,
                "paragraphs": [{"label": str, "text": str}],
            }
        ],
        "language": str,
        "valid_from": str,
        "amending_act": str,
    }
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))

TOKEN_MULTIPLIER = 1.3
SUBSECTION_TOKEN_LIMIT = 400


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * TOKEN_MULTIPLIER)


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_citation(section: str, subsection: Optional[str] = None, paragraph: Optional[str] = None) -> str:
    cit = f"ITA s.{section}"
    if subsection:
        cit += f"({subsection})"
    if paragraph:
        cit += f"({paragraph})"
    return cit


def _make_chunk(
    text: str,
    context_prefix: str,
    citation: str,
    part: str,
    division: str,
    section: str,
    subsection: Optional[str],
    paragraph: Optional[str],
    language: str,
    valid_from: str,
    amending_act: str,
) -> dict:
    full_text = f"{context_prefix} — {text}" if context_prefix else text
    return {
        "act": "ITA",
        "part": part,
        "division": division,
        "section": section,
        "subsection": subsection or "",
        "paragraph": paragraph or "",
        "citation": citation,
        "text": full_text,
        "context_prefix": context_prefix,
        "language": language,
        "valid_from": valid_from,
        "valid_to": None,
        "amending_act": amending_act,
        "version_hash": _sha256(full_text),
        "token_count": _estimate_tokens(full_text),
    }


def _subsection_full_text(sub: dict) -> str:
    """Reconstruct full text of a subsection dict."""
    parts = [sub["intro"]] if sub["intro"] else []
    for p in sub["paragraphs"]:
        parts.append(f"({p['label']}) {p['text']}" if p["label"] else p["text"])
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Import the shared parser from chunk_statute to reuse XML parsing
# ---------------------------------------------------------------------------

def _sections_from_xml(xml_path: Path, language: str, valid_from: str, amending_act: str) -> list[dict]:
    """
    Re-parse the XML into an intermediate section representation.
    Each item: {part, division, section, marginal_note, subsections, language, valid_from, amending_act}
    """
    from lxml import etree
    from ingestion.chunk_statute import (
        find_all, find_first, find_recursive, get_label,
        clean_text, strip_ns, parse_paragraph_text,
    )

    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    body_candidates = ["Body", "Corps", "Statute", "Loi"]
    body = None
    for name in body_candidates:
        body = find_first(root, name)
        if body is not None:
            break
    if body is None:
        body = root

    parts = find_all(body, "Part") or find_all(body, "Partie") or [body]

    sections_data: list[dict] = []

    for part_node in parts:
        part_label = get_label(part_node, "")
        divisions = find_all(part_node, "Division") or find_all(part_node, "Section")
        if not divisions:
            divisions = [part_node]
            div_label_override = ""
        else:
            div_label_override = None

        for div_node in divisions:
            div_label = div_label_override if div_label_override is not None else get_label(div_node, "")
            sections = find_all(div_node, "Section") or find_recursive(div_node, "Section")

            for sec_node in sections:
                sec_label = get_label(sec_node)
                if not sec_label:
                    continue
                marg = find_first(sec_node, "MarginalNote") or find_first(sec_node, "NoteMarg")
                marginal_text = clean_text(marg) if marg is not None else ""

                subsections = find_all(sec_node, "Subsection") or find_all(sec_node, "Paragraphe")
                subs_data = []

                if not subsections:
                    # Treat flat section as a single pseudo-subsection
                    text = clean_text(sec_node)
                    if marginal_text:
                        text = text.replace(marginal_text, "").strip()
                    subs_data.append({"label": "", "intro": text, "paragraphs": []})
                else:
                    for sub_node in subsections:
                        sub_label = get_label(sub_node)
                        intro_parts = []
                        for child in sub_node:
                            if strip_ns(child.tag) == "Text":
                                intro_parts.append(clean_text(child))
                        intro = " ".join(intro_parts)
                        paras = []
                        for p_node in find_all(sub_node, "Paragraph"):
                            p_label = get_label(p_node)
                            p_text = parse_paragraph_text(p_node)
                            paras.append({"label": p_label, "text": p_text})
                        subs_data.append({"label": sub_label, "intro": intro, "paragraphs": paras})

                sections_data.append({
                    "part": part_label,
                    "division": div_label,
                    "section": sec_label,
                    "marginal_note": marginal_text,
                    "subsections": subs_data,
                    "language": language,
                    "valid_from": valid_from,
                    "amending_act": amending_act,
                })

    return sections_data


def _context_prefix(sec: dict) -> str:
    breadcrumb_parts = []
    if sec["part"]:
        breadcrumb_parts.append(f"Part {sec['part']}")
    if sec["division"]:
        breadcrumb_parts.append(f"Division {sec['division']}")
    breadcrumb = ", ".join(breadcrumb_parts)
    if sec["marginal_note"] and breadcrumb:
        return f"{sec['marginal_note']} [{breadcrumb}]"
    elif sec["marginal_note"]:
        return sec["marginal_note"]
    elif breadcrumb:
        return f"[{breadcrumb}]"
    return f"ITA s.{sec['section']}"


# ---------------------------------------------------------------------------
# Strategy 1: strict_subsection
# ---------------------------------------------------------------------------

def strict_subsection(sections: list[dict]) -> list[dict]:
    """
    One chunk per subsection, no overlap.
    If a subsection exceeds SUBSECTION_TOKEN_LIMIT, it is split at paragraph
    boundaries but with NO shared content between adjacent chunks.
    """
    chunks = []
    for sec in sections:
        prefix = _context_prefix(sec)
        for sub in sec["subsections"]:
            full_text = _subsection_full_text(sub)
            if not full_text.strip():
                continue

            if _estimate_tokens(full_text) <= SUBSECTION_TOKEN_LIMIT or not sub["paragraphs"]:
                citation = _build_citation(sec["section"], sub["label"])
                chunks.append(_make_chunk(
                    text=full_text,
                    context_prefix=prefix,
                    citation=citation,
                    part=sec["part"],
                    division=sec["division"],
                    section=sec["section"],
                    subsection=sub["label"] or None,
                    paragraph=None,
                    language=sec["language"],
                    valid_from=sec["valid_from"],
                    amending_act=sec["amending_act"],
                ))
            else:
                # Split at paragraphs strictly — no overlap
                for p in sub["paragraphs"]:
                    p_text = f"({p['label']}) {p['text']}" if p["label"] else p["text"]
                    if not p_text.strip():
                        continue
                    citation = _build_citation(sec["section"], sub["label"], p["label"])
                    chunks.append(_make_chunk(
                        text=p_text,
                        context_prefix=prefix,
                        citation=citation,
                        part=sec["part"],
                        division=sec["division"],
                        section=sec["section"],
                        subsection=sub["label"] or None,
                        paragraph=p["label"] or None,
                        language=sec["language"],
                        valid_from=sec["valid_from"],
                        amending_act=sec["amending_act"],
                    ))
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: sliding_window
# ---------------------------------------------------------------------------

def sliding_window(sections: list[dict]) -> list[dict]:
    """
    Subsection-level chunks with 1-paragraph overlap with the immediately
    adjacent subsection (within the same section).
    Each chunk = current subsection text + first paragraph of next subsection,
    or last paragraph of previous subsection + current subsection text.
    We implement: current sub + first para of next sub (forward-looking overlap).
    """
    chunks = []
    for sec in sections:
        prefix = _context_prefix(sec)
        subs = sec["subsections"]
        for i, sub in enumerate(subs):
            full_text = _subsection_full_text(sub)
            if not full_text.strip():
                continue

            # Append first paragraph of next subsection as overlap context
            overlap_suffix = ""
            if i + 1 < len(subs):
                next_sub = subs[i + 1]
                next_paras = next_sub["paragraphs"]
                if next_paras:
                    p = next_paras[0]
                    overlap_suffix = f" [overlap→{next_sub['label']}] ({p['label']}) {p['text']}"
                elif next_sub["intro"]:
                    overlap_suffix = f" [overlap→{next_sub['label']}] {next_sub['intro'][:200]}"

            combined = full_text + overlap_suffix if overlap_suffix else full_text
            citation = _build_citation(sec["section"], sub["label"])
            chunks.append(_make_chunk(
                text=combined.strip(),
                context_prefix=prefix,
                citation=citation,
                part=sec["part"],
                division=sec["division"],
                section=sec["section"],
                subsection=sub["label"] or None,
                paragraph=None,
                language=sec["language"],
                valid_from=sec["valid_from"],
                amending_act=sec["amending_act"],
            ))
    return chunks


# ---------------------------------------------------------------------------
# Strategy 3: hierarchical (default, mirrors chunk_statute.py)
# ---------------------------------------------------------------------------

def hierarchical(sections: list[dict]) -> list[dict]:
    """
    Subsection as unit + parent section heading as context prefix.
    Split at paragraph boundaries when subsection > SUBSECTION_TOKEN_LIMIT.
    This mirrors the default strategy in chunk_statute.py.
    """
    chunks = []
    for sec in sections:
        prefix = _context_prefix(sec)
        for sub in sec["subsections"]:
            full_text = _subsection_full_text(sub)
            if not full_text.strip():
                continue

            if _estimate_tokens(full_text) <= SUBSECTION_TOKEN_LIMIT or not sub["paragraphs"]:
                citation = _build_citation(sec["section"], sub["label"])
                chunks.append(_make_chunk(
                    text=full_text,
                    context_prefix=prefix,
                    citation=citation,
                    part=sec["part"],
                    division=sec["division"],
                    section=sec["section"],
                    subsection=sub["label"] or None,
                    paragraph=None,
                    language=sec["language"],
                    valid_from=sec["valid_from"],
                    amending_act=sec["amending_act"],
                ))
            else:
                # Per-paragraph chunks sharing the same subsection context prefix
                sub_prefix = f"{prefix} — s.{sec['section']}({sub['label']})" if sub["label"] else prefix
                first = True
                for p in sub["paragraphs"]:
                    p_text = p["text"] if not p["label"] else f"({p['label']}) {p['text']}"
                    if first and sub["intro"]:
                        p_text = sub["intro"] + " " + p_text
                        first = False
                    if not p_text.strip():
                        continue
                    citation = _build_citation(sec["section"], sub["label"], p["label"])
                    chunks.append(_make_chunk(
                        text=p_text.strip(),
                        context_prefix=sub_prefix,
                        citation=citation,
                        part=sec["part"],
                        division=sec["division"],
                        section=sec["section"],
                        subsection=sub["label"] or None,
                        paragraph=p["label"] or None,
                        language=sec["language"],
                        valid_from=sec["valid_from"],
                        amending_act=sec["amending_act"],
                    ))
    return chunks


# ---------------------------------------------------------------------------
# Convenience: run all three strategies on an XML file
# ---------------------------------------------------------------------------

def chunk_all_strategies(
    xml_path: Path,
    language: str = "en",
    valid_from: str = "2024-01-01",
    amending_act: str = "",
) -> dict[str, list[dict]]:
    """
    Parse the XML once and apply all three chunking strategies.
    Returns {"strict_subsection": [...], "sliding_window": [...], "hierarchical": [...]}.
    """
    print(f"[variants] Parsing {xml_path} into section structures ...")
    sections = _sections_from_xml(xml_path, language, valid_from, amending_act)
    print(f"[variants]   {len(sections)} sections parsed.")

    results = {}
    for name, fn in [("strict_subsection", strict_subsection), ("sliding_window", sliding_window), ("hierarchical", hierarchical)]:
        chunks = fn(sections)
        print(f"[variants]   {name}: {len(chunks)} chunks")
        results[name] = chunks

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare chunking strategies for ITA XML")
    parser.add_argument("--xml", type=Path, default=DATA_DIR / "ita_en.xml")
    parser.add_argument("--language", default="en", choices=["en", "fr"])
    parser.add_argument("--valid-from", default="2024-01-01")
    parser.add_argument("--amending-act", default="")
    args = parser.parse_args()

    results = chunk_all_strategies(
        xml_path=args.xml,
        language=args.language,
        valid_from=args.valid_from,
        amending_act=args.amending_act,
    )

    print("\n=== Strategy comparison ===")
    for strategy, chunks in results.items():
        token_counts = [c["token_count"] for c in chunks]
        avg = sum(token_counts) / len(token_counts) if token_counts else 0
        max_t = max(token_counts) if token_counts else 0
        print(f"  {strategy:20s}: {len(chunks):5d} chunks  avg_tokens={avg:.0f}  max_tokens={max_t}")
