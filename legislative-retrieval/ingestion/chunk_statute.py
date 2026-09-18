"""
chunk_statute.py
Parse the Justice Laws XML for the Canadian Income Tax Act and produce
citation-aware chunks suitable for embedding and loading into Qdrant.

Chunking strategy: hierarchical
  - Retrievable unit: subsection (or paragraph if subsection > 400 tokens)
  - Context prefix: parent section's MarginalNote + "Part X, Division Y" breadcrumb
  - If a subsection exceeds 400 tokens, split at paragraph boundaries

Returns a list of dicts matching the Qdrant payload schema.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from lxml import etree

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))

# Rough token estimate: words * 1.3
TOKEN_MULTIPLIER = 1.3
SUBSECTION_TOKEN_LIMIT = 400


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * TOKEN_MULTIPLIER)


def sha256_hex(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(node) -> str:
    """Extract all text content from an lxml element, collapsing whitespace."""
    if node is None:
        return ""
    parts = []
    for part in node.itertext():
        parts.append(part)
    return " ".join(" ".join(parts).split())


def get_label(node, default: str = "") -> str:
    """Return label attribute, stripping whitespace."""
    return (node.get("label") or node.get("id") or default).strip()


def build_citation(
    section: str,
    subsection: Optional[str] = None,
    paragraph: Optional[str] = None,
    subparagraph: Optional[str] = None,
) -> str:
    """Build a human-readable ITA citation string."""
    cit = f"ITA s.{section}"
    if subsection:
        cit += f"({subsection})"
    if paragraph:
        cit += f"({paragraph})"
    if subparagraph:
        cit += f"({subparagraph})"
    return cit


def make_chunk(
    text: str,
    context_prefix: str,
    citation: str,
    act: str,
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
        "act": act,
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
        "version_hash": sha256_hex(full_text),
        "token_count": estimate_tokens(full_text),
    }


# ---------------------------------------------------------------------------
# XML namespace resolver
# ---------------------------------------------------------------------------

def strip_ns(tag: str) -> str:
    """Remove Clark-notation namespace from a tag name."""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def find_all(node, local_name: str):
    """Find all direct children whose local name matches, ignoring namespace."""
    return [c for c in node if strip_ns(c.tag) == local_name]


def find_first(node, local_name: str):
    """Find first child with matching local name."""
    for c in node:
        if strip_ns(c.tag) == local_name:
            return c
    return None


def find_recursive(node, local_name: str):
    """Recursively find all elements whose local name matches."""
    results = []
    for c in node.iter():
        if strip_ns(c.tag) == local_name:
            results.append(c)
    return results


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_paragraph_text(para_node) -> str:
    """Extract text from a Paragraph or Subparagraph node."""
    texts = []
    for child in para_node:
        local = strip_ns(child.tag)
        if local in ("Text", "Definition", "DefinedTermEn", "DefinedTermFr"):
            texts.append(clean_text(child))
        elif local in ("Subparagraph", "Clause", "Subclause"):
            sub_label = get_label(child)
            sub_text = parse_paragraph_text(child)
            if sub_label and sub_text:
                texts.append(f"({sub_label}) {sub_text}")
            elif sub_text:
                texts.append(sub_text)
    if not texts:
        texts.append(clean_text(para_node))
    return " ".join(t for t in texts if t)


def chunks_from_subsection(
    sub_node,
    sub_label: str,
    section_label: str,
    context_prefix: str,
    part_label: str,
    division_label: str,
    language: str,
    valid_from: str,
    amending_act: str,
    act: str = "ITA",
) -> list[dict]:
    """
    Produce one or more chunks from a single <Subsection> node.

    If total token count <= SUBSECTION_TOKEN_LIMIT, return one chunk for the
    whole subsection.  Otherwise, split at <Paragraph> boundaries.
    """
    # Collect top-level text (before any paragraph)
    intro_parts = []
    for child in sub_node:
        local = strip_ns(child.tag)
        if local == "Text":
            intro_parts.append(clean_text(child))

    intro_text = " ".join(intro_parts)

    # Collect paragraphs
    paragraphs = find_all(sub_node, "Paragraph")
    if not paragraphs:
        # No paragraphs — the subsection IS the atomic unit
        text = intro_text or clean_text(sub_node)
        if not text:
            return []
        citation = build_citation(section_label, sub_label)
        return [
            make_chunk(
                text=text,
                context_prefix=context_prefix,
                citation=citation,
                act=act,
                part=part_label,
                division=division_label,
                section=section_label,
                subsection=sub_label,
                paragraph=None,
                language=language,
                valid_from=valid_from,
                amending_act=amending_act,
            )
        ]

    # Build full subsection text to check size
    para_texts = {}
    for p in paragraphs:
        p_label = get_label(p)
        p_text = parse_paragraph_text(p)
        para_texts[p_label] = p_text

    all_text = intro_text + " " + " ".join(para_texts.values())
    if estimate_tokens(all_text) <= SUBSECTION_TOKEN_LIMIT:
        # Fits in one chunk
        citation = build_citation(section_label, sub_label)
        return [
            make_chunk(
                text=all_text.strip(),
                context_prefix=context_prefix,
                citation=citation,
                act=act,
                part=part_label,
                division=division_label,
                section=section_label,
                subsection=sub_label,
                paragraph=None,
                language=language,
                valid_from=valid_from,
                amending_act=amending_act,
            )
        ]

    # Split into per-paragraph chunks; include intro in first paragraph
    chunks = []
    first = True
    for p_label, p_text in para_texts.items():
        if first and intro_text:
            combined = f"{intro_text} ({p_label}) {p_text}"
            first = False
        else:
            combined = f"({p_label}) {p_text}"
        if not combined.strip():
            continue
        citation = build_citation(section_label, sub_label, p_label)
        chunks.append(
            make_chunk(
                text=combined.strip(),
                context_prefix=context_prefix,
                citation=citation,
                act=act,
                part=part_label,
                division=division_label,
                section=section_label,
                subsection=sub_label,
                paragraph=p_label,
                language=language,
                valid_from=valid_from,
                amending_act=amending_act,
            )
        )

    return chunks


def parse_ita_xml(
    xml_path: Path,
    language: str = "en",
    valid_from: str = "2024-01-01",
    amending_act: str = "",
) -> list[dict]:
    """
    Parse the ITA XML file and return a flat list of chunk dicts.
    Prints progress as it goes.
    """
    print(f"[parse] Reading {xml_path} ...")
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    chunks: list[dict] = []
    section_count = 0
    subsection_count = 0
    warning_count = 0

    # The root may be <Statute> or <Loi>; find Body or Corps
    body_candidates = ["Body", "Corps", "Statute", "Loi"]
    body = None
    for name in body_candidates:
        body = find_first(root, name)
        if body is not None:
            break
    if body is None:
        body = root  # fall back to root

    # Try to locate Part containers; if none, treat body as flat
    parts = find_all(body, "Part") or find_all(body, "Partie")
    if not parts:
        # Some XML versions nest everything directly under Body
        parts = [body]
        part_label_override = ""
    else:
        part_label_override = None

    for part_node in parts:
        if part_label_override is not None:
            part_label = part_label_override
        else:
            part_label = get_label(part_node, "?")

        # Divisions (optional)
        divisions = find_all(part_node, "Division") or find_all(part_node, "Section")
        # If no divisions, treat the part itself as a flat list of sections
        if not divisions:
            divisions = [part_node]
            div_label_override = ""
        else:
            div_label_override = None

        for div_node in divisions:
            if div_label_override is not None:
                div_label = div_label_override
            else:
                div_label = get_label(div_node, "?")

            # Find all Section elements anywhere under this division
            sections = find_all(div_node, "Section") or find_recursive(div_node, "Section")

            for sec_node in sections:
                sec_label = get_label(sec_node)
                if not sec_label:
                    warning_count += 1
                    continue

                # MarginalNote is the heading for this section
                marg = find_first(sec_node, "MarginalNote") or find_first(sec_node, "NoteMarg")
                marginal_text = clean_text(marg) if marg is not None else ""

                breadcrumb_parts = []
                if part_label:
                    breadcrumb_parts.append(f"Part {part_label}")
                if div_label:
                    breadcrumb_parts.append(f"Division {div_label}")
                breadcrumb = ", ".join(breadcrumb_parts)

                if marginal_text and breadcrumb:
                    context_prefix = f"{marginal_text} [{breadcrumb}]"
                elif marginal_text:
                    context_prefix = marginal_text
                elif breadcrumb:
                    context_prefix = f"[{breadcrumb}]"
                else:
                    context_prefix = f"ITA s.{sec_label}"

                section_count += 1

                # Find subsections
                subsections = find_all(sec_node, "Subsection") or find_all(sec_node, "Paragraphe")

                if not subsections:
                    # Flat section — treat body text as one chunk
                    text = clean_text(sec_node)
                    if marginal_text:
                        text = text.replace(marginal_text, "").strip()
                    if not text:
                        continue
                    citation = build_citation(sec_label)
                    chunks.append(
                        make_chunk(
                            text=text,
                            context_prefix=context_prefix,
                            citation=citation,
                            act="ITA",
                            part=part_label,
                            division=div_label,
                            section=sec_label,
                            subsection=None,
                            paragraph=None,
                            language=language,
                            valid_from=valid_from,
                            amending_act=amending_act,
                        )
                    )
                    subsection_count += 1
                    continue

                for sub_node in subsections:
                    sub_label = get_label(sub_node)
                    new_chunks = chunks_from_subsection(
                        sub_node=sub_node,
                        sub_label=sub_label,
                        section_label=sec_label,
                        context_prefix=context_prefix,
                        part_label=part_label,
                        division_label=div_label,
                        language=language,
                        valid_from=valid_from,
                        amending_act=amending_act,
                    )
                    chunks.extend(new_chunks)
                    subsection_count += 1

                if section_count % 50 == 0:
                    print(
                        f"[parse]   ... {section_count} sections, {len(chunks)} chunks so far"
                    )

    print(
        f"[parse] Done. Sections: {section_count}, subsections processed: {subsection_count}, "
        f"chunks produced: {len(chunks)}, warnings: {warning_count}"
    )
    return chunks


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Parse ITA XML into citation-aware chunks")
    parser.add_argument(
        "--xml",
        type=Path,
        default=DATA_DIR / "ita_en.xml",
        help="Path to the ITA XML file",
    )
    parser.add_argument(
        "--language",
        default="en",
        choices=["en", "fr"],
        help="Language tag for the chunks",
    )
    parser.add_argument(
        "--valid-from",
        default="2024-01-01",
        help="ISO date for valid_from field",
    )
    parser.add_argument(
        "--amending-act",
        default="",
        help="Amending act identifier (e.g. 'S.C. 2024, c. 15')",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path for inspection",
    )
    args = parser.parse_args()

    if not args.xml.exists():
        print(f"[parse] ERROR: XML file not found: {args.xml}", file=sys.stderr)
        print("[parse] Run ingestion/fetch_ita.py first.", file=sys.stderr)
        sys.exit(1)

    result = parse_ita_xml(
        xml_path=args.xml,
        language=args.language,
        valid_from=args.valid_from,
        amending_act=args.amending_act,
    )

    if args.out:
        args.out.write_text(json.dumps(result[:20], indent=2, ensure_ascii=False))
        print(f"[parse] Sample (20 chunks) written to {args.out}")
    else:
        print(f"\n[parse] First chunk sample:")
        if result:
            import json
            print(json.dumps(result[0], indent=2, ensure_ascii=False))
