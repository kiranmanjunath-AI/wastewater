"""
chunk_statute.py
Parse the Justice Laws XML for the Canadian Income Tax Act and produce
citation-aware chunks suitable for embedding and loading into Qdrant.

Actual Justice Laws XML structure (verified against live document):
  <Statute>
    <Identification> ... </Identification>
    <Body>
      <Heading level="1"> <Label>PART I</Label> <TitleText>Income Tax</TitleText> </Heading>
      <Heading level="2"> <Label>DIVISION A</Label> <TitleText>Liability for Tax</TitleText> </Heading>
      <Heading level="3"> <TitleText>Basic Rules</TitleText> </Heading>  (no Label)
      <Section>
        <MarginalNote>Tax payable...</MarginalNote>
        <Label>2</Label>
        <Subsection>
          <Label>(1)</Label>
          <Text>...</Text>
          <Paragraph> <Label>(a)</Label> <Text>...</Text> </Paragraph>
          <ContinuedSectionSubsection> <Text>...</Text> </ContinuedSectionSubsection>
        </Subsection>
        <HistoricalNote>...</HistoricalNote>
      </Section>
      ...
    </Body>
  </Statute>

Key facts:
  - Sections are FLAT under <Body>, not nested in <Part>/<Division> containers.
  - Part/Division context is tracked from preceding <Heading> siblings.
  - Section/subsection/paragraph numbers are in child <Label> elements, NOT attributes.
  - Subsection labels include parentheses: "(1)" -> strip to "1".
  - Paragraph labels include parentheses: "(a)" -> strip to "a".
  - <HistoricalNote> must be excluded from chunk text.
  - <ContinuedSectionSubsection> is continuation prose after a paragraph list.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from lxml import etree

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))

TOKEN_MULTIPLIER = 1.3
SUBSECTION_TOKEN_LIMIT = 400

LIMS_NS = "http://justice.gc.ca/lims"

# Tags whose text should be excluded from all chunks
EXCLUDED_TAGS = {"HistoricalNote", "HistoricalNoteSubItem", "Marginal Note", "ReaderNote"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_ns(tag: str) -> str:
    """Remove Clark-notation namespace prefix from a tag name."""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def local(node) -> str:
    return strip_ns(node.tag)


def find_children(node, *local_names: str):
    """Yield direct children whose local name is in local_names."""
    for child in node:
        if local(child) in local_names:
            yield child


def find_first_child(node, *local_names: str):
    for child in node:
        if local(child) in local_names:
            return child
    return None


def get_label_text(node) -> str:
    """Get the text from the <Label> child of node, stripped of whitespace and parens."""
    label_el = find_first_child(node, "Label")
    if label_el is None:
        return ""
    text = (label_el.text or "").strip()
    # Remove surrounding parentheses: "(1)" -> "1", "(a)" -> "a"
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    return text


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * TOKEN_MULTIPLIER)


def sha256_hex(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


_FRACTION_MAP = {
    "½": "one-half",   # ½
    "⅓": "one-third",  # ⅓
    "⅔": "two-thirds", # ⅔
    "¼": "one-quarter", # ¼
    "¾": "three-quarters", # ¾
}
_FRACTION_TABLE = str.maketrans(_FRACTION_MAP)


def collect_text(node, exclude_tags=frozenset(EXCLUDED_TAGS)) -> str:
    """
    Recursively collect all text under node, skipping excluded tags.
    Collapses whitespace. Normalises Unicode fraction characters to words
    so that BM25 tokenisation can match them.
    """
    parts = []
    _collect_text_into(node, parts, exclude_tags)
    raw = " ".join(" ".join(parts).split())
    return raw.translate(_FRACTION_TABLE)


def _collect_text_into(node, parts: list, exclude_tags: frozenset):
    tag = local(node)
    if tag in exclude_tags:
        return
    if node.text:
        parts.append(node.text)
    for child in node:
        _collect_text_into(child, parts, exclude_tags)
        if child.tail:
            parts.append(child.tail)


def build_citation(
    section: str,
    subsection: Optional[str] = None,
    paragraph: Optional[str] = None,
    subparagraph: Optional[str] = None,
) -> str:
    cit = f"ITA s.{section}"
    if subsection:
        cit += f"({subsection})"
    if paragraph:
        cit += f"({paragraph})"
    if subparagraph:
        cit += f"({subparagraph})"
    return cit


_PARA_TOPIC_PATTERNS = [
    (re.compile(r"\bprincipal residence\b", re.IGNORECASE), "principal residence"),
    (re.compile(r"\brent[s,]?\s+royalt|\broyalt", re.IGNORECASE), "rents and royalties"),
    (re.compile(r"\binterest\b.{0,60}\bborrow|\bborrow.{0,60}\binterest", re.IGNORECASE), "interest on borrowed money"),
    (re.compile(r"\bnet capital loss", re.IGNORECASE), "net capital losses"),
    (re.compile(r"\ballowable capital loss", re.IGNORECASE), "allowable capital losses"),
    (re.compile(r"\btaxable capital gain", re.IGNORECASE), "taxable capital gains"),
    # RRSP maturity — s.146(2)(b.4) says "the plan matures not later than December 31
    # of the calendar year in which the annuitant attains 71 years of age".
    # Hint bridges the query "RRSP converted/matured" to ITA vocabulary "matures/maturity".
    (re.compile(r"\bplan matures|maturity date.{0,80}retirement savings|retirement savings.{0,80}maturity\b", re.IGNORECASE), "RRSP maturity date retirement income options"),
    # RRSP over-contribution tax — s.204.1 imposes tax on cumulative excess RRSP amounts.
    # Hint distinguishes s.204.1 from s.204.2 ("Cumulative excess amounts" formula) and
    # s.207.01 (TFSA over-contribution, similar structure).
    (re.compile(r"\bcumulative excess amount.{0,80}registered retirement savings|registered retirement savings.{0,80}cumulative excess amount", re.IGNORECASE), "RRSP over-contribution cumulative excess tax payable"),
]


def _para_topic_hints(text: str) -> str:
    """Return a short topic string for known ITA concepts found in paragraph text."""
    seen = []
    for pat, label in _PARA_TOPIC_PATTERNS:
        if pat.search(text) and label not in seen:
            seen.append(label)
    return " | ".join(seen)


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
# Subsection/paragraph chunking
# ---------------------------------------------------------------------------

def _paragraph_text(para_node) -> str:
    """
    Extract text from a <Paragraph> or <Subparagraph>/<Clause> node.
    Recursively includes sub-levels with their labels.
    """
    label = get_label_text(para_node)
    # Collect Text children directly
    text_parts = []
    for child in para_node:
        tag = local(child)
        if tag == "Text":
            text_parts.append(collect_text(child))
        elif tag in ("Subparagraph", "Clause", "Subclause"):
            sub_label = get_label_text(child)
            sub_text = _paragraph_text(child)
            if sub_label and sub_text:
                text_parts.append(f"({sub_label}) {sub_text}")
            elif sub_text:
                text_parts.append(sub_text)
    if not text_parts:
        text_parts.append(collect_text(para_node))
    combined = " ".join(t for t in text_parts if t)
    if label:
        return f"({label}) {combined}"
    return combined


def _chunks_from_definitions_subsection(
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
    Process a subsection whose children are <Definition> elements (e.g., ITA s.248(1)).
    Produces one chunk per defined term rather than one giant blob.
    """
    chunks = []
    term_tag = "DefinedTermEn" if language == "en" else "DefinedTermFr"

    for child in sub_node:
        if local(child) != "Definition":
            continue

        # Extract term name from <Text><DefinedTermEn|Fr>…</Text>
        term = ""
        text_el = find_first_child(child, "Text")
        if text_el is not None:
            te = find_first_child(text_el, term_tag)
            if te is None:
                te = find_first_child(text_el, "DefinedTermEn", "DefinedTermFr")
            if te is not None:
                term = (te.text or "").strip()

        full_text = collect_text(child)
        if not full_text.strip():
            continue

        # Include the section citation in def_context so query expansion can
        # specifically target s.248(1) general definitions vs. local definitions
        # in other sections that also appear as "Definition of 'X' [...]".
        def_cite = build_citation(section_label, sub_label)
        def_context = f"Definition of '{term}' [{context_prefix}] [{def_cite}]" if term else context_prefix
        chunks.append(
            make_chunk(
                text=full_text.strip(),
                context_prefix=def_context,
                citation=build_citation(section_label, sub_label),
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
        )

    return chunks


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

    If total tokens <= SUBSECTION_TOKEN_LIMIT, one chunk for the whole subsection.
    Otherwise, split at <Paragraph> boundaries.
    """
    # Use subsection-level MarginalNote when available — it provides more specific
    # context than the section title for sections like s.146 ("Definitions") or
    # s.212 ("Tax") where each subsection covers a distinct topic.
    sub_marg = find_first_child(sub_node, "MarginalNote")
    sub_marg_text = collect_text(sub_marg).strip() if sub_marg is not None else ""
    citation_for_context = build_citation(section_label, sub_label)
    if sub_marg_text:
        effective_context = f"{sub_marg_text} [{citation_for_context}]"
    else:
        effective_context = context_prefix

    # Definitions subsection: delegate each <Definition> to its own chunk
    if any(local(c) == "Definition" for c in sub_node):
        return _chunks_from_definitions_subsection(
            sub_node=sub_node,
            sub_label=sub_label,
            section_label=section_label,
            context_prefix=effective_context,
            part_label=part_label,
            division_label=division_label,
            language=language,
            valid_from=valid_from,
            amending_act=amending_act,
            act=act,
        )

    # Intro text: <Text> children before any <Paragraph>
    intro_parts = []
    paragraphs = []
    continuation_parts = []

    for child in sub_node:
        tag = local(child)
        if tag == "Label":
            continue
        elif tag == "MarginalNote":
            continue  # already captured in effective_context above
        elif tag == "Text" and not paragraphs:
            intro_parts.append(collect_text(child))
        elif tag == "Paragraph":
            paragraphs.append(child)
        elif tag == "ContinuedSectionSubsection":
            for text_child in find_children(child, "Text"):
                continuation_parts.append(collect_text(text_child))
        elif tag == "HistoricalNote":
            continue

    intro_text = " ".join(intro_parts)
    continuation_text = " ".join(continuation_parts)

    if not paragraphs:
        text = " ".join(t for t in [intro_text, continuation_text] if t) or collect_text(sub_node)
        if not text.strip():
            return []
        citation = build_citation(section_label, sub_label)
        return [make_chunk(
            text=text.strip(),
            context_prefix=effective_context,
            citation=citation,
            act=act, part=part_label, division=division_label,
            section=section_label, subsection=sub_label, paragraph=None,
            language=language, valid_from=valid_from, amending_act=amending_act,
        )]

    # Check if whole subsection fits in one chunk
    para_texts = {get_label_text(p): _paragraph_text(p) for p in paragraphs}
    all_text = " ".join(t for t in [intro_text, *para_texts.values(), continuation_text] if t)

    if estimate_tokens(all_text) <= SUBSECTION_TOKEN_LIMIT:
        citation = build_citation(section_label, sub_label)
        return [make_chunk(
            text=all_text.strip(),
            context_prefix=effective_context,
            citation=citation,
            act=act, part=part_label, division=division_label,
            section=section_label, subsection=sub_label, paragraph=None,
            language=language, valid_from=valid_from, amending_act=amending_act,
        )]

    # Split at paragraph boundaries; attach continuation to last paragraph.
    # Non-first paragraphs get a truncated intro so preamble keywords (e.g.
    # "non-resident", "deducted") are present in their chunk text for BM25/CE.
    _INTRO_TRUNCATE_WORDS = 40
    intro_truncated = intro_text
    if intro_text:
        words = intro_text.split()
        if len(words) > _INTRO_TRUNCATE_WORDS:
            intro_truncated = " ".join(words[:_INTRO_TRUNCATE_WORDS]) + "…"

    chunks = []
    items = list(para_texts.items())
    for i, (p_label, p_text) in enumerate(items):
        is_first = i == 0
        is_last = i == len(items) - 1
        parts = []
        if intro_text:
            parts.append(intro_text if is_first else intro_truncated)
        parts.append(p_text)
        if is_last and continuation_text:
            parts.append(continuation_text)
        combined = " ".join(p for p in parts if p).strip()
        if not combined:
            continue
        citation = build_citation(section_label, sub_label, p_label)
        hints = _para_topic_hints(combined)
        para_context = (effective_context + " | " + hints) if hints else effective_context
        chunks.append(make_chunk(
            text=combined,
            context_prefix=para_context,
            citation=citation,
            act=act, part=part_label, division=division_label,
            section=section_label, subsection=sub_label, paragraph=p_label,
            language=language, valid_from=valid_from, amending_act=amending_act,
        ))

    return chunks


# ---------------------------------------------------------------------------
# Core parser: flat walk of <Body>
# ---------------------------------------------------------------------------

def parse_ita_xml(
    xml_path: Path,
    language: str = "en",
    valid_from: str = "2024-01-01",
    amending_act: str = "",
) -> list[dict]:
    """
    Parse the ITA XML and return a flat list of chunk dicts.

    Walks <Body> children sequentially, tracking Part/Division context from
    <Heading> elements and emitting chunks for every <Section>.
    """
    print(f"[parse] Reading {xml_path} ...")
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    # Find <Body> (may be namespaced)
    body = find_first_child(root, "Body", "Corps")
    if body is None:
        # Some versions nest Body inside another element; fall back to root
        for el in root.iter():
            if local(el) in ("Body", "Corps"):
                body = el
                break
    if body is None:
        print("[parse] WARNING: Could not find <Body> element; using root.", file=sys.stderr)
        body = root

    chunks: list[dict] = []
    section_count = 0
    subsection_count = 0
    warning_count = 0

    current_part = ""
    current_division = ""

    for child in body:
        tag = local(child)

        if tag == "Heading":
            level = child.get("level", "")
            label_el = find_first_child(child, "Label")
            label_text = (label_el.text or "").strip() if label_el is not None else ""
            title_el = find_first_child(child, "TitleText")
            title_text = (title_el.text or "").strip() if title_el is not None else ""

            if level == "1":
                current_part = label_text or title_text
                current_division = ""  # reset division when part changes
            elif level == "2":
                current_division = label_text or title_text
            # level 3 = sub-division (e.g. "Basic Rules") — no label, not tracked

        elif tag == "Section":
            sec_label = get_label_text(child)
            if not sec_label:
                warning_count += 1
                continue

            marg = find_first_child(child, "MarginalNote")
            marginal_text = collect_text(marg).strip() if marg is not None else ""

            breadcrumb_parts = []
            if current_part:
                p = current_part.upper()
                breadcrumb_parts.append(current_part if p.startswith("PART") else f"Part {current_part}")
            if current_division:
                d = current_division.upper()
                breadcrumb_parts.append(current_division if d.startswith("DIVISION") else f"Division {current_division}")
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

            subsections = list(find_children(child, "Subsection", "Paragraphe"))

            if not subsections:
                # Flat section: collect all non-heading, non-historical text
                text_parts = []
                for el in child:
                    tag2 = local(el)
                    if tag2 in ("MarginalNote", "Label", "HistoricalNote"):
                        continue
                    text_parts.append(collect_text(el))
                text = " ".join(t for t in text_parts if t).strip()
                if not text:
                    continue
                citation = build_citation(sec_label)
                chunks.append(make_chunk(
                    text=text,
                    context_prefix=context_prefix,
                    citation=citation,
                    act="ITA",
                    part=current_part,
                    division=current_division,
                    section=sec_label,
                    subsection=None,
                    paragraph=None,
                    language=language,
                    valid_from=valid_from,
                    amending_act=amending_act,
                ))
                subsection_count += 1
                continue

            for sub_node in subsections:
                sub_label = get_label_text(sub_node)
                new_chunks = chunks_from_subsection(
                    sub_node=sub_node,
                    sub_label=sub_label,
                    section_label=sec_label,
                    context_prefix=context_prefix,
                    part_label=current_part,
                    division_label=current_division,
                    language=language,
                    valid_from=valid_from,
                    amending_act=amending_act,
                )
                chunks.extend(new_chunks)
                subsection_count += 1

            if section_count % 50 == 0:
                print(f"[parse]   ... {section_count} sections, {len(chunks)} chunks so far")

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
    parser.add_argument("--language", default="en", choices=["en", "fr"])
    parser.add_argument("--valid-from", default="2024-01-01")
    parser.add_argument("--amending-act", default="")
    parser.add_argument("--out", type=Path, default=None, help="JSON output path for sample inspection")
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
        print(f"\n[parse] First chunk:")
        if result:
            print(json.dumps(result[0], indent=2, ensure_ascii=False))
        print(f"\n[parse] Last chunk:")
        if result:
            print(json.dumps(result[-1], indent=2, ensure_ascii=False))
