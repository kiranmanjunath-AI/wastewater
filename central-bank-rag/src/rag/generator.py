"""
Answer generation using Claude Sonnet 5.

The system prompt is marked for prompt caching — on repeat queries the cached
prompt tokens cost ~10% of normal input price. Caching activates once the prompt
exceeds 1024 tokens (Sonnet minimum); the current system prompt is ~450 tokens so
it will be cached once a long sources block pushes the total over the threshold.
"""

import logging
import os
from collections.abc import Iterator
from typing import Optional

import anthropic

from ..embed.chunker import Chunk

log = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-sonnet-5"
MAX_TOKENS   = 4096

SYSTEM_PROMPT = """\
You are a specialized analyst of central bank communications from the \
US Federal Reserve and the Bank of Canada. You answer questions strictly \
from the numbered source documents provided in each query.

Document types you may encounter:
- FOMC Statements: Federal Reserve policy decisions and forward guidance
- FOMC Minutes: Detailed committee deliberations
- SEP (Summary of Economic Projections): FOMC dot-plot and macro forecasts
- Beige Book: Regional economic conditions from all 12 Fed Districts
- Federal Reserve Monetary Policy Report: Semi-annual Congress report
- Bank of Canada Rate Decision Statements: Official rate announcements
- Bank of Canada Monetary Policy Reports (MPR): Quarterly economic outlook
- Bank of Canada Governing Council Deliberations: Internal policy discussion summaries

Rules for every answer:
1. CITATIONS — cite every factual claim with [N] matching the source number.
2. NUMBERS — quote exact figures (rates, GDP %, CPI) with their date and source.
3. LANGUAGE — preserve precise central-bank phrasing: "broadly balanced", \
"data-dependent", "gradual pace". Do not paraphrase these away.
4. DIVERGENCE — when Fed and BoC differ in tone or policy path, call it out explicitly.
5. PROJECTIONS vs ACTUALS — distinguish forecasts from realised data.
6. TEMPORAL ORDER — when sources span multiple dates, note how the view evolved.
7. GAPS — if a key document type is absent from the sources (e.g. no FOMC \
Statement for the period in question), name the gap explicitly ("No FOMC \
Statement from [period] was retrieved; the following is based on minutes only"). \
Never draw on outside knowledge to fill it.

Format: use short prose paragraphs. Add a markdown heading only when the \
question has clearly distinct sub-parts. Calibrate length to the complexity \
of the question — simple lookups get one paragraph, multi-source comparisons \
get the space they need, but never pad.\
"""


# ── Formatting ─────────────────────────────────────────────────────────────────

def _format_sources(chunks: list[Chunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        header = f"[{i}] {c.institution} | {c.doc_type} | {c.date}"
        if c.page_num:
            header += f" | p.{c.page_num}"
        parts.append(f"{header}\nTitle: {c.title}\n\n{c.text}")
    return "\n\n---\n\n".join(parts)


def _build_messages(query: str, chunks: list[Chunk]) -> list[dict]:
    sources = _format_sources(chunks)
    return [
        {
            "role": "user",
            "content": (
                f"Sources:\n\n{sources}\n\n"
                f"{'─' * 60}\n\n"
                f"Question: {query}"
            ),
        }
    ]


def _system_block() -> list[dict]:
    return [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]


# ── Public API ─────────────────────────────────────────────────────────────────

def stream_generate(query: str, chunks: list[Chunk]) -> Iterator[str]:
    """Yield answer tokens one by one (for Streamlit write_stream)."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=_system_block(),
        messages=_build_messages(query, chunks),
    ) as s:
        yield from s.text_stream


def generate(query: str, chunks: list[Chunk]) -> str:
    """Return the full answer as a string (non-streaming)."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=_system_block(),
        messages=_build_messages(query, chunks),
    )
    usage = resp.usage
    log.debug(
        "Tokens: input=%d output=%d cache_read=%s cache_created=%s",
        usage.input_tokens,
        usage.output_tokens,
        getattr(usage, "cache_read_input_tokens", "—"),
        getattr(usage, "cache_creation_input_tokens", "—"),
    )
    return resp.content[0].text
