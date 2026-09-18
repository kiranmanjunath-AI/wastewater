"""
mcp/server.py
MCP (Model Context Protocol) server wrapping the /retrieve endpoint.

Exposes one tool: search_canadian_tax_legislation
  — optimised for LLM tool use; returns verbatim cited passages that should
    be quoted directly from the return value rather than paraphrased.

Start with:
    python -m mcp.server
from the legislative-retrieval/ directory.

Requires the FastAPI server to be running (or calls retriever.search directly).
"""

from __future__ import annotations

import os
from datetime import date
from typing import Optional

from dotenv import load_dotenv
from mcp.server import FastMCP
from mcp.server.models import InitializationOptions

load_dotenv()

# Use the retriever module directly (no HTTP round-trip required when running
# in the same process or alongside the API).
# If you want HTTP-based separation, swap for requests.post calls to /retrieve.
from api.retriever import search, get_consolidated_as_of

mcp = FastMCP(
    name="canadian-tax-legislation",
    instructions=(
        "This server gives you exact, cited passages from the Canadian Income Tax Act (ITA). "
        "Use search_canadian_tax_legislation to look up provisions. "
        "IMPORTANT: always cite the returned `citation` field verbatim (e.g. 'ITA s.20(1)(c)') "
        "when quoting the retrieved text — do not paraphrase or omit the citation. "
        "The `text` field is the verbatim statutory passage and should be quoted directly. "
        "If `valid_to` is non-null, the provision may have been repealed — check the date. "
        "Use `as_of` to retrieve the version of the ITA applicable on a specific date."
    ),
)


@mcp.tool()
def search_canadian_tax_legislation(
    query: str,
    top_k: int = 5,
    language: str = "en",
    as_of: Optional[str] = None,
) -> dict:
    """
    Search the Canadian Income Tax Act for provisions relevant to the query.

    Returns exact statutory passages with precise ITA citations (e.g. "ITA s.20(1)(c)").
    These are verbatim passages from the consolidated ITA — quote them directly,
    including the citation, rather than summarising.

    Args:
        query:    Natural-language question or keyword string, e.g.
                  "interest deduction on borrowed money" or
                  "taxable capital gain inclusion rate"
        top_k:    Number of results to return (1–20, default 5).
        language: "en" for English provisions, "fr" for French (default "en").
        as_of:    ISO date string (YYYY-MM-DD) for the version of the ITA to
                  search. Defaults to today. Use this to research historical
                  provisions, e.g. as_of="2022-01-01".

    Returns:
        {
          "query": str,
          "as_of": str,
          "consolidated_as_of": str | null,   // date of most recent ingested version
          "results": [
            {
              "citation": "ITA s.20(1)(c)",   // ALWAYS cite this verbatim
              "text": "...",                  // verbatim statutory text; quote directly
              "score": float,                 // cosine similarity (0–1)
              "language": "en",
              "valid_from": "YYYY-MM-DD",
              "valid_to": null | "YYYY-MM-DD",
              "amending_act": str | null,
            }
          ]
        }
    """
    effective_as_of = as_of or date.today().isoformat()

    try:
        raw_results = search(
            query=query,
            top_k=max(1, min(top_k, 20)),
            language=language,
            as_of=effective_as_of,
        )
    except Exception as exc:
        return {
            "error": (
                f"Retrieval failed. Ensure Qdrant is running and the ITA has been ingested. "
                f"Run: docker-compose up -d && python -m ingestion.load_qdrant\n"
                f"Error: {exc}"
            ),
            "query": query,
            "as_of": effective_as_of,
            "results": [],
        }

    consolidated = get_consolidated_as_of()

    return {
        "query": query,
        "as_of": effective_as_of,
        "consolidated_as_of": consolidated,
        "results": [
            {
                "citation": r["citation"],
                "text": r["text"],
                "score": r["score"],
                "language": r["language"],
                "valid_from": r["valid_from"],
                "valid_to": r["valid_to"],
                "amending_act": r.get("amending_act"),
            }
            for r in raw_results
        ],
    }


if __name__ == "__main__":
    import asyncio

    print("[mcp] Starting Canadian Tax Legislation MCP server ...")
    mcp.run()
