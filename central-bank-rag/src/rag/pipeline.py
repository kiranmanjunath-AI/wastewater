"""
Thin orchestrator: retrieval → generation.

Usage (non-streaming):
    pipeline = RAGPipeline()
    response = pipeline.query("What is the Fed's current rate?")
    print(response.answer)
    for src in response.sources:
        print(src.chunk.title, src.chunk.date)

Usage (streaming, for Streamlit):
    tokens, sources = pipeline.stream_query("Compare Fed and BoC outlooks")
    for token in tokens:
        print(token, end="", flush=True)
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Optional

from .retriever import Retriever, RetrievalResult
from .generator import generate, stream_generate

log = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    query:   str
    answer:  str
    sources: list[RetrievalResult]


class RAGPipeline:

    def __init__(self) -> None:
        self.retriever = Retriever()

    def query(
        self,
        question:    str,
        *,
        top_n:       int           = 6,
        institution: Optional[str] = None,
        doc_type:    Optional[str] = None,
        date_from:   Optional[str] = None,
        date_to:     Optional[str] = None,
    ) -> RAGResponse:
        """Retrieve + generate, return everything at once."""
        results = self.retriever.query(
            question,
            top_n=top_n,
            institution=institution,
            doc_type=doc_type,
            date_from=date_from,
            date_to=date_to,
        )
        answer = generate(question, [r.chunk for r in results])
        return RAGResponse(query=question, answer=answer, sources=results)

    def stream_query(
        self,
        question:    str,
        *,
        top_n:       int           = 6,
        institution: Optional[str] = None,
        doc_type:    Optional[str] = None,
        date_from:   Optional[str] = None,
        date_to:     Optional[str] = None,
    ) -> tuple[Iterator[str], list[RetrievalResult]]:
        """
        Retrieve synchronously, then return a token stream and the source list.
        Sources are available immediately; the stream is consumed by the caller.
        """
        results = self.retriever.query(
            question,
            top_n=top_n,
            institution=institution,
            doc_type=doc_type,
            date_from=date_from,
            date_to=date_to,
        )
        token_stream = stream_generate(question, [r.chunk for r in results])
        return token_stream, results
