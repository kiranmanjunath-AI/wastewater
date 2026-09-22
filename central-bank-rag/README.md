# Central Bank RAG

A retrieval-augmented generation (RAG) system for querying central bank
documents from the Federal Reserve and Bank of Canada.

## What it does

Ask natural language questions against a corpus of 82 central bank documents
spanning January 2025 to September 2026 — including rate decision statements,
meeting minutes, monetary policy reports, and the Beige Book.

## Stack

- **Scraping** — Fed and BoC document scrapers (`src/ingest/`)
- **Embeddings** — Voyage Finance-2 (domain-tuned for financial text)
- **Vector store** — ChromaDB (dense) + BM25 (sparse), merged with Reciprocal Rank Fusion
- **Frontend** — Streamlit
- **LLM** — Claude Sonnet 5 (Anthropic)

## Getting started

```bash
git clone https://github.com/yourhandle/central-bank-rag
cd central-bank-rag
pip install -r requirements.txt
cp .env.example .env
# Add VOYAGE_API_KEY and ANTHROPIC_API_KEY to .env
python scripts/download_docs.py
python scripts/build_index.py
streamlit run app.py
```

## Configuration

Copy `.env.example` to `.env` and fill in:

| Variable | Description |
|---|---|
| `VOYAGE_API_KEY` | Voyage AI key for embeddings |
| `ANTHROPIC_API_KEY` | Anthropic key for answer generation |

## Project structure

```
central-bank-rag/
├── app.py                  # Streamlit UI
├── scripts/
│   ├── download_docs.py    # Scrape Fed + BoC documents
│   └── build_index.py      # Chunk, embed, and index into ChromaDB + BM25
└── src/
    ├── ingest/             # Fed and BoC scrapers
    ├── embed/              # Chunker and indexer
    └── rag/                # Retriever, generator, pipeline
```

## Document coverage

| Institution | Document types |
|---|---|
| Federal Reserve | FOMC Statements, FOMC Minutes, SEP, Beige Book, Monetary Policy Report |
| Bank of Canada | Rate Decision Statements, Monetary Policy Reports, Governing Council Deliberations |
