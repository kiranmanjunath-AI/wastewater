"""
Central Bank RAG — Streamlit demo.

Covers US Federal Reserve and Bank of Canada documents from Jan 2025 – Sep 2026.
Run: streamlit run app.py
Requires: .env with VOYAGE_API_KEY + ANTHROPIC_API_KEY, and a built index.
"""

import os
import sys
from datetime import date
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.embed.indexer import BM25_FILE, CHROMA_DIR
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import RetrievalResult

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Central Bank RAG",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Style ──────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .fed-badge {
        background: #1565C0; color: white;
        padding: 2px 8px; border-radius: 4px;
        font-size: 0.75em; font-weight: 700; letter-spacing: 0.05em;
    }
    .boc-badge {
        background: #B71C1C; color: white;
        padding: 2px 8px; border-radius: 4px;
        font-size: 0.75em; font-weight: 700; letter-spacing: 0.05em;
    }
    .source-meta { color: #666; font-size: 0.82em; }
    div[data-testid="stExpander"] { border: 1px solid #e0e0e0; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────────

FED_DOC_TYPES = [
    "FOMC Statement",
    "FOMC Minutes",
    "Summary of Economic Projections",
    "Beige Book",
    "Monetary Policy Report",
]
BOC_DOC_TYPES = [
    "Rate Decision Statement",
    "Monetary Policy Report",
    "Governing Council Deliberations",
]
ALL_DOC_TYPES = sorted(set(FED_DOC_TYPES + BOC_DOC_TYPES))

EXAMPLE_QUERIES = [
    "What is the current policy rate and what's the near-term outlook?",
    "How does the Bank of Canada's inflation view compare to the Fed's?",
    "What are the FOMC's GDP and inflation projections for 2026?",
    "What does the Beige Book say about the labour market?",
    "How has the language around rate cuts evolved since early 2025?",
]

# ── Pipeline (loaded once, cached across reruns) ───────────────────────────────

@st.cache_resource(show_spinner="Loading indexes — this takes ~10 seconds on first load…")
def _load_pipeline() -> RAGPipeline:
    return RAGPipeline()


# ── Source card renderer ───────────────────────────────────────────────────────

def _badge(institution: str) -> str:
    if institution == "Federal Reserve":
        return '<span class="fed-badge">FED</span>'
    return '<span class="boc-badge">BoC</span>'


def _render_sources(sources: list[RetrievalResult]) -> None:
    if not sources:
        return

    st.markdown("---")
    st.markdown(f"**{len(sources)} sources retrieved**")

    cols = st.columns(min(len(sources), 3))
    for i, r in enumerate(sources):
        c = r.chunk
        col = cols[i % len(cols)]
        with col:
            ranks = []
            if r.dense_rank:
                ranks.append(f"dense #{r.dense_rank}")
            if r.sparse_rank:
                ranks.append(f"sparse #{r.sparse_rank}")
            rank_str = " · ".join(ranks)

            with st.expander(
                f"[{i+1}] {c.doc_type} · {c.date}",
                expanded=False,
            ):
                st.markdown(
                    f"{_badge(c.institution)}&nbsp; **{c.title}**",
                    unsafe_allow_html=True,
                )
                st.caption(rank_str)
                st.markdown(
                    c.text[:450] + ("…" if len(c.text) > 450 else ""),
                )
                if c.url:
                    st.markdown(f"[Open source ↗]({c.url})")


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏦 Central Bank RAG")
    st.caption("Fed · Bank of Canada · Jan 2025 – Sep 2026")
    st.divider()

    # ── Filters (optional — collapsed by default) ──
    with st.expander("Filters", expanded=False):
        institution_choice = st.radio(
            "Institution",
            ["All", "Federal Reserve", "Bank of Canada"],
            horizontal=True,
        )

        doc_type_choice = st.selectbox(
            "Document type",
            ["All"] + ALL_DOC_TYPES,
        )

        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input(
                "From", value=date(2025, 1, 1),
                min_value=date(2025, 1, 1), max_value=date(2026, 9, 30),
            )
        with col2:
            date_to = st.date_input(
                "To", value=date(2026, 9, 30),
                min_value=date(2025, 1, 1), max_value=date(2026, 9, 30),
            )

        top_n = st.slider("Sources per answer", min_value=1, max_value=15, value=6)

    institution_filter = None if institution_choice == "All" else institution_choice
    doc_type_filter    = None if doc_type_choice == "All" else doc_type_choice

    st.divider()

    # ── Example queries ──
    st.markdown("### Try these")
    for q in EXAMPLE_QUERIES:
        if st.button(q, use_container_width=True, key=f"ex_{q[:30]}"):
            st.session_state["_draft"] = q

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Prerequisite check ─────────────────────────────────────────────────────────

if not (BM25_FILE.exists() and CHROMA_DIR.exists()):
    st.error(
        "**Index not built.**  \n\n"
        "1. Copy `.env.example` → `.env` and add your API keys  \n"
        "2. Run:  \n"
        "```bash\npython scripts/build_index.py\n```"
    )
    st.stop()

# ── Load pipeline ──────────────────────────────────────────────────────────────

try:
    pipeline = _load_pipeline()
except Exception as exc:
    st.error(f"**Failed to load pipeline:** {exc}")
    st.stop()

# ── Session state ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render chat history ────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            _render_sources(msg["sources"])

# ── Welcome screen ─────────────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown(
        "## Ask anything about Fed or BoC monetary policy\n\n"
        "Interest rates · Inflation outlook · GDP projections · "
        "Policy divergence · Sentiment shifts\n\n"
        "Pick an example from the sidebar or type your own question below."
    )

# ── Draft preview (example query selected from sidebar) ───────────────────────

draft = st.session_state.get("_draft", "")
if draft:
    st.info("**Example selected** — edit if you like, then press Send.")
    edited = st.text_area(
        "Query", value=draft, height=80,
        label_visibility="collapsed", key="_draft_edit",
    )
    col_send, col_cancel = st.columns([1, 5])
    with col_send:
        if st.button("Send ↵", type="primary"):
            st.session_state["_ready_prompt"] = edited
            del st.session_state["_draft"]
            st.rerun()
    with col_cancel:
        if st.button("Cancel"):
            del st.session_state["_draft"]
            st.rerun()

# ── Chat input (always visible) ────────────────────────────────────────────────

typed = st.chat_input(
    "e.g. What is the current policy rate and what's the near-term outlook?"
)

# Priority: draft send > typed input
prompt = st.session_state.pop("_ready_prompt", None) or typed

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": None})

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving sources…"):
                token_stream, sources = pipeline.stream_query(
                    prompt,
                    top_n=top_n,
                    institution=institution_filter,
                    doc_type=doc_type_filter,
                    date_from=str(date_from),
                    date_to=str(date_to),
                )
            answer = st.write_stream(token_stream)
            _render_sources(sources)
        except Exception as exc:
            answer = f"⚠️ Error: {exc}"
            st.error(answer)
            sources = []

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
