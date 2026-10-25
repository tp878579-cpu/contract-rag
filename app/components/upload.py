# app/components/upload.py
"""Sidebar: upload documents, pick a scope, choose retrieval settings."""

from pathlib import Path

import streamlit as st

from src.config import settings
from src.pipeline import RAGPipeline


def render_sidebar(pipeline: RAGPipeline) -> dict:
    if "ingested" not in st.session_state:
        st.session_state.ingested = set()

    with st.sidebar:
        st.header("Documents")
        files = st.file_uploader(
            "Upload contracts", type=["pdf", "docx", "txt"], accept_multiple_files=True
        )

        for f in files or []:
            key = (f.name, f.size)
            if key in st.session_state.ingested:
                continue
            path = settings.raw_dir / Path(f.name).name
            path.write_bytes(f.getbuffer())
            with st.spinner(f"Indexing {f.name}..."):
                try:
                    n = pipeline.ingest_file(path)
                    st.success(f"{f.name}: {n} chunks indexed")
                except Exception as e:
                    st.error(f"Could not read {f.name}: {e}")
            st.session_state.ingested.add(key)

        sources = pipeline.store.list_sources()
        st.caption(f"{len(sources)} document(s), {pipeline.store.count()} chunks stored")

        for name in sources:
            col1, col2 = st.columns([4, 1])
            col1.write(name)
            if col2.button("🗑", key=f"del_{name}", help=f"Remove {name}"):
                pipeline.remove_file(name)
                st.rerun()

        st.divider()
        st.header("Settings")
        scope = st.selectbox("Search in", ["All documents", *sources])
        mode = st.selectbox("Retrieval mode", ["hybrid", "vector", "bm25"])
        use_reranker = st.checkbox("Use reranker", value=settings.use_reranker)

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    return {
        "source": None if scope == "All documents" else scope,
        "mode": mode,
        "use_reranker": use_reranker,
    }