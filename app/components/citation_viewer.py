# app/components/citation_viewer.py
"""Show citations, warnings and stats under an answer."""

import streamlit as st

from src.pipeline import RAGAnswer


def render_answer_details(result: RAGAnswer) -> None:
    for w in result.warnings:
        st.warning(w)

    if result.citations:
        st.markdown("**Sources**")
        for c in result.citations:
            with st.expander(f"[{c.number}] {c.label}"):
                st.write(c.full_text)
                st.caption(f"Relevance score: {c.score:.2f}")

    if not result.refused:
        st.caption(
            f"{result.total_s:.1f}s total | retrieval {result.retrieval_s:.1f}s | "
            f"LLM {result.llm_s:.1f}s | {result.input_tokens} in / {result.output_tokens} out tokens | "
            f"mode: {result.mode}{' + rerank' if result.reranked else ''}"
        )