# app/components/citation_viewer.py
"""Show citations, warnings and stats under an answer."""

import html

import streamlit as st

from src.pipeline import RAGAnswer


def render_answer_details(result: RAGAnswer) -> None:
    for w in result.warnings:
        st.warning(w, icon="⚠️")

    if result.citations:
        st.markdown('<div class="sources-title">📎 Sources</div>', unsafe_allow_html=True)
        for c in result.citations:
            with st.expander(f"[{c.number}]  {c.label}"):
                body = html.escape(c.full_text).replace("\n", "<br>")
                st.markdown(f'<div class="quote">{body}</div>', unsafe_allow_html=True)
                if result.reranked:
                    st.caption(f"Relevance: {c.score * 100:.0f}%")

    if not result.refused:
        chips = [
            f'<span class="chip">⏱ {result.total_s:.1f}s</span>',
            f'<span class="chip">🔍 {result.mode}</span>',
            f'<span class="chip">🔤 {result.input_tokens + result.output_tokens} tokens</span>',
        ]
        if result.reranked:
            chips.append('<span class="chip green">✓ reranked</span>')
        if result.citations:
            chips.append(f'<span class="chip green">{len(result.citations)} source(s)</span>')
        st.markdown(f'<div class="chip-row">{"".join(chips)}</div>', unsafe_allow_html=True)