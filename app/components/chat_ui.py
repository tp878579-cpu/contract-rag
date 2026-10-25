# app/components/chat_ui.py
"""The chat area."""

import streamlit as st

from app.components.citation_viewer import render_answer_details
from src.pipeline import RAGPipeline


def render_chat(pipeline: RAGPipeline, options: dict) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("result"):
                render_answer_details(m["result"])

    question = st.chat_input("Ask a question about your contracts...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if pipeline.store.count() == 0:
            st.info("No documents yet. Upload a contract in the sidebar first.")
            return

        with st.spinner("Searching the documents..."):
            try:
                result = pipeline.ask(question, **options)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                return

        st.markdown(result.answer)
        render_answer_details(result)

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "result": result}
    )