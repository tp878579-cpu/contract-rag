# app/components/chat_ui.py
"""The chat area."""

import streamlit as st

from app.components.citation_viewer import render_answer_details
from src.pipeline import RAGAnswer, RAGPipeline

USER_AVATAR = "🧑‍💼"
BOT_AVATAR = "📄"

EXAMPLES = [
    "How can I end the contract?",
    "When are fees due?",
    "How long does the agreement last?",
    "What is the limit of liability?",
]


def _show_answer(result: RAGAnswer) -> None:
    if result.refused:
        st.info(result.answer, icon="🔎")
    else:
        st.markdown(result.answer)
    render_answer_details(result)


def _render_welcome() -> None:
    st.markdown('<div class="welcome-title">👋 What would you like to know?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="welcome-sub">Try one of these, or type your own question below.</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        if cols[i % 2].button(q, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending = q
            st.rerun()


def render_chat(pipeline: RAGPipeline, options: dict) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    typed = st.chat_input("Ask a question about your contracts...")
    question = typed or st.session_state.pop("pending", None)

    if not st.session_state.messages and not question:
        if pipeline.store.count() == 0:
            st.info("No documents yet. Upload a contract in the sidebar to get started.", icon="📤")
        else:
            _render_welcome()

    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar=USER_AVATAR if m["role"] == "user" else BOT_AVATAR):
            if m.get("result"):
                _show_answer(m["result"])
            else:
                st.markdown(m["content"])

    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        if pipeline.store.count() == 0:
            st.info("No documents yet. Upload a contract in the sidebar first.", icon="📤")
            return

        with st.spinner("Searching the documents..."):
            try:
                result = pipeline.ask(question, **options)
            except Exception as e:
                msg = str(e)
                if "503" in msg or "429" in msg:
                    st.warning("The AI model is busy right now. Please wait a minute and ask again.", icon="⏳")
                else:
                    st.error(f"Something went wrong: {msg}")
                return

        _show_answer(result)

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "result": result}
    )