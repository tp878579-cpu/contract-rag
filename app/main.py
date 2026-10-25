# app/main.py
"""Run with:  python -m streamlit run app/main.py"""

import sys
from pathlib import Path

# Streamlit only puts app/ on the import path, so add the project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components.chat_ui import render_chat  # noqa: E402
from app.components.upload import render_sidebar  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

st.set_page_config(page_title="Contract Analyzer", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading models (first run takes a while)...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def main() -> None:
    st.title("📄 Contract Analyzer")
    st.caption("Ask questions about your contracts. Every answer cites its source clause.")
    pipeline = get_pipeline()
    options = render_sidebar(pipeline)
    render_chat(pipeline, options)


main()