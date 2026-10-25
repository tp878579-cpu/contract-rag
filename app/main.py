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
from src.config import settings  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

st.set_page_config(
    page_title="Contract Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    css = (ROOT / "app" / "assets" / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models (first run takes a while)...")
def get_pipeline() -> RAGPipeline:
    pipeline = RAGPipeline()
    if pipeline.store.count() == 0:       # e.g. fresh Hugging Face Space
        for f in sorted(settings.raw_dir.iterdir()):
            if f.suffix.lower() in {".pdf", ".docx", ".txt"}:
                pipeline.ingest_file(f)
    return pipeline


def render_hero() -> None:
    st.markdown(
        '<div class="hero">'
        '<span class="hero-badge">HYBRID SEARCH · RERANKING · CITATIONS</span>'
        "<h1>📄 Contract Analyzer</h1>"
        "<p>Ask questions in plain English. Every answer points to the exact file, page and clause.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    load_css()
    render_hero()
    pipeline = get_pipeline()
    options = render_sidebar(pipeline)
    render_chat(pipeline, options)


main()