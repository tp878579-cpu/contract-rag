FROM python:3.12-slim

# Hugging Face Spaces runs containers as user 1000
RUN useradd -m -u 1000 user && mkdir -p /app && chown user /app
WORKDIR /app

COPY --chown=user requirements.txt .
# CPU-only torch keeps the image far smaller than the default build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user

ENV HF_HOME=/home/user/.cache/huggingface \
    PYTHONPATH=/app

# Download the embedding model at build time so the first request isn't slow
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.fileWatcherType=none"]