# src/embeddings/embedder.py
"""Turn text into vectors using a local sentence-transformers model."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# BGE models work better when short search queries get this prefix.
# Documents are embedded WITHOUT it.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = settings.embedding_model):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Load the model on first use (it takes a few seconds)."""
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed chunks for storage."""
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   # unit length, so dot product = cosine similarity
            show_progress_bar=len(texts) > 50,
        )
        logger.info("Embedded %d text(s)", len(texts))
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a user question for searching."""
        vector = self.model.encode(
            QUERY_INSTRUCTION + query,
            normalize_embeddings=True,
        )
        return vector.tolist()


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Shared instance, so the model is loaded only once per process."""
    return Embedder()