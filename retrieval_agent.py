import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """
    TF-IDF based retrieval over provided document chunks.

    This implementation removes the dependency on ChromaDB and embeddings
    and performs an on-demand TF-IDF search across the supplied chunks.
    """

    def __init__(self, top_k: int = 6):
        self.top_k = top_k
        logger.info("RetrievalAgent ready — TF-IDF retrieval (no persistent store).")

    @property
    def count(self) -> int:
        """Number of chunks currently indexed — not applicable for TF-IDF on demand."""
        return 0

    def index(self, chunks: list) -> int:
        """No-op for indexing when embeddings/ChromaDB are removed.

        Returns 0 to indicate no embedding/indexing occurred.
        """
        logger.debug("Index called but embeddings removed; skipping indexing.")
        return 0

    def retrieve(self, query: str, chunks: list = None) -> tuple:
        """Find the top-k chunks most relevant to *query* using TF-IDF.

        Returns: (results: list[dict], max_score: float)
        """
        if not chunks:
            return [], 0.0

        return self._tfidf_retrieve(query, chunks)

    def clear(self) -> None:
        """No persistent store to clear in TF-IDF mode."""
        logger.debug("Clear called but no persistent store exists; skipping.")

    def _tfidf_retrieve(self, query: str, chunks: list) -> tuple:
        texts = [c.get("content", "") for c in chunks]
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=10_000)
            matrix = vec.fit_transform(texts)
            q_vec = vec.transform([query])
            scores = cosine_similarity(q_vec, matrix).flatten()
        except Exception as exc:
            logger.error("TF-IDF retrieval error: %s", exc)
            return [], 0.0

        top_indices = scores.argsort()[::-1][: self.top_k]
        results = []
        for idx in top_indices:
            entry = chunks[idx].copy()
            entry["score"] = float(scores[idx])
            results.append(entry)

        max_score = float(scores.max()) if len(scores) > 0 else 0.0
        logger.debug("TF-IDF retrieved %d chunk(s) (max_score=%.3f).", len(results), max_score)
        return results, max_score
