from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RetrievalAgent:
    """
    Handles TF-IDF based retrieval over document chunks.
    
    To optimize performance, this agent fits the vectorizer and pre-computes
    document embeddings (TF-IDF matrix) at indexing time, avoiding expensive
    on-demand fitting during query execution.
    """

    def __init__(self, top_k: int = 6):
        self.top_k = top_k
        self.chunks = []
        self.vectorizer = None
        self.matrix = None

    @property
    def count(self) -> int:
        """Returns the number of indexed document chunks."""
        return len(self.chunks)

    def index(self, chunks: list) -> int:
        """
        Indexes new document chunks by extending the local database
        and rebuilding the TF-IDF vector space.
        """
        if not chunks:
            return 0

        self.chunks.extend(chunks)
        self._rebuild_index()
        return len(chunks)

    def retrieve(self, query: str, chunks: list = None) -> tuple:
        """
        Finds the top-k chunks matching the query using cosine similarity.
        Falls back to on-demand calculation if no cached index is available.
        """
        if chunks is not None and not self.chunks:
            return self._tfidf_retrieve_ondemand(query, chunks)

        if not self.chunks or self.matrix is None:
            return [], 0.0

        try:
            query_vector = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vector, self.matrix).flatten()
        except Exception as exc:
            return [], 0.0

        top_indices = scores.argsort()[::-1][: self.top_k]
        results = []
        for idx in top_indices:
            entry = self.chunks[idx].copy()
            entry["score"] = float(scores[idx])
            results.append(entry)

        max_score = float(scores.max()) if len(scores) > 0 else 0.0
        return results, max_score

    def clear(self) -> None:
        """Clears all indexed document chunks and resets the vectorizer."""
        self.chunks = []
        self.vectorizer = None
        self.matrix = None

    def _rebuild_index(self) -> None:
        """Fits the TF-IDF vectorizer and transforms all document chunks."""
        if not self.chunks:
            self.matrix = None
            self.vectorizer = None
            return

        texts = [chunk.get("content", "") for chunk in self.chunks]
        try:
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
            self.matrix = self.vectorizer.fit_transform(texts)
        except Exception as exc:
            self.matrix = None
            self.vectorizer = None

    def _tfidf_retrieve_ondemand(self, query: str, chunks: list) -> tuple:
        """Calculates TF-IDF on-the-fly for ad-hoc chunk lists."""
        texts = [c.get("content", "") for c in chunks]
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=10000)
            matrix = vec.fit_transform(texts)
            q_vec = vec.transform([query])
            scores = cosine_similarity(q_vec, matrix).flatten()
        except Exception as exc:
            return [], 0.0

        top_indices = scores.argsort()[::-1][: self.top_k]
        results = []
        for idx in top_indices:
            entry = chunks[idx].copy()
            entry["score"] = float(scores[idx])
            results.append(entry)

        max_score = float(scores.max()) if len(scores) > 0 else 0.0
        return results, max_score
