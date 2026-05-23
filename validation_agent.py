import logging

from utilities import call_llm

logger = logging.getLogger(__name__)

# Heuristics for skipping LLM validation depending on TF-IDF cosine similarity scores
HARD_REJECT_THRESHOLD = 0.04
HARD_ACCEPT_THRESHOLD = 0.30


class ValidationAgent:
    """Evaluates whether the retrieved document chunks contain context relevant to answering the query."""

    def validate(self, query: str, chunks: list, retrieval_score: float, config: dict = None) -> dict:
        """
        Determines if retrieved content is relevant. If the score is in the ambiguous middle range,
        it uses the LLM to perform semantic validation.
        
        Returns:
            dict: {"relevant": bool, "reason": str}
        """
        # Immediately reject retrieval results with extremely low keyword matching scores
        if not chunks or retrieval_score < HARD_REJECT_THRESHOLD:
            return {
                "relevant": False,
                "reason": "Retrieval score is too low; no useful content found in documents.",
            }

        # Accept very high scores directly without calling the LLM, saving latency
        if retrieval_score >= HARD_ACCEPT_THRESHOLD:
            return {
                "relevant": True,
                "reason": f"High-confidence retrieval (score {retrieval_score:.2f}).",
            }

        # Formulate prompt for the LLM to inspect the retrieved text snippets
        excerpt = "\n\n".join(c["content"][:300] for c in chunks[:3])
        prompt = (
            "You are a relevance validator.\n\n"
            f"Query: {query}\n\n"
            "Document excerpts:\n"
            f"{excerpt}\n\n"
            "Do the document excerpts contain information that can help answer the query?\n"
            "Answer with exactly one word: YES or NO."
        )

        try:
            answer = call_llm(prompt, config=config).strip().upper()
            relevant = answer.startswith("YES")
            return {
                "relevant": relevant,
                "reason": (
                    "LLM confirmed documents are relevant."
                    if relevant
                    else "LLM determined documents do not contain relevant information."
                ),
            }
        except Exception as exc:
            # Fallback to the similarity score in case LLM generation fails
            logger.warning("Validation LLM call failed (%s); falling back to score.", exc)
            return {
                "relevant": retrieval_score >= HARD_REJECT_THRESHOLD,
                "reason": f"Score-based fallback (score {retrieval_score:.2f}).",
            }
