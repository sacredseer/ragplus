import logging

from utilities import call_llm

logger = logging.getLogger(__name__)

HARD_REJECT_THRESHOLD = 0.04
HARD_ACCEPT_THRESHOLD = 0.30


class ValidationAgent:
    """Validates whether retrieved chunks actually contain information relevant to the query."""

    def validate(self, query: str, chunks: list, retrieval_score: float) -> dict:
        """
        Determine if the retrieved content can answer the query.
        Returns: {"relevant": bool, "reason": str}
        """
        if not chunks or retrieval_score < HARD_REJECT_THRESHOLD:
            return {
                "relevant": False,
                "reason": "Retrieval score is too low; no useful content found in documents.",
            }

        if retrieval_score >= HARD_ACCEPT_THRESHOLD:
            return {
                "relevant": True,
                "reason": f"High-confidence retrieval (score {retrieval_score:.2f}).",
            }

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
            answer = call_llm(prompt).strip().upper()
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
            logger.warning("Validation LLM call failed (%s); falling back to score.", exc)
            return {
                "relevant": retrieval_score >= HARD_REJECT_THRESHOLD,
                "reason": f"Score-based fallback (score {retrieval_score:.2f}).",
            }
