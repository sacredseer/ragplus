import logging

from ..util.utilities import call_llm

logger = logging.getLogger(__name__)

HARD_REJECT_THRESHOLD = 0.04
HARD_ACCEPT_THRESHOLD = 0.30


class ValidationAgent:
    """Evaluates whether the retrieved document chunks contain context relevant to answering the query."""

    def _is_general_query(self, query: str) -> bool:
        q = query.lower().strip()
        indicators = [
            "summarize", "summary", "overview", "tldr", "tl;dr", 
            "what is this", "what is it", "main points", "key takeaways", 
            "explain this", "what does this say", "what's in this",
            "what is this document about", "what is the document about",
            "what is this file about", "what is the file about",
            "tell me about"
        ]
        doc_refs = ["document", "file", "pdf", "text", "upload", "paper", "article", "attachment"]
        
        if any(ind in q for ind in indicators):
            return True
        if any(ref in q for ref in doc_refs) and any(w in q for w in ["what", "about", "explain", "describe", "detail"]):
            return True
        words = q.split()
        if len(words) <= 5 and any(w in q for w in ["what", "about", "summary", "summarize", "overview", "explain"]):
            return True
        return False

    def _count_tokens(self, text: str) -> int:
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            return int(len(text.split()) * 1.35)

    def validate(self, query: str, chunks: list, retrieval_score: float, config: dict = None) -> dict:
        """
        Determines if retrieved content is relevant. If the score is in the ambiguous middle range,
        it uses the LLM to perform semantic validation.
        
        Returns:
            dict: {"relevant": bool, "reason": str, "tokens": int}
        """
        if chunks and self._is_general_query(query):
            return {
                "relevant": True,
                "reason": "General document query detected; bypassing score threshold to provide overview/summary.",
                "tokens": 0,
            }

        if not chunks or retrieval_score < HARD_REJECT_THRESHOLD:
            return {
                "relevant": False,
                "reason": "Retrieval score is too low; no useful content found in documents.",
                "tokens": 0,
            }

        if retrieval_score >= HARD_ACCEPT_THRESHOLD:
            return {
                "relevant": True,
                "reason": f"High-confidence retrieval (score {retrieval_score:.2f}).",
                "tokens": 0,
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
            answer = call_llm(prompt, config=config).strip().upper()
            relevant = answer.startswith("YES")
            tokens = self._count_tokens(prompt) + self._count_tokens(answer)
            return {
                "relevant": relevant,
                "reason": (
                    "LLM confirmed documents are relevant."
                    if relevant
                    else "LLM determined documents do not contain relevant information."
                ),
                "tokens": tokens,
            }
        except Exception as exc:
            logger.warning("Validation LLM call failed (%s); falling back to score.", exc)
            return {
                "relevant": retrieval_score >= HARD_REJECT_THRESHOLD,
                "reason": f"Score-based fallback (score {retrieval_score:.2f}).",
                "tokens": 0,
            }
