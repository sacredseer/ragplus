import logging

from utilities import call_llm

logger = logging.getLogger(__name__)

_MAX_HISTORY_TURNS = 4

class ReviewAgent:
    """Polishes the final answer for clarity and completeness without adding hallucinations."""

    def review(
        self,
        query: str,
        draft: str,
        chunks: list,
        conversation_history: list,
    ) -> str:
        """
        Produce a refined, conversational final answer grounded strictly in *chunks*.
        Falls back to *draft* if the LLM call fails.
        """
        history_block = self._format_history(conversation_history)
        context = "\n\n".join(
            f"[{c.get('source', 'unknown')}]: {c['content'][:400]}" for c in chunks[:6]
        )

        prompt = (
            "You are a knowledgeable assistant providing accurate answers.\n\n"
            + (f"Conversation history:\n{history_block}\n\n" if history_block else "")
            + "Source material:\n"
            f"{context}\n\n"
            "Query:\n"
            f"{query}\n\n"
            "Draft answer:\n"
            f"{draft}\n\n"
            "Your task:\n"
            "1. Rewrite the draft into a clear, well-structured answer.\n"
            "2. Use ONLY information present in the source material.\n"
            "3. If the draft contains any claim not in the sources, remove it.\n"
            "4. Keep the tone formal and concise.\n"
            "5. Do NOT include a 'Sources:' section — that is handled separately.\n\n"
            "Final answer:"
        )

        try:
            refined = call_llm(prompt).strip()
            return refined if refined else draft
        except Exception as exc:
            logger.warning("Review LLM call failed (%s); returning draft.", exc)
            return draft

    def _format_history(self, history: list) -> str:
        if not history:
            return ""
        recent = history[-_MAX_HISTORY_TURNS:]
        lines = [f"{m['role'].capitalize()}: {m['content'][:200]}" for m in recent]
        return "\n".join(lines)
