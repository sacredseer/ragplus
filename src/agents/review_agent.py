import logging

from ..util.utilities import call_llm

logger = logging.getLogger(__name__)

_MAX_HISTORY_TURNS = 4


class ReviewAgent:
    """Polishes and structures the generated answer for optimal presentation."""

    def _count_tokens(self, text: str) -> int:
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            return int(len(text.split()) * 1.35)

    def review(
        self,
        query: str,
        draft: str,
        chunks: list,
        conversation_history: list,
        config: dict = None,
    ) -> tuple[str, int]:
        """
        Reviews and formats the draft answer into a polished, coherent final response.
        Ensures strict grounding to prevent any hallucinated facts from slipping in.
        
        Returns:
            tuple[str, int]: The polished final answer and tokens consumed.
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
            refined = call_llm(prompt, config=config).strip()
            final_answer = refined if refined else draft
            tokens = self._count_tokens(prompt) + self._count_tokens(final_answer)
            return final_answer, tokens
        except Exception as exc:
            logger.warning("Review LLM call failed (%s); returning draft.", exc)
            return draft, self._count_tokens(prompt) + self._count_tokens(draft)

    def _format_history(self, history: list) -> str:
        if not history:
            return ""
        recent = history[-_MAX_HISTORY_TURNS:]
        lines = [f"{m['role'].capitalize()}: {m['content'][:200]}" for m in recent]
        return "\n".join(lines)
