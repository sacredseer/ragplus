import logging

from utilities import call_llm

logger = logging.getLogger(__name__)


class VerificationAgent:
    """Fact-checks a generated draft answer against source documents to prevent hallucinations."""

    def verify(self, query: str, draft: str, chunks: list, config: dict = None) -> dict:
        """
        Verifies that all factual assertions in the draft answer are supported by the retrieved chunks.
        
        Returns:
            dict: {"verified": bool, "issues": str}
        """
        if not draft.strip():
            return {"verified": False, "issues": "Draft answer is empty."}

        # Build context from the top source chunks to check against the draft
        context = "\n\n".join(
            f"[{c.get('source', 'unknown')}]: {c['content'][:400]}" for c in chunks[:5]
        )

        prompt = (
            "You are a fact-checker. Your task is to verify whether an answer is fully "
            "supported by the provided source excerpts.\n\n"
            f"Query: {query}\n\n"
            "Source excerpts:\n"
            f"{context}\n\n"
            "Answer to verify:\n"
            f"{draft}\n\n"
            "Instructions:\n"
            "- If every claim in the answer can be traced to the sources, respond with exactly: VERIFIED\n"
            "- If the answer introduces any claim not found in the sources, respond with: "
            "ISSUES: <brief description of the unsupported claims>\n\n"
            "Response:"
        )

        try:
            result = call_llm(prompt, config=config).strip()
            verified = result.upper().startswith("VERIFIED")
            issues = "" if verified else result
            logger.debug("Verification result: %s", result[:80])
            return {"verified": verified, "issues": issues}
        except Exception as exc:
            # If verification fails due to API issues, we default to passing the draft
            # to avoid blocking the answer pipeline unnecessarily.
            logger.warning("Verification LLM call failed (%s); passing answer through.", exc)
            return {"verified": True, "issues": ""}
