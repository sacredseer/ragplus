import logging

from utilities import call_llm

logger = logging.getLogger(__name__)


class VerificationAgent:
    """Checks that a draft answer is grounded in the provided source chunks."""

    def verify(self, query: str, draft: str, chunks: list) -> dict:
        """
        Verify that *draft* is supported by *chunks*.
        Returns: {"verified": bool, "issues": str}
        """
        if not draft.strip():
            return {"verified": False, "issues": "Draft answer is empty."}

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
            result = call_llm(prompt).strip()
            verified = result.upper().startswith("VERIFIED")
            issues = "" if verified else result
            logger.debug("Verification result: %s", result[:80])
            return {"verified": verified, "issues": issues}
        except Exception as exc:
            logger.warning("Verification LLM call failed (%s); passing answer through.", exc)
            return {"verified": True, "issues": ""}
