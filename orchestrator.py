import logging

from retrieval_agent import RetrievalAgent
from validation_agent import ValidationAgent
from verification_agent import VerificationAgent
from review_agent import ReviewAgent
from websearch_agent import WebSearchAgent
from utilities import call_llm

logger = logging.getLogger(__name__)

_NO_RESULTS_MSG = (
    "I was unable to find relevant information in the uploaded documents or via web search. "
    "Please upload documents related to your question, or refine your query."
)
_NO_DOCS_NO_WEB_MSG = (
    "No documents have been uploaded and web search is not configured. "
    "Please upload a document to begin."
)
_NO_DOCS_WEB_FAIL_MSG = (
    "No documents have been uploaded. Web search returned no results for your query."
)


class AgenticRAGOrchestrator:
    """
    Sequential multi-agent RAG pipeline.

    Stage order:
        1. Query Resolution   — rewrite follow-up queries into standalone questions
        2. Retrieval Agent    — TF-IDF retrieval over document chunks
        3. Validation Agent   — confirm retrieved chunks are relevant
        4. Web Search Agent   — fallback when documents yield no relevant content
        5. Answer Generation  — LLM drafts an answer grounded in retrieved content
        6. Verification Agent — checks the draft is fully supported by sources
        7. Review Agent       — polishes and finalises the answer
    """

    def __init__(self):
        self._retrieval = RetrievalAgent(top_k=6)
        self._validation = ValidationAgent()
        self._verification = VerificationAgent()
        self._review = ReviewAgent()
        self._websearch = WebSearchAgent()

    def run(self, query: str, document_chunks: list, conversation_history: list = None) -> dict:
        """
        Execute the pipeline and return a result dict:
        {
            "answer":       str,
            "sources":      list[str],
            "agent_trace":  list[str],
            "from_web":     bool,
        }
        """
        history = conversation_history or []
        trace = []

        trace.append("Stage 1 — Query Resolution: resolving query with conversation context.")
        resolved = self._resolve_query(query, history)
        logger.info("Resolved query: %s", resolved)

        trace.append("Stage 2 — Retrieval Agent: searching uploaded documents.")
        retrieved, max_score = self._retrieval.retrieve(resolved, document_chunks)
        trace.append(f"  Retrieved {len(retrieved)} chunk(s) (top score: {max_score:.3f}).")

        trace.append("Stage 3 — Validation Agent: checking relevance of retrieved content.")
        validation = self._validation.validate(resolved, retrieved, max_score)
        trace.append(f"  {validation['reason']}")

        from_web = False
        final_chunks = retrieved

        if not validation["relevant"]:
            trace.append("Stage 4 — Web Search Agent: no document results; searching the web.")
            if not self._websearch.available:
                trace.append("  Web search is not configured (TAVILY_API_KEY missing).")
                if not document_chunks:
                    return self._no_result(_NO_DOCS_NO_WEB_MSG, trace)

                trace.append("  Web search unavailable; falling back to direct LLM answer from document chunks.")
                final_chunks = document_chunks[:6]
            else:
                web_results = self._websearch.search(resolved)
                if not web_results:
                    trace.append("  Web search returned no results.")
                    if not document_chunks:
                        return self._no_result(_NO_DOCS_WEB_FAIL_MSG, trace)
                    trace.append("  Falling back to direct LLM answer from document chunks.")
                    final_chunks = document_chunks[:6]
                else:
                    final_chunks = web_results
                    from_web = True
                    trace.append(f"  Web search returned {len(web_results)} result(s).")
        else:
            trace.append("Stage 4 — Web Search Agent: skipped (document results are sufficient).")

        trace.append("Stage 5 — Answer Generation: drafting answer from retrieved content.")
        draft = self._generate_answer(resolved, final_chunks, history)

        if not draft:
            return self._no_result(_NO_RESULTS_MSG, trace)

        trace.append("Stage 6 — Verification Agent: verifying answer against sources.")
        verification = self._verification.verify(resolved, draft, final_chunks)
        if not verification["verified"]:
            trace.append(f"  Issues found: {verification['issues'][:120]}")
            trace.append("  Regenerating strictly grounded answer.")
            draft = self._generate_grounded_answer(resolved, final_chunks, history)
        else:
            trace.append("  Answer verified.")

        trace.append("Stage 7 — Review Agent: refining and finalising answer.")
        final_answer = self._review.review(resolved, draft, final_chunks, history)
        trace.append("  Pipeline complete.")

        sources = list(dict.fromkeys(c.get("source", "") for c in final_chunks))

        return {
            "answer": final_answer,
            "sources": sources,
            "agent_trace": trace,
            "from_web": from_web,
        }

    def _resolve_query(self, query: str, history: list) -> str:
        if not history:
            return query

        recent = history[-4:]
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content'][:200]}" for m in recent
        )
        prompt = (
            "Given the conversation history below, rewrite the latest query as a standalone, "
            "self-contained question that does not require the conversation history to understand.\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Latest query: {query}\n\n"
            "Return ONLY the rewritten query, nothing else."
        )
        try:
            resolved = call_llm(prompt).strip()
            return resolved if resolved else query
        except Exception as exc:
            logger.warning("Query resolution failed (%s); using original.", exc)
            return query

    def _generate_answer(self, query: str, chunks: list, history: list) -> str:
        history_block = ""
        if history:
            recent = history[-4:]
            history_block = "\n".join(
                f"{m['role'].capitalize()}: {m['content'][:200]}" for m in recent
            )

        context = "\n\n".join(
            f"[Source: {c.get('source', 'unknown')}]\n{c['content']}" for c in chunks[:6]
        )

        prompt = (
            "You are a precise and helpful assistant. Answer the query using ONLY the context "
            "provided below. Do not use any outside knowledge. If the context does not fully "
            "answer the question, say clearly what is and is not covered.\n\n"
            + (f"Conversation history:\n{history_block}\n\n" if history_block else "")
            + f"Context:\n{context}\n\n"
            f"Query: {query}\n\n"
            "Answer:"
        )
        try:
            return call_llm(prompt).strip()
        except Exception as exc:
            logger.error("Answer generation failed: %s", exc)
            raise

    def _generate_grounded_answer(self, query: str, chunks: list, history: list) -> str:
        context = "\n\n".join(
            f"[Source: {c.get('source', 'unknown')}]\n{c['content']}" for c in chunks[:6]
        )
        prompt = (
            "Answer STRICTLY using only the sources listed below. "
            "Do NOT use any outside knowledge. If the sources do not contain sufficient "
            "information, say: 'The provided documents do not contain sufficient information "
            "to answer this question.'\n\n"
            f"Sources:\n{context}\n\n"
            f"Query: {query}\n\n"
            "Strictly grounded answer:"
        )
        try:
            return call_llm(prompt).strip()
        except Exception as exc:
            logger.error("Grounded answer generation failed: %s", exc)
            raise

    @staticmethod
    def _no_result(message: str, trace: list) -> dict:
        return {
            "answer": message,
            "sources": [],
            "agent_trace": trace,
            "from_web": False,
        }
