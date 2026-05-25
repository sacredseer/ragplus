import logging
import time
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from .retrieval_agent import RetrievalAgent
from .validation_agent import ValidationAgent
from .verification_agent import VerificationAgent
from .review_agent import ReviewAgent
from .websearch_agent import WebSearchAgent
from ..util.utilities import call_llm

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


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return int(len(text.split()) * 1.35)


class AgentState(TypedDict):
    query: str
    resolved_query: str
    history: List[Dict[str, str]]
    document_chunks: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    final_chunks: List[Dict[str, Any]]
    retrieval_score: float
    is_relevant: bool
    from_web: bool
    draft_answer: str
    is_verified: bool
    verification_issues: str
    final_answer: str
    agent_trace: List[str]
    config: Dict[str, Any]


class AgenticRAGOrchestrator:
    """
    Coordinates a sequential multi-agent RAG workflow built with LangGraph.
    
    Workflows flow through the following nodes:
    1. Resolve Query Node: rewrites user queries based on context.
    2. Retrieval Node: fetches chunks matching the query.
    3. Validation Node: checks relevance of retrieval results.
    4. Web Search Node: falls back to Tavily search when documents lack content.
    5. Draft Node: synthesizes the initial draft answer.
    6. Verification Node: fact-checks the draft against the source chunks.
    7. Grounded Draft Node: strictly regenerates the answer if fact-check fails.
    8. Review Node: polishes formatting and style.
    """

    def __init__(self):
        self._retrieval = RetrievalAgent(top_k=6)
        self._validation = ValidationAgent()
        self._verification = VerificationAgent()
        self._review = ReviewAgent()
        self._websearch = WebSearchAgent()
        self._graph = self._build_workflow_graph()
        self._status_callback = None

    def _build_workflow_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("resolve_query", self._node_resolve_query)
        workflow.add_node("retrieve", self._node_retrieve)
        workflow.add_node("validate", self._node_validate)
        workflow.add_node("websearch", self._node_websearch)
        workflow.add_node("draft", self._node_draft)
        workflow.add_node("verify", self._node_verify)
        workflow.add_node("grounded_draft", self._node_grounded_draft)
        workflow.add_node("review", self._node_review)

        workflow.set_entry_point("resolve_query")
        workflow.add_edge("resolve_query", "retrieve")
        workflow.add_edge("retrieve", "validate")

        workflow.add_conditional_edges(
            "validate",
            self._route_after_validation,
            {
                "relevant": "draft",
                "not_relevant": "websearch",
            }
        )

        workflow.add_conditional_edges(
            "websearch",
            self._route_after_websearch,
            {
                "continue_to_draft": "draft",
                "end_pipeline": END,
            }
        )

        workflow.add_edge("draft", "verify")

        workflow.add_conditional_edges(
            "verify",
            self._route_after_verification,
            {
                "verified": "review",
                "regenerate": "grounded_draft",
            }
        )

        workflow.add_edge("grounded_draft", "review")
        workflow.add_edge("review", END)

        return workflow.compile()

    def _node_resolve_query(self, state: AgentState) -> Dict[str, Any]:
        if self._status_callback:
            self._status_callback("Step 1 — Query Resolution: rewriting query based on conversation context...")
            
        trace = list(state.get("agent_trace", []))
        start_time = time.perf_counter()
        
        resolved, tokens = self._resolve_query(state["query"], state["history"], config=state["config"])
        elapsed = time.perf_counter() - start_time
        
        trace.append(f"Step 1 — Query Resolution ({elapsed:.3f}s : {tokens} tokens): resolving query with conversation context.")
        logger.info("Resolved query: %s", resolved)
        return {
            "resolved_query": resolved,
            "agent_trace": trace,
        }

    def _node_retrieve(self, state: AgentState) -> Dict[str, Any]:
        if self._status_callback:
            self._status_callback("Step 2 — Retrieval Agent: searching document chunks...")
            
        trace = list(state.get("agent_trace", []))
        start_time = time.perf_counter()
        
        if self._retrieval.count != len(state["document_chunks"]) and state["document_chunks"]:
            self._retrieval.clear()
            self._retrieval.index(state["document_chunks"])
            
        retrieved, max_score = self._retrieval.retrieve(state["resolved_query"])
        elapsed = time.perf_counter() - start_time
        
        trace.append(f"Step 2 — Retrieval Agent ({elapsed:.3f}s : 0 tokens): searching uploaded documents.")
        trace.append(f"  Retrieved {len(retrieved)} chunk(s) (top score: {max_score:.3f}).")
        return {
            "retrieved_chunks": retrieved,
            "retrieval_score": max_score,
            "agent_trace": trace,
        }

    def _node_validate(self, state: AgentState) -> Dict[str, Any]:
        if self._status_callback:
            self._status_callback("Step 3 — Validation Agent: evaluating context relevance...")
            
        trace = list(state.get("agent_trace", []))
        start_time = time.perf_counter()
        
        validation = self._validation.validate(
            state["resolved_query"],
            state["retrieved_chunks"],
            state["retrieval_score"],
            config=state["config"]
        )
        elapsed = time.perf_counter() - start_time
        tokens = validation.get("tokens", 0)
        
        trace.append(f"Step 3 — Validation Agent ({elapsed:.3f}s : {tokens} tokens): checking relevance of retrieved content.")
        trace.append(f"  {validation['reason']}")
        return {
            "is_relevant": validation["relevant"],
            "agent_trace": trace,
        }

    def _route_after_validation(self, state: AgentState) -> str:
        return "relevant" if state["is_relevant"] else "not_relevant"

    def _node_websearch(self, state: AgentState) -> Dict[str, Any]:
        if self._status_callback:
            self._status_callback("Step 4 — Web Search Agent: falling back to Tavily search...")
            
        trace = list(state.get("agent_trace", []))
        start_time = time.perf_counter()
        
        tavily_key = (state["config"].get("TAVILY_API_KEY") or "").strip()
        if not tavily_key:
            elapsed = time.perf_counter() - start_time
            trace.append(f"Step 4 — Web Search Agent ({elapsed:.3f}s : 0 tokens): no document results; searching the web.")
            trace.append("  Web search is not configured (TAVILY_API_KEY missing).")
            if not state["document_chunks"]:
                return {
                    "final_answer": _NO_DOCS_NO_WEB_MSG,
                    "agent_trace": trace,
                    "from_web": False,
                }
            trace.append("  Web search unavailable. Ending pipeline.")
            return {
                "final_answer": _NO_RESULTS_MSG,
                "agent_trace": trace,
                "from_web": False,
            }
        
        web_results = self._websearch.search(state["resolved_query"], api_key=tavily_key)
        elapsed = time.perf_counter() - start_time
        
        trace.append(f"Step 4 — Web Search Agent ({elapsed:.3f}s : 0 tokens): no document results; searching the web.")
        if not web_results:
            trace.append("  Web search returned no results.")
            if not state["document_chunks"]:
                return {
                    "final_answer": _NO_DOCS_WEB_FAIL_MSG,
                    "agent_trace": trace,
                    "from_web": False,
                }
            trace.append("  Web search returned no results. Ending pipeline.")
            return {
                "final_answer": _NO_RESULTS_MSG,
                "agent_trace": trace,
                "from_web": False,
            }

        trace.append(f"  Web search returned {len(web_results)} result(s).")
        return {
            "final_chunks": web_results,
            "from_web": True,
            "agent_trace": trace,
        }

    def _route_after_websearch(self, state: AgentState) -> str:
        if state.get("final_answer"):
            return "end_pipeline"
        return "continue_to_draft"

    def _node_draft(self, state: AgentState) -> Dict[str, Any]:
        if self._status_callback:
            self._status_callback("Step 5 — Answer Generation: drafting answer from sources...")
            
        trace = list(state.get("agent_trace", []))
        start_time = time.perf_counter()
        
        chunks = state.get("final_chunks")
        if chunks is None:
            chunks = state["retrieved_chunks"]
            
        draft, tokens = self._generate_answer(state["resolved_query"], chunks, state["history"], config=state["config"])
        elapsed = time.perf_counter() - start_time
        
        trace.append(f"Step 5 — Answer Generation ({elapsed:.3f}s : {tokens} tokens): drafting answer from retrieved content.")
        return {
            "draft_answer": draft,
            "final_chunks": chunks,
            "agent_trace": trace,
        }

    def _node_verify(self, state: AgentState) -> Dict[str, Any]:
        if self._status_callback:
            self._status_callback("Step 6 — Verification Agent: checking facts for accuracy...")
            
        trace = list(state.get("agent_trace", []))
        start_time = time.perf_counter()
        
        draft = state.get("draft_answer", "")
        if not draft:
            elapsed = time.perf_counter() - start_time
            return {
                "is_verified": False,
                "verification_issues": "Draft answer is empty.",
                "agent_trace": trace,
            }

        verification = self._verification.verify(
            state["resolved_query"],
            draft,
            state["final_chunks"],
            config=state["config"]
        )
        elapsed = time.perf_counter() - start_time
        tokens = verification.get("tokens", 0)
        
        trace.append(f"Step 6 — Verification Agent ({elapsed:.3f}s : {tokens} tokens): verifying answer against sources.")
        if not verification["verified"]:
            trace.append(f"  Issues found: {verification['issues'][:120]}")
            trace.append("  Regenerating strictly grounded answer.")
            return {
                "is_verified": False,
                "verification_issues": verification["issues"],
                "agent_trace": trace,
            }
        
        trace.append("  Answer verified.")
        return {
            "is_verified": True,
            "verification_issues": "",
            "agent_trace": trace,
        }

    def _route_after_verification(self, state: AgentState) -> str:
        return "verified" if state["is_verified"] else "regenerate"

    def _node_grounded_draft(self, state: AgentState) -> Dict[str, Any]:
        if self._status_callback:
            self._status_callback("Step 6b — Grounded Draft Agent: strictly regenerating answer...")
            
        trace = list(state.get("agent_trace", []))
        start_time = time.perf_counter()
        
        draft, tokens = self._generate_grounded_answer(
            state["resolved_query"],
            state["final_chunks"],
            state["history"],
            config=state["config"]
        )
        elapsed = time.perf_counter() - start_time
        trace.append(f"  Grounded Answer Generation ({elapsed:.3f}s : {tokens} tokens): strictly regenerated answer.")
        return {
            "draft_answer": draft,
            "agent_trace": trace,
        }

    def _node_review(self, state: AgentState) -> Dict[str, Any]:
        if self._status_callback:
            self._status_callback("Step 7 — Review Agent: polishing final response styling...")
            
        trace = list(state.get("agent_trace", []))
        start_time = time.perf_counter()
        
        draft = state.get("draft_answer", "")
        if not draft:
            elapsed = time.perf_counter() - start_time
            return {
                "final_answer": _NO_RESULTS_MSG,
                "agent_trace": trace,
            }
            
        final_answer, tokens = self._review.review(
            state["resolved_query"],
            draft,
            state["final_chunks"],
            state["history"],
            config=state["config"]
        )
        elapsed = time.perf_counter() - start_time
        
        trace.append(f"Step 7 — Review Agent ({elapsed:.3f}s : {tokens} tokens): refining and finalising answer.")
        trace.append("  Pipeline complete.")
        return {
            "final_answer": final_answer,
            "agent_trace": trace,
        }

    def run(self, query: str, document_chunks: list, conversation_history: list = None, config: dict = None, status_callback = None) -> dict:
        """
        Executes the LangGraph-managed sequential multi-agent RAG workflow.
        """
        self._status_callback = status_callback
        initial_state: AgentState = {
            "query": query,
            "resolved_query": query,
            "history": conversation_history or [],
            "document_chunks": document_chunks,
            "retrieved_chunks": [],
            "final_chunks": None,
            "retrieval_score": 0.0,
            "is_relevant": False,
            "from_web": False,
            "draft_answer": "",
            "is_verified": False,
            "verification_issues": "",
            "final_answer": "",
            "agent_trace": [],
            "config": config or {},
        }

        try:
            final_output = self._graph.invoke(initial_state)
        finally:
            self._status_callback = None

        final_chunks = final_output.get("final_chunks") or []
        sources = list(dict.fromkeys(c.get("source", "") for c in final_chunks))

        return {
            "answer": final_output.get("final_answer", _NO_RESULTS_MSG),
            "sources": sources,
            "agent_trace": final_output.get("agent_trace", []),
            "from_web": final_output.get("from_web", False),
        }

    def _resolve_query(self, query: str, history: list, config: dict = None) -> tuple[str, int]:
        if not history:
            return query, 0

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
            resolved = call_llm(prompt, config=config).strip()
            resolved_query = resolved if resolved else query
            tokens = _count_tokens(prompt) + _count_tokens(resolved_query)
            return resolved_query, tokens
        except Exception as exc:
            logger.warning("Query resolution failed (%s); using original query.", exc)
            return query, _count_tokens(prompt)

    def _generate_answer(self, query: str, chunks: list, history: list, config: dict = None) -> tuple[str, int]:
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
            draft = call_llm(prompt, config=config).strip()
            tokens = _count_tokens(prompt) + _count_tokens(draft)
            return draft, tokens
        except Exception as exc:
            logger.error("Draft answer generation failed: %s", exc)
            raise

    def _generate_grounded_answer(self, query: str, chunks: list, history: list, config: dict = None) -> tuple[str, int]:
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
            draft = call_llm(prompt, config=config).strip()
            tokens = _count_tokens(prompt) + _count_tokens(draft)
            return draft, tokens
        except Exception as exc:
            logger.error("Strictly grounded answer generation failed: %s", exc)
            raise
