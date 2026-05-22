import logging
import os

import streamlit as st

from document_agent import DocumentAgent
from orchestrator import AgenticRAGOrchestrator

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — minimalist green / white formal theme, no icons
# IAM NOT A FRONTEND DEVELOPER TO MICROSOFT COPILOT DID THIS!!!!

st.markdown(
    """
    <style>
    /* - Global - */
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", system-ui, sans-serif;
    }

    /* - Sidebar - */
    [data-testid="stSidebar"] {
        background-color: #F1F8F1;
        border-right: 1px solid #C8E6C9;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #1B5E20;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.5rem;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        font-size: 0.82rem;
        color: #37474F;
    }

    /* - Main header - */
    .rag-header {
        border-bottom: 2px solid #2E7D32;
        padding-bottom: 0.6rem;
        margin-bottom: 1.2rem;
    }
    .rag-header h1 {
        color: #1B5E20;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .rag-header p {
        color: #546E7A;
        font-size: 0.82rem;
        margin: 0.25rem 0 0;
    }

    /* - Chat messages - */
    [data-testid="stChatMessage"] {
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.5rem;
    }
    [data-testid="stChatMessage"][aria-label="user"] {
        background-color: #E8F5E9;
        border-left: 3px solid #43A047;
    }
    [data-testid="stChatMessage"][aria-label="assistant"] {
        background-color: #FFFFFF;
        border-left: 3px solid #A5D6A7;
        border: 1px solid #E0E0E0;
    }

    /* - Chat input - */
    [data-testid="stChatInput"] textarea {
        border: 1px solid #A5D6A7 !important;
        border-radius: 4px !important;
        font-size: 0.9rem;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #2E7D32 !important;
        box-shadow: 0 0 0 2px rgba(46,125,50,0.15) !important;
    }

    /* - Buttons - */
    .stButton > button {
        background-color: #2E7D32;
        color: #FFFFFF;
        border: none;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 500;
        padding: 0.35rem 0.9rem;
        transition: background-color 0.2s;
    }
    .stButton > button:hover {
        background-color: #1B5E20;
        color: #FFFFFF;
        border: none;
    }

    /* - Source badge - */
    .source-tag {
        display: inline-block;
        background-color: #E8F5E9;
        border: 1px solid #A5D6A7;
        color: #2E7D32;
        font-size: 0.74rem;
        padding: 0.15rem 0.5rem;
        border-radius: 3px;
        margin-right: 0.35rem;
        margin-top: 0.25rem;
    }

    /* - Document list in sidebar - */
    .doc-item {
        background-color: #FFFFFF;
        border: 1px solid #C8E6C9;
        border-radius: 4px;
        padding: 0.4rem 0.6rem;
        margin-bottom: 0.3rem;
        font-size: 0.8rem;
        color: #37474F;
    }

    /* - Expander (agent trace) - */
    .streamlit-expanderHeader {
        font-size: 0.78rem;
        color: #4CAF50;
        font-weight: 500;
    }

    /* - Divider - */
    hr {
        border: none;
        border-top: 1px solid #E0E0E0;
        margin: 0.8rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state initialisation

_CFG_KEYS = [
    "OLLAMA_API_URL",
    "LLM_API_VERSION",
    "LLM_API_KEY",
    "MODEL_NAME",
    "TAVILY_API_KEY",
]

_CFG_DEFAULTS = {
    "OLLAMA_API_URL": "http://localhost:11434/api/generate",
    "LLM_API_VERSION": "",
    "LLM_API_KEY": "",
    "MODEL_NAME": "mistral",
    "TAVILY_API_KEY": "",
}


if "cfg" not in st.session_state:
    # Initialize configuration with defaults only (session-only, no persistence)
    st.session_state.cfg = {k: _CFG_DEFAULTS[k] for k in _CFG_KEYS}
    # Propagate into os.environ so utilities/agents pick them up
    for _k, _v in st.session_state.cfg.items():
        os.environ[_k] = _v

if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []
if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = set()
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = AgenticRAGOrchestrator()

doc_agent = DocumentAgent()


# Sidebar — document upload
with st.sidebar:
    st.markdown("### Documents")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "csv", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.uploaded_names:
                try:
                    chunks = doc_agent.process(uf.name, uf.read())
                    with st.spinner(f"Processing '{uf.name}'…"):
                        indexed = st.session_state.orchestrator._retrieval.index(chunks)
                        st.session_state.document_chunks.extend(chunks)
                        st.session_state.uploaded_names.add(uf.name)
                        logger.info("Processed '%s': %d chunks, %d indexed.", uf.name, len(chunks), indexed)
                        st.toast(f"'{uf.name}' ready — {len(chunks)} chunks processed.")
                except ValueError as exc:
                    logger.warning("Upload error for '%s': %s", uf.name, exc)
                    st.toast(f"Could not read '{uf.name}': {exc}")
                except RuntimeError as exc:
                    logger.error("Processing error for '%s': %s", uf.name, exc)
                    st.toast(f"Processing failed for '{uf.name}'. Check logs for details.")
                except Exception as exc:
                    logger.error("Unexpected error for '%s': %s", uf.name, exc)
                    st.toast(f"Unexpected error processing '{uf.name}'.")

    # Uploaded document list
    if st.session_state.uploaded_names:
        st.markdown("**Indexed documents**")
        for name in sorted(st.session_state.uploaded_names):
            st.markdown(f'<div class="doc-item">{name}</div>', unsafe_allow_html=True)

        if st.button("Clear all documents"):
            st.session_state.orchestrator._retrieval.clear()
            st.session_state.document_chunks = []
            st.session_state.uploaded_names = set()
            st.toast("All documents cleared.")
            st.rerun()
    else:
        st.markdown(
            '<p style="color:#78909C;font-size:0.8rem;">No documents uploaded yet.</p>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # --- Credentials / Configuration ---
    st.markdown("### Configuration")

    with st.expander("API Credentials & Endpoints", expanded=False):
        cfg = st.session_state.cfg

        st.markdown(
            '<p style="color:#1B5E20;font-size:0.78rem;font-weight:600;">LLM</p>',
            unsafe_allow_html=True,
        )
        cfg_ollama_url = st.text_input(
            "LLM Endpoint URL",
            value=cfg["OLLAMA_API_URL"],
            help="Generate endpoint for your Ollama / OpenAI-compatible LLM server.",
        )
        cfg_llm_api_version = st.text_input(
            "LLM API Version",
            value=cfg["LLM_API_VERSION"],
            placeholder="e.g. v1  or  2024-02-01  (leave blank if not required)",
            help="Sent as the X-API-Version header on every LLM request.",
        )
        cfg_llm_api_key = st.text_input(
            "LLM API Key",
            value=cfg["LLM_API_KEY"],
            type="password",
            placeholder="e.g. sk-... (leave blank if not required)",
            help="API key for authentication. Sent as Authorization header.",
        )
        cfg_model = st.text_input(
            "LLM Model Name",
            value=cfg["MODEL_NAME"],
            help="Name of the model used for generation (e.g. mistral, gpt-4o).",
        )

        st.markdown(
            '<p style="color:#1B5E20;font-size:0.78rem;font-weight:600;margin-top:0.6rem;">Web Search</p>',
            unsafe_allow_html=True,
        )
        cfg_tavily_key = st.text_input(
            "Tavily API Key",
            value=cfg["TAVILY_API_KEY"],
            type="password",
            help="Optional. Enables web-search fallback. Get a key at https://app.tavily.com.",
        )
        if st.button("Save Configuration"):
            new_cfg = {
                "OLLAMA_API_URL": cfg_ollama_url,
                "LLM_API_VERSION": cfg_llm_api_version,
                "LLM_API_KEY": cfg_llm_api_key,
                "MODEL_NAME": cfg_model,
                "TAVILY_API_KEY": cfg_tavily_key,
            }
            st.session_state.cfg = new_cfg
            # Push into os.environ so utilities and agents pick them up immediately
            for _k, _v in new_cfg.items():
                os.environ[_k] = _v
            # Force orchestrator re-init so WebSearchAgent picks up new Tavily key
            if "orchestrator" in st.session_state:
                del st.session_state["orchestrator"]
            st.toast("Configuration applied.")
            st.rerun()

    st.markdown("")
    st.markdown("### About")
    st.markdown(
        '<p>Sequential multi-agent RAG pipeline. '
        'Agents: Retrieval, Validation, Verification, Review, Web Search.</p>',
        unsafe_allow_html=True,
    )
    if st.session_state.orchestrator._websearch.available:
        st.markdown(
            '<p style="color:#2E7D32;font-size:0.78rem;">Web search: active</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p style="color:#78909C;font-size:0.78rem;">Web search: not configured</p>',
            unsafe_allow_html=True,
        )

# 
# Main area
# 
st.markdown(
    '<div class="rag-header">'
    "<h1>RAG Assistant</h1>"
    "<p>Upload documents and ask questions. The pipeline retrieves, validates, verifies, "
    "and reviews every answer.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Render sources and trace for assistant turns
        if msg["role"] == "assistant" and msg.get("sources"):
            badges = "".join(
                f'<span class="source-tag">{s}</span>' for s in msg["sources"]
            )
            st.markdown(
                f'<div style="margin-top:0.5rem;">{badges}</div>',
                unsafe_allow_html=True,
            )
        if msg["role"] == "assistant" and msg.get("agent_trace"):
            with st.expander("Agent pipeline trace"):
                for i, step in enumerate(msg["agent_trace"], 1):
                    st.markdown(f"`{i}.` {step}")


# Chat input
user_input = st.chat_input("Ask a question about your documents…")

if user_input:
    user_input = user_input.strip()
    if not user_input:
        st.stop()

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    if not st.session_state.document_chunks and not st.session_state.orchestrator._websearch.available:
        answer_data = {
            "answer": (
                "Please upload at least one document before asking a question. "
                "Web search is not configured either."
            ),
            "sources": [],
            "agent_trace": [],
            "from_web": False,
        }
        st.toast("No documents uploaded and web search is not configured.")
    else:
        # Run the agent pipeline
        with st.spinner("Processing through agent pipeline…"):
            try:
                answer_data = st.session_state.orchestrator.run(
                    query=user_input,
                    document_chunks=st.session_state.document_chunks,
                    conversation_history=st.session_state.messages[:-1],
                )
                if answer_data["from_web"]:
                    st.toast("Answer sourced from web search.")
            except RuntimeError as exc:
                logger.error("Pipeline error: %s", exc)
                st.toast(f"Error: {exc}")
                answer_data = {
                    "answer": str(exc),
                    "sources": [],
                    "agent_trace": [],
                    "from_web": False,
                }
            except Exception as exc:
                logger.error("Unexpected pipeline error: %s", exc, exc_info=True)
                st.toast("An unexpected error occurred. Check logs for details.")
                answer_data = {
                    "answer": "An unexpected error occurred. Please try again.",
                    "sources": [],
                    "agent_trace": [],
                    "from_web": False,
                }

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(answer_data["answer"])
        if answer_data.get("sources"):
            badges = "".join(
                f'<span class="source-tag">{s}</span>' for s in answer_data["sources"]
            )
            st.markdown(
                f'<div style="margin-top:0.5rem;">{badges}</div>',
                unsafe_allow_html=True,
            )
        if answer_data.get("agent_trace"):
            with st.expander("Agent pipeline trace"):
                for i, step in enumerate(answer_data["agent_trace"], 1):
                    st.markdown(f"`{i}.` {step}")

    # Persist to history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_data["answer"],
            "sources": answer_data.get("sources", []),
            "agent_trace": answer_data.get("agent_trace", []),
        }
    )

