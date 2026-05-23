import logging
import os
import streamlit as st

from document_agent import DocumentAgent
from orchestrator import AgenticRAGOrchestrator

# Setup application logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Basic layout config for Streamlit
st.set_page_config(
    page_title="RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Show configuration saved notification across page reruns
if st.session_state.get("config_saved"):
    st.toast("Configuration saved and applied!")
    st.session_state.config_saved = False

# I am not a frontend developer so the code below are MS Copilot generated.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --material-green: #2E7D32;
        --material-green-light: rgba(46, 125, 50, 0.1);
        --material-green-border: rgba(46, 125, 50, 0.3);
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', system-ui, sans-serif;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--material-green-border) !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--material-green) !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    .rag-header {
        border-bottom: 2px solid var(--material-green);
        padding-bottom: 0.8rem;
        margin-bottom: 1.5rem;
    }
    .rag-header h1 {
        color: var(--material-green) !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        letter-spacing: -0.02em !important;
    }
    .rag-header p {
        color: var(--text-color) !important;
        opacity: 0.8;
        font-size: 0.85rem !important;
        margin: 0.3rem 0 0 !important;
    }

    [data-testid="stChatMessage"] {
        border: 1px solid var(--material-green-border) !important;
        border-radius: 8px !important;
        padding: 0.6rem 0.8rem !important;
        margin-bottom: 0.6rem !important;
    }
    [data-testid="stChatMessage"] > div {
        background-color: transparent !important;
    }

    [data-testid="stChatMessage"][aria-label="user"] {
        background-color: var(--material-green-light) !important;
        border-left: 4px solid var(--material-green) !important;
    }

    [data-testid="stChatInput"] textarea {
        border: 1px solid var(--material-green-border) !important;
        border-radius: 6px !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--material-green) !important;
        box-shadow: 0 0 0 2px rgba(46, 125, 50, 0.2) !important;
    }

    .stButton > button {
        background-color: var(--material-green) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        padding: 0.4rem 1rem !important;
        transition: opacity 0.2s, transform 0.1s !important;
    }
    .stButton > button:hover {
        opacity: 0.9 !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    .source-tag {
        display: inline-block;
        background-color: var(--material-green-light);
        border: 1px solid var(--material-green-border);
        color: var(--material-green);
        font-size: 0.72rem;
        font-weight: 500;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        margin-right: 0.4rem;
        margin-top: 0.3rem;
    }

    .doc-item {
        border: 1px solid var(--material-green-border);
        border-radius: 6px;
        padding: 0.5rem 0.7rem;
        margin-bottom: 0.4rem;
        font-size: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .streamlit-expanderHeader {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid var(--material-green-border) !important;
        border-radius: 6px !important;
        font-size: 0.8rem !important;
        color: var(--material-green) !important;
    }

    hr {
        border: none !important;
        border-top: 1px solid var(--material-green-border) !important;
        margin: 1rem 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Keys and defaults used for the dynamic configuration interface
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

# Initialize session configuration with default empty/local values.
# To keep credentials completely secure, they are never written to process environment variables
# and are cleared entirely when the page is refreshed or the tab is closed.
if "cfg" not in st.session_state:
    st.session_state.cfg = {k: _CFG_DEFAULTS[k] for k in _CFG_KEYS}

if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []
if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = set()
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = AgenticRAGOrchestrator()

doc_agent = DocumentAgent()

# Documents Panel in the Sidebar
with st.sidebar:
    st.markdown("### Documents")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "csv", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Process new documents on upload
    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.uploaded_names:
                try:
                    chunks = doc_agent.process(uf.name, uf.read())
                    with st.spinner(f"Processing '{uf.name}'..."):
                        indexed = st.session_state.orchestrator._retrieval.index(chunks)
                        st.session_state.document_chunks.extend(chunks)
                        st.session_state.uploaded_names.add(uf.name)
                        logger.info("Processed '%s': %d chunks indexed.", uf.name, len(chunks))
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

    # Render files list or empty message
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

    # Credentials & API settings panel
    st.markdown("### Configuration")

    with st.expander("API Credentials & Endpoints", expanded=False):
        cfg = st.session_state.cfg

        st.markdown(
            '<p style="color:var(--primary-green);font-size:0.78rem;font-weight:600;">LLM</p>',
            unsafe_allow_html=True,
        )
        cfg_ollama_url = st.text_input(
            "LLM Endpoint URL",
            value=cfg["OLLAMA_API_URL"],
            help="The full API URL of your LLM generation endpoint (e.g. http://localhost:11434/api/generate).",
        )
        cfg_llm_api_version = st.text_input(
            "LLM API Version",
            value=cfg["LLM_API_VERSION"],
            placeholder="e.g. v1 (optional)",
            help="Value sent as the X-API-Version header on LLM API calls.",
        )
        cfg_llm_api_key = st.text_input(
            "LLM API Key",
            value=cfg["LLM_API_KEY"],
            type="password",
            placeholder="e.g. sk-... (optional)",
            help="Authorization key sent as a Bearer token in the headers.",
        )
        cfg_model = st.text_input(
            "LLM Model Name",
            value=cfg["MODEL_NAME"],
            help="Model identifier parameter for the LLM request payload.",
        )

        st.markdown(
            '<p style="color:var(--primary-green);font-size:0.78rem;font-weight:600;margin-top:0.6rem;">Web Search</p>',
            unsafe_allow_html=True,
        )
        cfg_tavily_key = st.text_input(
            "Tavily API Key",
            value=cfg["TAVILY_API_KEY"],
            type="password",
            help="Optional token to search the web if document retrieval fails.",
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
            # Recreate orchestrator instance to reset the state machine
            if "orchestrator" in st.session_state:
                del st.session_state["orchestrator"]
            st.session_state.config_saved = True
            st.rerun()

    st.markdown("")
    st.markdown("### About")
    st.markdown(
        '<p>Sequential multi-agent RAG pipeline. '
        'Stages: Query Resolution, Retrieval, Validation, Web Search, Answer Generation, Verification, Review.</p>',
        unsafe_allow_html=True,
    )
    if st.session_state.cfg.get("TAVILY_API_KEY", "").strip():
        st.markdown(
            '<p style="color:var(--primary-green);font-size:0.78rem;font-weight:500;">Web search: active</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p style="color:#78909C;font-size:0.78rem;">Web search: not configured</p>',
            unsafe_allow_html=True,
        )

# Main chat interface layout
st.markdown(
    '<div class="rag-header">'
    "<h1>RAG Assistant</h1>"
    "<p>Upload documents and ask questions. The pipeline retrieves, validates, verifies, "
    "and reviews every answer.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# Render the active chat history log
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
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

# Receive and process user queries
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    user_input = user_input.strip()
    if not user_input:
        st.stop()

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Validate that we have search options available
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
        with st.spinner("Processing through agent pipeline..."):
            try:
                answer_data = st.session_state.orchestrator.run(
                    query=user_input,
                    document_chunks=st.session_state.document_chunks,
                    conversation_history=st.session_state.messages[:-1],
                    config=st.session_state.cfg,
                )
                if answer_data["from_web"]:
                    st.toast("Answer sourced from web search.")
            except RuntimeError as exc:
                logger.error("Pipeline run error: %s", exc)
                st.toast(f"Error: {exc}")
                answer_data = {
                    "answer": str(exc),
                    "sources": [],
                    "agent_trace": [],
                    "from_web": False,
                }
            except Exception as exc:
                logger.error("Unexpected pipeline execution error: %s", exc, exc_info=True)
                st.toast("An unexpected error occurred. Check logs for details.")
                answer_data = {
                    "answer": "An unexpected error occurred. Please try again.",
                    "sources": [],
                    "agent_trace": [],
                    "from_web": False,
                }

    # Render the assistant's processed response
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

    # Log response to conversation state
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_data["answer"],
            "sources": answer_data.get("sources", []),
            "agent_trace": answer_data.get("agent_trace", []),
        }
    )
