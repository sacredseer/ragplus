import os
import streamlit as st

from src.agents.document_agent import DocumentAgent
from src.agents.orchestrator import AgenticRAGOrchestrator
from src.util.oauth import OAuthClient

DEFAULT_AVATAR = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%2378909C'><path d='M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z'/></svg>"

try:
    base_url = st.context.url.rstrip("/")
    if not base_url:
        base_url = "http://localhost:8501"
except Exception:
    base_url = "http://localhost:8501"

st.set_page_config(
    page_title="Document Buddy",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css("src/ui/style.css")

if "banners" not in st.session_state:
    st.session_state.banners = []

if st.session_state.get("config_saved"):
    st.session_state.banners.append({"type": "success", "text": "Configuration saved and applied!"})
    st.session_state.config_saved = False

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

code = st.query_params.get("code")
state = st.query_params.get("state")

if code:
    try:
        provider = st.session_state.get("oauth_provider", "google")
        client_id = st.session_state.get(f"{provider}_client_id")
        client_secret = st.session_state.get(f"{provider}_client_secret")
        client = OAuthClient(
            provider=provider,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=base_url
        )
        user_info = client.exchange_code_for_user(code)
        st.session_state.user = user_info
        st.query_params.clear()
        st.rerun()
    except Exception as exc:
        st.error(f"OAuth authentication failed: {exc}")

if "user" not in st.session_state:
    google_client_id = st.session_state.get("google_client_id")
    google_client_secret = st.session_state.get("google_client_secret")
    google_client = OAuthClient(provider="google", client_id=google_client_id, client_secret=google_client_secret, redirect_uri=base_url)
    google_auth_url = google_client.get_authorization_url()

    github_client_id = st.session_state.get("github_client_id")
    github_client_secret = st.session_state.get("github_client_secret")
    github_client = OAuthClient(provider="github", client_id=github_client_id, client_secret=github_client_secret, redirect_uri=base_url)
    github_auth_url = github_client.get_authorization_url()

    st.markdown(
        f"""
        <div class="login-card">
            <div class="login-title">Document Buddy</div>
            <div class="login-subtitle">Login using social providers or developer mode to access.</div>
            <div class="social-login-btn-container">
                <a href="{google_auth_url}" target="_self" style="text-decoration: none;">
                    <div class="social-login-btn google-btn">
                        <svg class="btn-icon" viewBox="0 0 24 24" width="18" height="18" xmlns="http://www.w3.org/2000/svg" style="margin-right: 12px; vertical-align: middle;"><path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/><path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/><path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z" fill="#FBBC05"/><path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/></svg>
                        Sign In with Google
                    </div>
                </a>
                <a href="{github_auth_url}" target="_self" style="text-decoration: none;">
                    <div class="social-login-btn github-btn">
                        <svg class="btn-icon" viewBox="0 0 24 24" width="18" height="18" xmlns="http://www.w3.org/2000/svg" style="margin-right: 12px; fill: currentColor; vertical-align: middle;"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>
                        Sign In with GitHub
                    </div>
                </a>
                <a href="{base_url}/?code=mock_code_dev_user&state=mock_state" target="_self" style="text-decoration: none;">
                    <div class="social-login-btn dev-btn">
                        <svg class="btn-icon" viewBox="0 0 24 24" width="18" height="18" xmlns="http://www.w3.org/2000/svg" style="margin-right: 12px; fill: currentColor; vertical-align: middle;"><path d="M9.4 16.6L4.8 12l4.6-4.6L8 6l-6 6 6 6 1.4-1.4zm5.2 0l4.6-4.6-4.6-4.6L16 6l6 6-6 6-1.4-1.4z"/></svg>
                        Developer Login
                    </div>
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

with st.sidebar:
    st.markdown(
        '<div class="sidebar-header">'
        '<h1>Document Buddy</h1>'
        '<p>Upload documents and ask questions. The pipeline retrieves, validates, verifies, and reviews every answer.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    user = st.session_state.user
    avatar = user.get("avatar") or DEFAULT_AVATAR
    st.markdown(
        f'<div class="sidebar-profile">'
        f'<div class="sidebar-profile-avatar"><img src="{avatar}" /></div>'
        f'<div class="sidebar-profile-details">'
        f'<span class="sidebar-profile-name">Hi, {user.get("name")}</span>'
        f'<span class="sidebar-profile-email">{user.get("email", "")}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    if st.button("Logout", key="logout_btn", use_container_width=True):
        del st.session_state["user"]
        st.query_params.clear()
        st.rerun()

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
                    with st.spinner(f"Processing '{uf.name}'..."):
                        indexed = st.session_state.orchestrator._retrieval.index(chunks)
                        st.session_state.document_chunks.extend(chunks)
                        st.session_state.uploaded_names.add(uf.name)
                        st.session_state.banners.append({
                            "type": "success", 
                            "text": f"Document '{uf.name}' processed successfully. {len(chunks)} chunks indexed."
                        })
                except ValueError as exc:
                    st.session_state.banners.append({"type": "error", "text": f"Could not read '{uf.name}': {exc}"})
                except RuntimeError as exc:
                    st.session_state.banners.append({"type": "error", "text": f"Processing failed for '{uf.name}'."})
                except Exception as exc:
                    st.session_state.banners.append({"type": "error", "text": f"Unexpected error processing '{uf.name}'."})

    if st.session_state.uploaded_names:
        st.markdown("**Indexed documents**")
        for name in sorted(st.session_state.uploaded_names):
            st.markdown(f'<div class="doc-item">{name}</div>', unsafe_allow_html=True)

        if st.button("Clear all documents"):
            st.session_state.orchestrator._retrieval.clear()
            st.session_state.document_chunks = []
            st.session_state.uploaded_names = set()
            st.session_state.banners.append({"type": "info", "text": "All documents cleared."})
            st.rerun()
    else:
        st.markdown(
            '<p style="color:#78909C;font-size:0.8rem;">No documents uploaded yet.</p>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("### Configuration")

    with st.expander("API Credentials & Endpoints", expanded=False):
        cfg = st.session_state.cfg

        st.markdown(
            '<p style="color:var(--gemini-blue);font-size:0.78rem;font-weight:600;">LLM</p>',
            unsafe_allow_html=True,
        )
        cfg_ollama_url = st.text_input(
            "LLM Endpoint URL",
            value=cfg["OLLAMA_API_URL"],
            help="The API URL of your LLM generation endpoint.",
        )
        cfg_llm_api_version = st.text_input(
            "LLM API Version",
            value=cfg["LLM_API_VERSION"],
            placeholder="e.g. v1 (optional)",
        )
        cfg_llm_api_key = st.text_input(
            "LLM API Key",
            value=cfg["LLM_API_KEY"],
            type="password",
            placeholder="e.g. sk-... (optional)",
        )
        cfg_model = st.text_input(
            "LLM Model Name",
            value=cfg["MODEL_NAME"],
        )

        st.markdown(
            '<p style="color:var(--gemini-blue);font-size:0.78rem;font-weight:600;margin-top:0.6rem;">Web Search</p>',
            unsafe_allow_html=True,
        )
        cfg_tavily_key = st.text_input(
            "Tavily API Key",
            value=cfg["TAVILY_API_KEY"],
            type="password",
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
            if "orchestrator" in st.session_state:
                del st.session_state["orchestrator"]
            st.session_state.config_saved = True
            st.rerun()

    st.markdown("")
    st.markdown("### System Diagnostics")
    tavily_configured = bool(st.session_state.cfg.get("TAVILY_API_KEY", "").strip())
    docs_indexed = len(st.session_state.document_chunks)
    model_name = st.session_state.cfg.get("MODEL_NAME", "Unknown")
    llm_url = st.session_state.cfg.get("OLLAMA_API_URL", "")
    llm_type = "OpenAI Compatible" if "/v1" in llm_url else "Ollama Local"

    st.markdown(
        f'<div class="diagnostics-panel">'
        f'<div class="diagnostic-row"><span class="diagnostic-label">Web Search</span><span class="diagnostic-value" style="color: {"#2E7D32" if tavily_configured else "#D32F2F"};">{"Active" if tavily_configured else "Not Configured"}</span></div>'
        f'<div class="diagnostic-row"><span class="diagnostic-label">Documents Uploaded</span><span class="diagnostic-value">{len(st.session_state.uploaded_names)}</span></div>'
        f'<div class="diagnostic-row"><span class="diagnostic-label">LLM Provider</span><span class="diagnostic-value">{llm_type}</span></div>'
        f'<div class="diagnostic-row"><span class="diagnostic-label">Active Model</span><span class="diagnostic-value">{model_name}</span></div>'
        f'<div class="diagnostic-row"><span class="diagnostic-label">OAuth Provider</span><span class="diagnostic-value" style="text-transform: capitalize;">{user.get("provider", "mock")}</span></div>'
        f'</div>',
        unsafe_allow_html=True
    )

if st.session_state.get("banners"):
    for banner in st.session_state.banners:
        if banner["type"] == "success":
            st.success(banner["text"])
        elif banner["type"] == "error":
            st.error(banner["text"])
        else:
            st.info(banner["text"])
    st.session_state.banners = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("agent_trace"):
            with st.expander("Agent pipeline trace"):
                for i, step in enumerate(msg["agent_trace"], 1):
                    st.markdown(f"`{i}.` {step}")

missing_items = []
if not st.session_state.uploaded_names:
    missing_items.append("document upload")

cfg = st.session_state.cfg
endpoint_url = cfg.get("OLLAMA_API_URL", "").strip()
if not endpoint_url:
    missing_items.append("endpoint URL")

api_key = cfg.get("LLM_API_KEY", "").strip()
is_local_ollama = "localhost" in endpoint_url or "127.0.0.1" in endpoint_url
is_v1 = "/v1" in endpoint_url
if not api_key and (is_v1 or not is_local_ollama):
    missing_items.append("API key")

is_chat_disabled = len(missing_items) > 0
if is_chat_disabled:
    st.warning(f"⚠️ Chat is disabled. Please configure the following mandatory requirements: {', '.join(missing_items)}.")

user_input = st.chat_input(
    "Ask a question about your documents..." if not is_chat_disabled else "Chat is disabled due to missing configuration or document.",
    disabled=is_chat_disabled
)

if user_input:
    user_input = user_input.strip()
    if not user_input:
        st.stop()

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
        st.session_state.banners.append({"type": "error", "text": "No documents uploaded and web search is not configured."})
    else:
        status_placeholder = st.empty()
        def update_status(text):
            status_placeholder.markdown(f"**{text}**")
            
        try:
            answer_data = st.session_state.orchestrator.run(
                query=user_input,
                document_chunks=st.session_state.document_chunks,
                conversation_history=st.session_state.messages[:-1],
                config=st.session_state.cfg,
                status_callback=update_status
            )
            if answer_data["from_web"]:
                st.session_state.banners.append({"type": "info", "text": "Answer sourced from web search."})
        except RuntimeError as exc:
            st.session_state.banners.append({"type": "error", "text": f"Error: {exc}"})
            answer_data = {
                "answer": str(exc),
                "sources": [],
                "agent_trace": [],
                "from_web": False,
            }
        except Exception as exc:
            st.session_state.banners.append({"type": "error", "text": "An unexpected error occurred."})
            answer_data = {
                "answer": "An unexpected error occurred. Please try again.",
                "sources": [],
                "agent_trace": [],
                "from_web": False,
            }
        finally:
            status_placeholder.empty()

    with st.chat_message("assistant"):
        st.markdown(answer_data["answer"])
        

        if answer_data.get("agent_trace"):
            with st.expander("Agent pipeline trace"):
                for i, step in enumerate(answer_data["agent_trace"], 1):
                    st.markdown(f"`{i}.` {step}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_data["answer"],
            "sources": answer_data.get("sources", []),
            "agent_trace": answer_data.get("agent_trace", []),
        }
    )
