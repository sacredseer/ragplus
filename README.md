# RAG Assistant: Agentic sequential multi-agent RAG pipeline

An advanced Retrieval-Augmented Generation (RAG) assistant designed for parsing, validating, fact-checking, and answering questions about local documents. Built as a sequential multi-agent state machine orchestrated by **LangGraph** and presented through a custom **Material Minimalist** Streamlit interface.

## Tech Stack & Libraries
- **Core Orchestrator**: LangGraph (`StateGraph`), LangChain Core
- **Frontend UI**: Streamlit (with custom HSL CSS-variable themes)
- **Natural Language Processing**: Scikit-Learn (`TfidfVectorizer`)
- **Document Parsing**: PyMuPDF (Fitz) for PDFs, Pandas for CSVs, UTF-8 parser for Markdown/Plain Text
- **Fallback Web Search**: Tavily API

---

## Technical Logic Flow

The execution graph is structured as a sequential LangGraph pipeline:

```
                  ┌──────────────────────┐
                  │   User Text Input    │
                  └──────────┬───────────┘
                             │
                             ▼
                 ┌──────────────────────┐
                 │ 1. Resolve Query     │
                 └──────────┬───────────┘
                             │
                             ▼
                 ┌──────────────────────┐
                 │ 2. Retrieval Agent   │
                 └──────────┬───────────┘
                             │
                             ▼
                 ┌──────────────────────┐
                 │ 3. Validation Agent  │
                 └──────────┬───────────┘
                             │
                             ├──────────────────────────────┐
                 [If Relevant]                        [If Irrelevant]
                             │                              │
                             ▼                              ▼
                 ┌──────────────────────┐        ┌──────────────────────┐
                 │ 5. Answer Generation │        │ 4. Web Search Agent  │
                 └──────────┬───────────┘        └──────────┬───────────┘
                             │                              │
                             │                       [If Search Empty]
                             │                              │
                             │                              ▼
                             │                    [Terminate Pipeline]
                             │                    (No relevant content)
                             │                              │
                             ├◄─────────────────────────────┘
                             │
                             ▼
                 ┌──────────────────────┐
                 │ 6. Verification Agent│
                 └──────────┬───────────┘
                             │
                  ┌──────────┴──────────┐
           [If verified]         [If issues found]
                  │                     │
                  ▼                     ▼
                 │             ┌──────────────────────┐
                 │             │ 6b. Grounded Draft   │
                 │             └────────┬─────────────┘
                 │                      │
                 ▼◄─────────────────────┘
                 │
                 ▼
                 ┌──────────────────────┐
                 │ 7. Review Agent      │
                 └──────────┬───────────┘
                             │
                             ▼
                 ┌──────────────────────┐
                 │   Rendered Response  │
                 └──────────────────────┘
```

### Orchestrator Node Actions
1. **Query Resolution**: Uses conversation context to rewrite follow-up questions into standalone queries.
2. **Retrieval Agent**: Runs an optimized cosine similarity lookup against a pre-computed TF-IDF index.
3. **Validation Agent**: Runs relevance checks on document snippets to determine if they contain sufficient answers.
4. **Web Search Agent**: Falls back to Tavily search when documents lack context. If search is empty or disabled, the pipeline terminates immediately with a user-friendly notice.
5. **Answer Generation**: Generates an initial answer using LLM prompts.
6. **Verification Agent**: Fact-checks and checks for hallucinations. If unsupported assertions are detected, the **Grounded Draft** node strictly regenerates fact-based answers.
7. **Review Agent**: Formats, polishes style, and produces the finalized response.

---

## Installation & Setup

### Requirements
- **Python**: `>=3.10` and `<4.0` (fully compatible up to Python `3.14`)

### Local Setup
1. Clone the repository to your machine.
2. Setup a virtual environment and install the required dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```

### Execution
Run the Streamlit application using the Python module execution flag (recommended if executable paths are not in your system environment PATH):
```bash
python -m streamlit run app.py
```

---

## Configuration Settings
Configuration is handled dynamically per session in the sidebar:
- **LLM Endpoint URL**: Supports local Ollama servers (e.g., `http://localhost:11434/api/generate`) and OpenAI-compatible API hosts (e.g., `https://ollama.com/v1`).
- **LLM API Key**: Provide authentication tokens (such as Tavily API keys or hosted model keys).
- **Model Name**: Identify the model target payload parameter (e.g. `mistral` or custom identifiers).

## Deploying to Streamlit Community Edition
This repository is configured out-of-the-box for Streamlit Community Edition:
1. Push the code repository to your GitHub account.
2. Connect your GitHub repository to Streamlit Community Edition.
3. Set your private credentials (e.g. `TAVILY_API_KEY`, custom endpoints) securely in the **Secrets** section of the Streamlit dashboard. The application will automatically pick them up securely from the environment.