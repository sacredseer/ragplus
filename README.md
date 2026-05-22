## Overall Flow
```bash
User query
    │
    ▼  Stage 1 — Query Resolution
         (rewrites follow-up questions into standalone queries)
    │
    ▼  Stage 2 — Retrieval Agent
         (TF-IDF search over uploaded document chunks)
    │
    ▼  Stage 3 — Validation Agent
         (score-based + LLM check: is content relevant?)
    │
    ├─ relevant ──▶  Stage 5 — Answer Generation
    │
    └─ not relevant ▶  Stage 4 — Web Search Agent (Tavily)
                           │
                           └─ no results → "No information found"
    │
    ▼  Stage 6 — Verification Agent
         (LLM checks draft is 100% grounded in sources)
    │
    ▼  Stage 7 — Review Agent
         (polishes, formats, keeps conversation context)
    │
    ▼  Final answer + source badges + collapsible trace
```

## Running this tool
This repository is compatible with Python 3.14 and later Python 3.x versions below 4.0.

```bash
poetry run streamlit run app.py
```