import os
import logging
import requests
import json
from urllib.parse import urlparse, urlunparse
from requests.exceptions import JSONDecodeError

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "OLLAMA_API_URL": "http://localhost:11434/api/generate",
    "MODEL_NAME": "mistral",
    "LLM_API_VERSION": "",
    "LLM_API_KEY": "",
    
}


def _cfg(key: str) -> str:
    """Read a config value from os.environ at call-time so updates take effect immediately."""
    return os.environ.get(key, _DEFAULTS.get(key, "")).strip()


def _normalize_llm_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if not path or path == "/":
        if "ollama" in parsed.netloc:
            path = "/api/generate"
    elif path == "/v1":
        if "ollama" in parsed.netloc:
            path = "/api/generate"
    if path != parsed.path:
        parsed = parsed._replace(path=path)
    return urlunparse(parsed)


def call_llm(prompt: str, system: str = None) -> str:
    """Send a prompt to the Ollama API and return the full response text."""
    ollama_api_url = _cfg("OLLAMA_API_URL") or _DEFAULTS["OLLAMA_API_URL"]
    ollama_api_url = _normalize_llm_url(ollama_api_url)
    model_name = _cfg("MODEL_NAME") or _DEFAULTS["MODEL_NAME"]
    llm_api_version = _cfg("LLM_API_VERSION")
    llm_api_key = _cfg("LLM_API_KEY")

    headers = {"Content-Type": "application/json"}
    if llm_api_version:
        headers["X-API-Version"] = llm_api_version
    if llm_api_key:
        headers["Authorization"] = f"Bearer {llm_api_key}"
    payload: dict = {"model": model_name, "prompt": prompt, "stream": True}
    if system:
        payload["system"] = system

    try:
        response = requests.post(ollama_api_url, headers=headers, json=payload, stream=True, timeout=120)
        response.raise_for_status()

        collected = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    data = json.loads(line)
                    collected += data.get("response", "")
                    if data.get("done"):
                        break
                except JSONDecodeError:
                    logger.warning("Non-JSON chunk skipped during streaming.")
                    continue

        return collected.strip()

    except requests.exceptions.ConnectionError as exc:
        logger.error("Cannot connect to Ollama: %s", exc)
        raise RuntimeError(
            "Cannot connect to the LLM service. Ensure Ollama is running at "
            f"{ollama_api_url}."
        ) from exc
    except requests.exceptions.RequestException as exc:
        logger.error("LLM API request failed: %s", exc)
        raise RuntimeError(f"LLM request failed: {exc}") from exc



def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks
