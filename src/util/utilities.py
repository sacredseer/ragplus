import os
import requests
import json
from requests.exceptions import JSONDecodeError

_DEFAULTS = {
    "OLLAMA_API_URL": "http://localhost:11434/api/generate",
    "MODEL_NAME": "mistral",
    "LLM_API_VERSION": "",
    "LLM_API_KEY": "",
}


def call_llm(prompt: str, system: str = None, config: dict = None) -> str:
    """
    Calls the configured LLM API endpoint and streams the response.
    
    Supports both Ollama's native `/api/generate` format and OpenAI's 
    standard `/v1/chat/completions` formats. Config is passed down per-session.
    """
    cfg = config or {}
    api_url = (cfg.get("OLLAMA_API_URL") or _DEFAULTS["OLLAMA_API_URL"]).strip()
    model_name = (cfg.get("MODEL_NAME") or _DEFAULTS["MODEL_NAME"]).strip()
    api_version = (cfg.get("LLM_API_VERSION") or _DEFAULTS["LLM_API_VERSION"]).strip()
    api_key = (cfg.get("LLM_API_KEY") or _DEFAULTS["LLM_API_KEY"]).strip()

    if "/v1" in api_url and not api_url.endswith("/chat/completions"):
        api_url = api_url.rstrip("/") + "/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_version:
        headers["X-API-Version"] = api_version
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    is_openai_format = "/v1" in api_url
    if is_openai_format:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": True
        }
    else:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": True
        }
        if system:
            payload["system"] = system

    try:
        response = requests.post(api_url, headers=headers, json=payload, stream=True, timeout=120)
        response.raise_for_status()

        collected_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                if line.startswith("data: "):
                    line = line[len("data: "):].strip()
                if line == "[DONE]":
                    break
                    
                data = json.loads(line)
                
                if "response" in data:
                    collected_text += data.get("response", "")
                    if data.get("done"):
                        break
                elif "choices" in data and len(data["choices"]) > 0:
                    delta = data["choices"][0].get("delta", {})
                    collected_text += delta.get("content", "")
                    if data["choices"][0].get("finish_reason") is not None:
                        break
            except (JSONDecodeError, KeyError):
                continue

        return collected_text.strip()

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Unable to connect to the LLM service at '{api_url}'. "
            "Please check if the service is running and the endpoint is accessible."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"LLM API request failed: {exc}") from exc


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list:
    """Splits a document text into overlapping words-based blocks for retrieval."""
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks
