import io

import fitz
import pandas as pd

from ..util.utilities import chunk_text


class DocumentAgent:
    """Parses uploaded documents (PDF, CSV, TXT, Markdown) into structured text chunks."""

    def process(self, file_name: str, file_bytes: bytes) -> list:
        ext = file_name.rsplit(".", 1)[-1].lower()

        try:
            text = self._extract_text(file_name, file_bytes, ext)
        except Exception as exc:
            raise ValueError(f"Could not read content from '{file_name}': {exc}") from exc

        if not text.strip():
            raise ValueError(f"The file '{file_name}' appears to be empty or unreadable.")

        chunks = chunk_text(text)
        return [
            {"source": file_name, "content": chunk, "chunk_index": i}
            for i, chunk in enumerate(chunks)
        ]

    def _extract_text(self, file_name: str, file_bytes: bytes, ext: str) -> str:
        if ext == "pdf":
            return self._parse_pdf(file_bytes)
        if ext == "csv":
            return self._parse_csv(file_bytes)
        if ext in ("txt", "md", "markdown"):
            return file_bytes.decode("utf-8", errors="ignore")
        
        return file_bytes.decode("utf-8", errors="ignore")

    def _parse_pdf(self, file_bytes: bytes) -> str:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return " ".join(page.get_text() for page in doc)

    def _parse_csv(self, file_bytes: bytes) -> str:
        df = pd.read_csv(io.BytesIO(file_bytes))
        return df.to_string(index=False)
