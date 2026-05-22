import io
import logging

import fitz
import pandas as pd

from utilities import chunk_text

logger = logging.getLogger(__name__)


class DocumentAgent:
    """Parses uploaded files (PDF, CSV, TXT, Markdown) into text chunks."""

    def process(self, file_name: str, file_bytes: bytes) -> list:
        """
        Extract text from a file and split it into overlapping chunks.

        Returns a list of dicts: {source, content, chunk_index}
        Raises ValueError if the file type is unsupported or the file is empty.
        """
        ext = file_name.rsplit(".", 1)[-1].lower()

        try:
            text = self._extract_text(file_name, file_bytes, ext)
        except Exception as exc:
            logger.error("Failed to extract text from %s: %s", file_name, exc)
            raise ValueError(f"Could not read '{file_name}': {exc}") from exc

        if not text.strip():
            raise ValueError(f"'{file_name}' appears to be empty or unreadable.")

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
        logger.warning("Unrecognised extension '%s' for %s; treating as plain text.", ext, file_name)
        return file_bytes.decode("utf-8", errors="ignore")

    def _parse_pdf(self, file_bytes: bytes) -> str:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return " ".join(page.get_text() for page in doc)

    def _parse_csv(self, file_bytes: bytes) -> str:
        df = pd.read_csv(io.BytesIO(file_bytes))
        return df.to_string(index=False)
