import io
import logging

import fitz
import pandas as pd

from utilities import chunk_text

logger = logging.getLogger(__name__)


class DocumentAgent:
    """Parses uploaded documents (PDF, CSV, TXT, Markdown) into structured text chunks."""

    def process(self, file_name: str, file_bytes: bytes) -> list:
        """
        Parses raw file bytes, extracts its textual content, and splits it into chunks.
        
        Returns a list of dictionaries where each item represents a chunk:
        {
            "source": file_name,
            "content": chunk_text,
            "chunk_index": index
        }
        """
        ext = file_name.rsplit(".", 1)[-1].lower()

        try:
            text = self._extract_text(file_name, file_bytes, ext)
        except Exception as exc:
            logger.error("Failed text extraction from file %s: %s", file_name, exc)
            raise ValueError(f"Could not read content from '{file_name}': {exc}") from exc

        if not text.strip():
            raise ValueError(f"The file '{file_name}' appears to be empty or unreadable.")

        chunks = chunk_text(text)
        return [
            {"source": file_name, "content": chunk, "chunk_index": i}
            for i, chunk in enumerate(chunks)
        ]

    def _extract_text(self, file_name: str, file_bytes: bytes, ext: str) -> str:
        # Route to the appropriate parser based on file extension
        if ext == "pdf":
            return self._parse_pdf(file_bytes)
        if ext == "csv":
            return self._parse_csv(file_bytes)
        if ext in ("txt", "md", "markdown"):
            return file_bytes.decode("utf-8", errors="ignore")
        
        # Log a warning and fallback to decoding as plain text
        logger.warning("Unknown file extension '%s' for %s. Attempting UTF-8 fallback.", ext, file_name)
        return file_bytes.decode("utf-8", errors="ignore")

    def _parse_pdf(self, file_bytes: bytes) -> str:
        # Extract plain text from all pages in the PDF document
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return " ".join(page.get_text() for page in doc)

    def _parse_csv(self, file_bytes: bytes) -> str:
        # Convert tabular CSV data to a readable string representation
        df = pd.read_csv(io.BytesIO(file_bytes))
        return df.to_string(index=False)
