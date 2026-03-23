import re
from pathlib import Path
from dataclasses import dataclass, field

import pdfplumber

# -- Result dataclass -- 

@dataclass
class LoadedDocument:
    text:      str
    metadata:  dict          = field(default_factory=dict)
    source:    str           = ""    # original file path as string
    page_count: int          = 0     # 0 for non-PDF sources


# -- Loader --

class DocumentLoader:
    """
    Loads HR policy documents from PDF, HTML, or plain text files
    and returns a normalized LoadedDocument.

    All sources produce the same output shape so the indexer does not
    need to know or care where the text came from.
    """

    def load(self, path: str | Path) -> LoadedDocument:
        """
        Dispatches to the correct loader based on file extension.
        Raises ValueError for unsupported extensions.
        Raises FileNotFoundError if the path does not exist.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found : {path}")

        ext = path.suffix.lower()

        if ext == ".pdf":
            return self._load_pdf(path)
        elif ext in (".html", ".htm"):
            return self._load_html(path)
        elif ext == ".txt":
            return self._load_txt(path)
        else:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported : .pdf, .html, .htm, .txt"
            )

    # -- PDF --

    def _load_pdf(self, path: Path) -> LoadedDocument:
        """
        Extracts text from a PDF using pdfplumber.

        pdfplumber preserves layout information (bounding boxes, font
        sizes) which lets us detect section headers via font size
        heuristics in _detect_sections(). This is more reliable than
        regex on raw text for PDFs that use visual hierarchy rather than
        markdown-style headers.

        Pages that return None from extract_text() are skipped silently —
        this happens on pages that are pure images (scanned documents).
        If all pages are images, text will be empty and the indexer will
        skip this document.
        """
        pages_text = []
        page_count = 0

        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)

            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text.strip())

        full_text = "\n\n".join(pages_text)
        full_text = self._clean_text(full_text)

        return LoadedDocument(
            text=full_text,
            metadata=self._extract_pdf_metadata(path),
            source=str(path),
            page_count=page_count,
        )

    def _extract_pdf_metadata(self, path: Path) -> dict:
        """
        Builds metadata from the filename since PDF internal metadata
        (author, title fields) is unreliable and often empty in HR docs
        downloaded from the web.

        Filename convention assumed : politique_vacances.pdf
        → title  : "politique_vacances"
        → category : inferred from known keywords in the filename
        """
        stem = path.stem   # filename without extension

        return {
            "title":       stem,
            "category":    self._infer_category(stem),
            "source_file": str(path),
            "file_type":   "pdf",
        }

    # -- HTML --

    def _load_html(self, path: Path) -> LoadedDocument:
        """
        Extracts readable text from an HTML file using BeautifulSoup.

        Removes script, style, nav, header, and footer tags before
        extraction — these contain boilerplate that would pollute the
        chunks with navigation text and CSS rather than policy content.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "beautifulsoup4 is required for HTML loading. "
                "Run : pip install beautifulsoup4"
            )

        raw = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "html.parser")

        # Remove noise tags before text extraction
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = self._clean_text(text)

        return LoadedDocument(
            text=text,
            metadata={
                "title":       path.stem,
                "category":    self._infer_category(path.stem),
                "source_file": str(path),
                "file_type":   "html",
            },
            source=str(path),
        )

    # -- Plain text --

    def _load_txt(self, path: Path) -> LoadedDocument:
        text = path.read_text(encoding="utf-8", errors="replace")
        text = self._clean_text(text)

        return LoadedDocument(
            text=text,
            metadata={
                "title":       path.stem,
                "category":    self._infer_category(path.stem),
                "source_file": str(path),
                "file_type":   "txt",
            },
            source=str(path),
        )

    # -- Shared utilities --

    def _clean_text(self, text: str) -> str:
        """
        Normalizes whitespace without destroying paragraph structure.

        Three steps :
          1. Collapse runs of spaces and tabs into a single space
          2. Collapse runs of 3+ newlines into exactly 2 (preserve
             paragraph breaks, remove excessive blank lines)
          3. Strip leading and trailing whitespace from the whole string

        Deliberately does not strip accents or lowercase — that is the
        retriever's job at search time, not the loader's job at index time.
        Storing normalized text would make the original source unreadable
        in citations.
        """
        # Step 1 — collapse horizontal whitespace
        text = re.sub(r"[ \t]+", " ", text)

        # Step 2 — collapse excessive vertical whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Step 3 — strip edges
        return text.strip()

    def _infer_category(self, stem: str) -> str:
        """
        Infers a category label from the filename stem.
        Used to populate the 'category' metadata field in ChromaDB,
        which the retriever can use for metadata filtering.

        Falls back to 'general' if no keyword matches.
        """
        stem_lower = stem.lower()

        category_map = {
            "conges":      ["vacance", "conge", "rti", "ferie"],
            "avantages":   ["avantage", "assurance", "regime", "remboursement"],
            "performance": ["evaluation", "performance", "objectif"],
            "formation":   ["formation", "developpement", "certification"],
            "remuneration":["salaire", "remuneration", "paie", "prime"],
            "sante":       ["sante", "invalidite", "accident"],
            "disciplinaire":["disciplinaire", "avertissement", "congediment"],
        }

        for category, keywords in category_map.items():
            if any(kw in stem_lower for kw in keywords):
                return category

        return "general"