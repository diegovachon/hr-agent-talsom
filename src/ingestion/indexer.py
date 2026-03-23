from pathlib import Path
from dataclasses import dataclass, field

import chromadb
from sentence_transformers import SentenceTransformer

from src.ingestion.loader import DocumentLoader, LoadedDocument

# -- Configuration --

EMBEDDING_MODEL    = "nomic-ai/nomic-embed-text-v1"
CHUNK_SIZE         = 150    # words per chunk (was 400)
CHUNK_OVERLAP      = 30     # words of overlap between consecutive chunks (was 80)
MIN_CHUNK_WORDS    = 20     # chunks shorter than this are discarded (was 30)


# -- Result dataclass --
@dataclass
class IndexingResult:
    source:         str
    chunks_indexed: int
    chunks_skipped: int
    error:          str = ""


# -- Indexer --

class DocumentIndexer:

    def __init__(self, chroma_path: str):
        self.loader   = DocumentLoader()
        self.embedder = SentenceTransformer(
            EMBEDDING_MODEL,
            trust_remote_code=True,
        )
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name="hr_policies",
            metadata={"hnsw:space": "cosine"},
        )

    # -- Public interface --

    def index_file(self, path: str | Path) -> IndexingResult:
        """
        Loads, chunks, embeds, and stores a single document.
        Returns an IndexingResult describing what happened.
        Never raises — all errors are captured in IndexingResult.error.
        """
        path = Path(path)

        try:
            document = self.loader.load(path)
        except Exception as e:
            return IndexingResult(
                source=str(path),
                chunks_indexed=0,
                chunks_skipped=0,
                error=f"Loading failed : {e}",
            )

        if not document.text.strip():
            return IndexingResult(
                source=str(path),
                chunks_indexed=0,
                chunks_skipped=0,
                error="Document is empty or could not be extracted (scanned PDF?).",
            )

        chunks   = self._chunk(document)
        filtered = self._filter(chunks)
        skipped  = len(chunks) - len(filtered)

        if not filtered:
            return IndexingResult(
                source=str(path),
                chunks_indexed=0,
                chunks_skipped=skipped,
                error="All chunks were below the minimum word count.",
            )

        self._embed_and_store(filtered, document)

        return IndexingResult(
            source=str(path),
            chunks_indexed=len(filtered),
            chunks_skipped=skipped,
        )

    def index_directory(self, directory: str | Path) -> list[IndexingResult]:
        """
        Indexes all supported files in a directory non-recursively.
        Returns one IndexingResult per file found.
        """
        directory = Path(directory)
        supported = (".pdf", ".html", ".htm", ".txt")
        results   = []

        files = [f for f in directory.iterdir() if f.suffix.lower() in supported]

        if not files:
            print(f"No supported files found in {directory}")
            return results

        for file in sorted(files):
            print(f"Indexing {file.name} ...", end=" ", flush=True)
            result = self.index_file(file)

            if result.error:
                print(f"ERROR — {result.error}")
            else:
                print(f"{result.chunks_indexed} chunks indexed, "
                      f"{result.chunks_skipped} skipped.")

            results.append(result)

        return results

    def delete_document(self, title: str) -> int:
        """
        Deletes all chunks from ChromaDB whose metadata title matches.
        Returns the number of chunks deleted.
        Useful when re-indexing an updated policy document.
        """
        existing = self.collection.get(
            where={"title": title},
            include=[],
        )
        ids_to_delete = existing["ids"]

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)

        return len(ids_to_delete)

    # -- Chunking --

    def _chunk(self, document: LoadedDocument) -> list[dict]:
        """
        Splits document text into overlapping word-window chunks.

        Each chunk is a dict with :
          - text     : the raw chunk text
          - chunk_id : deterministic id built from title + chunk index
          - metadata : document metadata enriched with chunk position info

        Overlap ensures that sentences spanning a chunk boundary are
        fully present in at least one chunk. Without overlap, a policy
        rule that starts at the end of one chunk and finishes at the
        start of the next would never be retrieved as a complete unit.
        """
        words  = document.text.split()
        chunks = []
        index  = 0
        i      = 0

        while i < len(words):
            window = words[i : i + CHUNK_SIZE]
            text   = " ".join(window)

            chunk_id = f"{document.metadata.get('title', 'doc')}_{index}"

            chunks.append({
                "text":     text,
                "chunk_id": chunk_id,
                "metadata": {
                    **document.metadata,
                    "section":     f"chunk_{index}",
                    "chunk_index": index,
                    "word_start":  i,
                    "word_end":    i + len(window),
                },
            })

            index += 1
            i     += CHUNK_SIZE - CHUNK_OVERLAP

        return chunks

    def _filter(self, chunks: list[dict]) -> list[dict]:
        """
        Removes chunks that are too short to be meaningful.

        The last chunk of a document is often a fragment — a few words
        from the final paragraph that didn't fill a full window. Indexing
        a 12-word chunk wastes an embedding slot and can produce misleading
        retrieval results where a fragment scores highly due to term overlap
        with the query but contains no actionable information.
        """
        return [
            c for c in chunks
            if len(c["text"].split()) >= MIN_CHUNK_WORDS
        ]

    # -- Embedding and storage --

    def _embed_and_store(
        self,
        chunks:   list[dict],
        document: LoadedDocument,
    ) -> None:
        """
        Embeds all chunks in one batched call and writes them to ChromaDB.

        Batching matters here — calling embedder.encode() once for a list
        of 20 chunks is significantly faster than calling it 20 times for
        individual chunks, because the sentence-transformers library can
        parallelize the forward pass across the batch on CPU.

        Existing chunks with the same ids are overwritten automatically
        by ChromaDB's upsert semantics — re-indexing a document that was
        already indexed does not create duplicates.
        """
        texts      = [c["text"]     for c in chunks]
        ids        = [c["chunk_id"] for c in chunks]
        metadatas  = [c["metadata"] for c in chunks]

        # Single batched encode call
        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=len(chunks) > 10,
            batch_size=32,
        ).tolist()

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )