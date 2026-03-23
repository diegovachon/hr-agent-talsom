"""
One-shot script to index all HR policy documents into ChromaDB.

Run this once before starting the application :
    python scripts/ingest_all.py

Run again after adding or updating documents in data/raw/.
Existing chunks are overwritten automatically (upsert semantics).
"""

import sys
from pathlib import Path

# Allow imports from the project root when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.indexer import DocumentIndexer

# -- Configuration --

RAW_DIR     = Path("./data/raw")
CHROMA_PATH = "./data/chroma_db"


# -- Main --

def main():
    print("=" * 60)
    print("HR Policy Indexer")
    print("=" * 60)

    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        print(
            f"\nNo documents found in {RAW_DIR}.\n"
            f"Add PDF, HTML, or TXT files to {RAW_DIR} before indexing.\n"
            f"\nSuggested sources :\n"
            f"  - https://www.rhinfo.com/politiques\n"
            f"  - Search 'politique vacances entreprise filetype:pdf'\n"
            f"  - Any publicly available HR policy document\n"
        )
        sys.exit(1)

    indexer = DocumentIndexer(chroma_path=CHROMA_PATH)

    print(f"\nSource directory : {RAW_DIR.resolve()}")
    print(f"Vector store     : {Path(CHROMA_PATH).resolve()}\n")

    results = indexer.index_directory(RAW_DIR)

    # -- Summary --

    total_indexed = sum(r.chunks_indexed for r in results)
    total_skipped = sum(r.chunks_skipped for r in results)
    total_errors  = sum(1 for r in results if r.error)

    print("\n" + "=" * 60)
    print("Indexing complete")
    print("=" * 60)
    print(f"  Files processed : {len(results)}")
    print(f"  Chunks indexed  : {total_indexed}")
    print(f"  Chunks skipped  : {total_skipped}")
    print(f"  Errors          : {total_errors}")

    if total_errors > 0:
        print("\nFiles with errors :")
        for r in results:
            if r.error:
                print(f"  {r.source} — {r.error}")

    if total_indexed == 0:
        print(
            "\nNo chunks were indexed. The vector store is empty.\n"
            "The application will not be able to answer any questions.\n"
            "Check the errors above and verify your source documents."
        )
        sys.exit(1)

    print(f"\nReady. Run the application with :\n  python -m src.ui.app\n")


if __name__ == "__main__":
    main()