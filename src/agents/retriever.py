import numpy as np
from rank_bm25 import BM25Okapi 
import chromadb
from sentence_transformers import SentenceTransformer

# CONFIGURATION
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"
DEFAULT_ALPHA   = 0.6       # weight of dense score vs BM25
                            # 0.6 dense + 0.4 BM25 favors semantic understanding over exact keyword match


class HybridRetriever:

    def __init__(self, chroma_path: str, alpha: float = DEFAULT_ALPHA):
        self.alpha = alpha

        self.embedder = SentenceTransformer(
            EMBEDDING_MODEL,
            trust_remote_code=True,
        )

        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name="hr_policies",
            metadata={"hnsw:space": "cosine"},  # cosine similarity for dense scores
        )

        # BM25 index is built from whatever is currently in ChromaDB
        # If documents are added after initialization, call _build_bm25 again before the next search
        self._build_bm25()

    # Public interface
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Runs hybrid retrieval and returns the top_k most relevant chunks

        Each result is a dict with 3 keys:
          - text     : raw chunk text (str)
          - metadata : title, section, category, source_file (dict)
          - score    : hybrid relevance score between 0.0 and 1.0 (float)

        Returns empty list of collection is empty or if no documents score above zero
        """
        if self._collection_is_empty():
            return []
        
        dense_scores  = self._dense_search(query, top_k)
        bm25_scores   = self._bm25_search(query)
        hybrid_scores = self._fuse(dense_scores, bm25_scores)

        return self._build_results(hybrid_scores, top_k)
    
    def refresh(self) -> None:
        """
        Rebuilds the BM25 index from the current ChromaDB contents

        Call this after adding new documents via the indexer if the 
        retriever instance is already running
        """
        self._build_bm25()

    # DENSE RETRIEVAL

    def _dense_search(self, query: str, top_k: int) -> dict[str, float]:
        """
        Embeds the query and retrieves the closest vectors from ChromaDB
        Returns a dict mapping chunk id -> normalized cosine similarity score
        """
        embedding = self.embedder.encode(query).tolist()

        n_results = min(top_k * 2, self._collection_size())

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        ids = results["ids"][0]
        distances = results["distances"][0]

        # ChromaDB with cosine space returns distances in [0, 2]
        # Convert to similarity in [0, 1] : similarity = 1 - distance/2
        scores = {}
        for id_, dist in zip(ids, distances):
            scores[id_] = round(1 - dist / 2, 4)

        return scores
    
    # BM25 RETRIEVAL

    def _build_bm25(self) -> None:
        """
        Loads all documents from ChromaDB and builds the BM25 index
        """
        if self._collection_is_empty():
            self._bm25_ids  = []
            self._bm25_docs = []
            self._bm25      = None
            return

        all_data = self.collection.get(include=["documents"])

        self._bm25_ids  = all_data["ids"]
        self._bm25_docs = all_data["documents"]

        tokenized = [doc.lower().split() for doc in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized)

    def _bm25_search(self, query: str) -> dict[str, float]:
        """
        Scores all documents against the query using BM25
        Normalizes scores to [0, 1] by dividing by the max score
        Returns a dict mapping chunk id -> normalized BM25 score
        """
        if self._bm25 is None:
            return {}

        raw_scores = self._bm25.get_scores(query.lower().split())
        max_score  = float(np.max(raw_scores)) if raw_scores.any() else 1.0

        if max_score == 0.0:
            return {}

        return {
            id_: round(float(score) / max_score, 4)
            for id_, score in zip(self._bm25_ids, raw_scores)
        }

    # FUSION

    def _fuse(
            self,
            dense_scores: dict[str, float],
            bm25_scores:  dict[str, float],
    ) -> dict[str, float]:
        """
        Combines dense and BM25 scores using weighted linear fusion:
            hybrid = alpha · dense + (1 - alpha) · bm25

        Chunks that only appear in one index get a score of 0.0 for the missing component
        they are not excluded entirely, because a strong BM25 match with no dense result
        or vice versa is still meaningfull
        """
        all_ids = set(dense_scores.keys()) | set(bm25_scores.keys())

        return {
            id_: round(
                self.alpha       * dense_scores.get(id_, 0.0)
                + (1 - self.alpha) * bm25_scores.get(id_,  0.0),
                4,
            )
            for id_ in all_ids
        }
    
    # Result construction

    def _build_results(
        self,
        hybrid_scores: dict[str, float],
        top_k: int,
    ) -> list[dict]:
        """
        Sorts by hybrid score, it fetches full text and metadata for the top_k
        chunks from ChromaDB, and returns a clean list of dicts
        """
        sorted_ids = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)
        top_ids    = sorted_ids[:top_k]

        if not top_ids:
            return []

        # Single ChromaDB fetch for all top ids — one round trip
        fetched = self.collection.get(
            ids=top_ids,
            include=["documents", "metadatas"],
        )

        # collection.get() returns results in the order ChromaDB stores them,
        # not in the order of top_ids. Rebuild a lookup dict first.
        id_to_doc  = dict(zip(fetched["ids"], fetched["documents"]))
        id_to_meta = dict(zip(fetched["ids"], fetched["metadatas"]))

        results = []
        for id_ in top_ids:
            if id_ not in id_to_doc:
                continue    # chunk was deleted from ChromaDB after BM25 was built

            results.append({
                "text":     id_to_doc[id_],
                "metadata": id_to_meta.get(id_, {}),
                "score":    hybrid_scores[id_],
            })

        return results
    
    # UTILITIES

    def _collection_is_empty(self) -> bool:
        return self.collection.count() == 0

    def _collection_size(self) -> int:
        return self.collection.count()