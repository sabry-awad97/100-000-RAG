"""
Hybrid retrieval combining dense vectors, sparse retrieval, and reranking.

This module implements a production-ready hybrid search system using
Reciprocal Rank Fusion and cross-encoder reranking.
"""

from typing import List, Tuple, Dict
import logging

from .base_retriever import (
    BaseRetriever,
    VectorRetriever,
    SparseRetriever,
    Reranker,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval combining multiple search strategies.

    SOLID Principles Applied:
    - Single Responsibility: Orchestrates retrieval methods
    - Open/Closed: Extensible via strategy injection
    - Liskov Substitution: Implements BaseRetriever contract
    - Interface Segregation: Depends only on needed interfaces
    - Dependency Inversion: Depends on abstractions, not implementations

    Attributes:
        vector_retriever: Dense vector search implementation
        sparse_retriever: Sparse retrieval implementation (e.g., BM25)
        embedding_model: Model for encoding queries into embeddings
        reranker: Optional reranker for final scoring
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        sparse_retriever: SparseRetriever,
        embedding_model,
        reranker: Reranker = None,
        fusion_k: int = 60,
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_retriever: Vector search implementation
            sparse_retriever: Sparse search implementation
            embedding_model: Embedding model for query encoding
            reranker: Optional reranker
            fusion_k: RRF constant for rank fusion
        """
        self.vector_retriever = vector_retriever
        self.sparse_retriever = sparse_retriever
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.fusion_k = fusion_k

        # Cache for storing retrieved documents during fusion
        self._doc_cache: Dict[str, RetrievalResult] = {}

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve most relevant documents using hybrid approach.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            Ranked list of relevant documents
        """
        # Clear document cache for new query
        self._doc_cache.clear()

        # Get dense vector results
        dense_results = self._get_dense_results(query, top_k * 2)

        # Cache dense results for later lookup
        for result in dense_results:
            self._doc_cache[result.doc_id] = result

        # Get sparse (BM25) results
        sparse_results = self.sparse_retriever.search_sparse(query, top_k * 2)

        # Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results)

        # Convert to RetrievalResult objects
        candidates = self._build_candidates(fused_results[: top_k * 2])

        # Optional reranking
        if self.reranker:
            candidates = self.reranker.rerank(query, candidates)

        return candidates[:top_k]

    def _get_dense_results(self, query: str, top_k: int) -> List[RetrievalResult]:
        """
        Get dense vector search results.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            Dense retrieval results
        """
        # Embed the query using the injected embedding model
        query_embedding = self.embedding_model.embed(query)

        # Search using vector retriever
        return self.vector_retriever.search_vectors(
            query_vector=query_embedding, top_k=top_k
        )

    def _reciprocal_rank_fusion(
        self, dense: List[RetrievalResult], sparse: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Combine rankings using Reciprocal Rank Fusion.

        RRF formula: score = Σ(1 / (k + rank_i))

        Args:
            dense: Dense retrieval results
            sparse: Sparse retrieval results as (doc_id, score) tuples

        Returns:
            Fused ranking with combined scores
        """
        scores = {}

        # Score dense results
        for rank, result in enumerate(dense, 1):
            doc_id = result.doc_id
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (self.fusion_k + rank)

        # Score sparse results
        for rank, (doc_id, _) in enumerate(sparse, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (self.fusion_k + rank)

        # Sort by combined score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def _build_candidates(
        self, fused_results: List[Tuple[str, float]]
    ) -> List[RetrievalResult]:
        """
        Build RetrievalResult objects from fused rankings.

        Uses cached documents from the retrieval phase to materialize
        RetrievalResult objects with fused scores.

        Args:
            fused_results: List of (doc_id, fused_score) tuples

        Returns:
            List of RetrievalResult objects with fused scores
        """
        candidates = []

        for doc_id, fused_score in fused_results:
            # Look up document from cache (populated during retrieval)
            if doc_id in self._doc_cache:
                cached_doc = self._doc_cache[doc_id]

                # Create new RetrievalResult with fused score
                result = RetrievalResult(
                    doc_id=cached_doc.doc_id,
                    score=fused_score,  # Use fused score instead of original
                    text=cached_doc.text,
                    metadata=cached_doc.metadata,
                )
                candidates.append(result)
            else:
                # Document not in cache - log and skip
                # This can happen if sparse retrieval returns docs not in dense results
                logger.warning(
                    f"Document {doc_id} not found in cache during fusion. "
                    "This may indicate sparse retrieval returned documents "
                    "not available in the vector store."
                )
                continue

        return candidates


class ReciprocalRankFusion:
    """
    Standalone Reciprocal Rank Fusion implementation.

    Can be used independently for combining any ranked lists.
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF.

        Args:
            k: RRF constant (typically 60)
        """
        self.k = k

    def fuse(self, *ranked_lists: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Fuse multiple ranked lists.

        Args:
            *ranked_lists: Variable number of ranked lists as (id, score) tuples

        Returns:
            Fused ranking
        """
        scores = {}

        for ranked_list in ranked_lists:
            for rank, (item_id, _) in enumerate(ranked_list, 1):
                scores[item_id] = scores.get(item_id, 0) + 1 / (self.k + rank)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
