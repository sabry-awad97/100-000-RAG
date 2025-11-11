"""
Cross-encoder reranking for deep semantic relevance scoring.

This module provides production-ready cross-encoder reranking
with configurable models and batch processing.
"""

from typing import List
from .base_retriever import Reranker, RetrievalResult


class CrossEncoderReranker(Reranker):
    """
    Cross-encoder based reranking.

    SOLID Principles:
    - Single Responsibility: Handles only reranking logic
    - Open/Closed: Extensible for different cross-encoder models
    - Liskov Substitution: Implements Reranker interface
    - Dependency Inversion: Can work with any cross-encoder implementation

    Attributes:
        model: Cross-encoder model instance
        ce_weight: Weight for cross-encoder scores (0-1)
        fusion_weight: Weight for original fusion scores (0-1)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ce_weight: float = 0.7,
        fusion_weight: float = 0.3,
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model identifier
            ce_weight: Weight for cross-encoder scores
            fusion_weight: Weight for original scores
        """
        self.model_name = model_name
        self.ce_weight = ce_weight
        self.fusion_weight = fusion_weight
        self._model = None  # Lazy loading

    @property
    def model(self):
        """Lazy load cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, candidates: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Rerank candidates using cross-encoder.

        Args:
            query: Original query
            candidates: Candidate results to rerank

        Returns:
            Reranked results with updated scores
        """
        if not candidates:
            return candidates

        # Prepare query-document pairs
        pairs = [[query, candidate.text] for candidate in candidates]

        # Score with cross-encoder
        ce_scores = self.model.predict(pairs)

        # Combine with original scores
        reranked = []
        for candidate, ce_score in zip(candidates, ce_scores):
            combined_score = (
                self.ce_weight * ce_score + self.fusion_weight * candidate.score
            )

            # Create new result with updated score
            reranked_result = RetrievalResult(
                doc_id=candidate.doc_id,
                score=combined_score,
                text=candidate.text,
                metadata={
                    **candidate.metadata,
                    "original_score": candidate.score,
                    "ce_score": float(ce_score),
                    "combined_score": combined_score,
                },
            )
            reranked.append(reranked_result)

        # Sort by combined score
        reranked.sort(key=lambda x: x.score, reverse=True)

        return reranked

    def batch_rerank(
        self, queries: List[str], candidates_list: List[List[RetrievalResult]]
    ) -> List[List[RetrievalResult]]:
        """
        Rerank multiple query-candidate sets in batch.

        Args:
            queries: List of queries
            candidates_list: List of candidate lists

        Returns:
            List of reranked candidate lists
        """
        return [
            self.rerank(query, candidates)
            for query, candidates in zip(queries, candidates_list)
        ]
