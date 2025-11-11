"""
Base retriever interface following Interface Segregation Principle.

This module defines abstract interfaces for retrieval components,
enabling flexible composition and testing.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """
    Standardized retrieval result.

    Attributes:
        doc_id: Document identifier
        score: Relevance score
        text: Document text content
        metadata: Additional document metadata
    """

    doc_id: str
    score: float
    text: str
    metadata: Dict


class BaseRetriever(ABC):
    """
    Abstract base class for retrieval strategies.

    SOLID Principles:
    - Interface Segregation: Minimal interface for retrieval
    - Dependency Inversion: Clients depend on this abstraction
    - Open/Closed: Extensible without modification
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve most relevant documents for query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of retrieval results sorted by relevance
        """
        pass


class VectorRetriever(ABC):
    """Interface for vector-based retrieval."""

    @abstractmethod
    def search_vectors(
        self, query_vector: List[float], top_k: int = 10, filters: Dict = None
    ) -> List[RetrievalResult]:
        """
        Search using vector similarity.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results
            filters: Optional metadata filters

        Returns:
            List of retrieval results
        """
        pass


class SparseRetriever(ABC):
    """Interface for sparse retrieval (e.g., BM25)."""

    @abstractmethod
    def search_sparse(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search using sparse retrieval method.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of (doc_id, score) tuples
        """
        pass


class Reranker(ABC):
    """Interface for result reranking."""

    @abstractmethod
    def rerank(
        self, query: str, candidates: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Rerank candidate results.

        Args:
            query: Original query
            candidates: Candidate results to rerank

        Returns:
            Reranked results
        """
        pass


class EmbeddingModel(ABC):
    """Interface for embedding generation."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        pass
