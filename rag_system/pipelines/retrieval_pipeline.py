"""
End-to-end retrieval pipeline.

This module orchestrates the complete retrieval workflow:
query processing, hybrid search, caching, and monitoring.
"""

from typing import List, Dict, Optional
import logging

from ..core.retrieval import BaseRetriever, RetrievalResult
from ..core.cache import SemanticCache
from ..core.monitoring import RAGMonitor
from ..config import settings


logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """
    End-to-end retrieval workflow.

    SOLID Principles:
    - Single Responsibility: Orchestrates retrieval workflow
    - Open/Closed: Extensible via dependency injection
    - Dependency Inversion: Depends on abstractions

    Attributes:
        retriever: Retrieval implementation
        cache: Optional semantic cache
        monitor: Optional monitoring instance
        embedding_model: Embedding model for queries
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        embedding_model: object,
        cache: Optional[SemanticCache] = None,
        monitor: Optional[RAGMonitor] = None,
    ):
        """
        Initialize retrieval pipeline.

        Args:
            retriever: Retrieval implementation
            embedding_model: Embedding model
            cache: Optional semantic cache
            monitor: Optional monitor
        """
        self.retriever = retriever
        self.embedding_model = embedding_model
        self.cache = cache
        self.monitor = monitor

    def retrieve(self, query: str, top_k: int = None, use_cache: bool = True) -> Dict:
        """
        Retrieve relevant documents for query.

        Args:
            query: User query
            top_k: Number of results (uses config default if None)
            use_cache: Whether to use cache

        Returns:
            Dictionary with results and metadata
        """
        top_k = top_k or settings.retrieval.top_k

        # Start monitoring
        if self.monitor:
            self.monitor.start_query()

        # Generate query embedding
        query_embedding = self.embedding_model.embed(query)

        # Check cache
        cache_hit = False
        if use_cache and self.cache:
            cached_result = self.cache.get(query, query_embedding)
            if cached_result:
                logger.info(f"Cache hit for query: {query[:50]}...")
                cache_hit = True
                results = cached_result["results"]

                # Log to monitor
                if self.monitor:
                    self.monitor.log_query(
                        query=query, retrieved_docs=results, cache_hit=True
                    )

                return {
                    "query": query,
                    "results": results,
                    "cache_hit": True,
                    "similarity": cached_result.get("similarity", 1.0),
                    "num_results": len(results),
                }

        # Retrieve from retriever
        try:
            retrieval_results = self.retriever.retrieve(query, top_k)

            # Convert to dictionaries
            results = [
                {
                    "doc_id": r.doc_id,
                    "score": r.score,
                    "text": r.text,
                    "metadata": r.metadata,
                }
                for r in retrieval_results
            ]

            # Cache results
            if use_cache and self.cache:
                self.cache.set(query, query_embedding, results)

            # Log to monitor
            if self.monitor:
                self.monitor.log_query(
                    query=query, retrieved_docs=results, cache_hit=cache_hit
                )

            return {
                "query": query,
                "results": results,
                "cache_hit": False,
                "num_results": len(results),
            }

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")

            if self.monitor:
                self.monitor.log_error(e, {"query": query})

            raise

    def batch_retrieve(self, queries: List[str], top_k: int = None) -> List[Dict]:
        """
        Retrieve for multiple queries.

        Args:
            queries: List of queries
            top_k: Number of results per query

        Returns:
            List of retrieval results
        """
        return [self.retrieve(query, top_k) for query in queries]
