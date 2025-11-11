"""
Combined RAG pipeline for end-to-end query processing.

This module integrates retrieval and generation for complete
RAG functionality with monitoring and caching.
"""

from typing import Dict, Optional
import logging

from .retrieval_pipeline import RetrievalPipeline
from ..core.generation import RAGGenerator
from ..core.monitoring import RAGMonitor
from ..config import settings


logger = logging.getLogger(__name__)


class GenerationPipeline:
    """
    Combined RAG pipeline.

    SOLID Principles:
    - Single Responsibility: Orchestrates RAG workflow
    - Open/Closed: Extensible via dependency injection
    - Dependency Inversion: Depends on abstractions

    Attributes:
        retrieval_pipeline: Retrieval pipeline instance
        generator: RAG generator instance
        monitor: Optional monitoring instance
    """

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        generator: RAGGenerator,
        monitor: Optional[RAGMonitor] = None,
    ):
        """
        Initialize generation pipeline.

        Args:
            retrieval_pipeline: Retrieval pipeline
            generator: RAG generator
            monitor: Optional monitor
        """
        self.retrieval_pipeline = retrieval_pipeline
        self.generator = generator
        self.monitor = monitor

    def query(
        self, query: str, top_k: Optional[int] = None, use_cache: bool = True
    ) -> Dict:
        """
        Process query end-to-end.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            use_cache: Whether to use cache

        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info(f"Processing query: {query[:100]}...")

        # Start monitoring
        if self.monitor:
            self.monitor.start_query()

        try:
            # Retrieve relevant documents
            retrieval_result = self.retrieval_pipeline.retrieve(
                query=query, top_k=top_k, use_cache=use_cache
            )

            # Generate answer
            generation_result = self.generator.generate(
                query=query, retrieved_docs=retrieval_result["results"]
            )

            # Combine results
            result = {
                "query": query,
                "answer": generation_result["answer"],
                "sources": generation_result["sources"],
                "context_used": generation_result["context_used"],
                "num_context_docs": generation_result["num_context_docs"],
                "cache_hit": retrieval_result.get("cache_hit", False),
                "num_retrieved": retrieval_result.get("num_results", 0),
            }

            logger.info(
                f"Successfully processed query with {result['num_context_docs']} context docs"
            )

            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}")

            if self.monitor:
                self.monitor.log_error(e, {"query": query})

            raise

    def query_with_feedback(
        self,
        query: str,
        user_feedback: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> Dict:
        """
        Process query and log user feedback.

        Args:
            query: User query
            user_feedback: Optional user rating (1-5)
            top_k: Number of documents to retrieve

        Returns:
            Query result dictionary
        """
        result = self.query(query, top_k)

        # Log feedback if provided
        if user_feedback and self.monitor:
            self.monitor.log_query(
                query=query,
                retrieved_docs=result["context_used"],
                user_feedback=user_feedback,
                cache_hit=result["cache_hit"],
            )

        return result


class RAGPipeline:
    """
    Complete RAG system pipeline factory.

    This class provides a convenient way to instantiate the full
    RAG pipeline with all components configured.
    """

    @staticmethod
    def create_from_settings() -> GenerationPipeline:
        """
        Create RAG pipeline from settings.

        Returns:
            Configured GenerationPipeline instance
        """
        from ..core.retrieval import HybridRetriever, CrossEncoderReranker
        from ..core.cache import CacheConfig
        from ..core.monitoring import RAGMonitor
        from ..core.generation import RAGGenerator, OpenAIClient

        # Create components
        # Note: This is a simplified example
        # In production, implement proper dependency injection

        logger.info("Creating RAG pipeline from settings")

        # Create retriever (placeholder - implement with actual components)
        # retriever = HybridRetriever(...)

        # Create embedding model (placeholder)
        # embedding_model = ...

        # Create cache if enabled
        cache = None
        if settings.cache.enabled:
            cache_config = CacheConfig(
                redis_host=settings.redis.host,
                redis_port=settings.redis.port,
                redis_db=settings.redis.db,
                similarity_threshold=settings.cache.similarity_threshold,
                default_ttl=settings.cache.ttl,
            )
            cache = cache_config.create_cache()

        # Create monitor if enabled
        monitor = None
        if settings.monitoring.enabled:
            monitor = RAGMonitor()

        # Create retrieval pipeline
        # retrieval_pipeline = RetrievalPipeline(
        #     retriever=retriever,
        #     embedding_model=embedding_model,
        #     cache=cache,
        #     monitor=monitor
        # )

        # Create LLM client
        llm_client = OpenAIClient(api_key=settings.openai.api_key)

        # Create generator
        # generator = RAGGenerator(llm_client=llm_client)

        # Create generation pipeline
        # return GenerationPipeline(
        #     retrieval_pipeline=retrieval_pipeline,
        #     generator=generator,
        #     monitor=monitor
        # )

        raise NotImplementedError("Implement with actual component instantiation")
