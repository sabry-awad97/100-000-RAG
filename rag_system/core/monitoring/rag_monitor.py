"""
RAG monitoring module for query logging and metric tracking.

This module provides comprehensive monitoring for RAG systems,
tracking retrieval quality, latency, and user feedback.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import time
import json
import logging


@dataclass
class QueryMetrics:
    """
    Metrics for a single query execution.

    Attributes:
        timestamp: Query timestamp
        query: User query text
        num_results: Number of retrieved documents
        top_score: Highest relevance score
        latency_ms: Query latency in milliseconds
        user_feedback: Optional user rating (1-5)
        cache_hit: Whether query was served from cache
        error: Optional error message
    """

    timestamp: float
    query: str
    num_results: int
    top_score: float
    latency_ms: float
    user_feedback: Optional[int] = None
    cache_hit: bool = False
    error: Optional[str] = None


class RAGMonitor:
    """
    Monitor retrieval quality and system performance.

    SOLID Principles:
    - Single Responsibility: Handles only monitoring/logging
    - Open/Closed: Extensible for new metrics
    - Dependency Inversion: Depends on logger abstraction

    Attributes:
        logger: Python logger instance
        metrics_store: Optional metrics storage backend
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        metrics_store: Optional[object] = None,
    ):
        """
        Initialize RAG monitor.

        Args:
            logger: Optional custom logger
            metrics_store: Optional metrics storage backend
        """
        self.logger = logger or self._create_default_logger()
        self.metrics_store = metrics_store
        self._query_start_time = None

    def start_query(self):
        """Start timing a query."""
        self._query_start_time = time.time()

    def log_query(
        self,
        query: str,
        retrieved_docs: List[Dict],
        user_feedback: Optional[int] = None,
        cache_hit: bool = False,
        error: Optional[str] = None,
    ):
        """
        Log query execution for monitoring and analysis.

        Args:
            query: User query
            retrieved_docs: Retrieved documents
            user_feedback: Optional user rating (1-5)
            cache_hit: Whether served from cache
            error: Optional error message
        """
        latency_ms = self._measure_latency()

        metrics = QueryMetrics(
            timestamp=time.time(),
            query=query,
            num_results=len(retrieved_docs),
            top_score=retrieved_docs[0].get("score", 0) if retrieved_docs else 0,
            latency_ms=latency_ms,
            user_feedback=user_feedback,
            cache_hit=cache_hit,
            error=error,
        )

        # Log to logger
        self.logger.info(json.dumps(asdict(metrics)))

        # Store in metrics backend if available
        if self.metrics_store:
            self._store_metrics(metrics)

        # Track retrieval quality metrics
        if user_feedback:
            self._update_quality_metrics(metrics)

    def log_error(self, error: Exception, context: Dict):
        """
        Log error with context.

        Args:
            error: Exception that occurred
            context: Additional context information
        """
        error_data = {
            "timestamp": time.time(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
        }

        self.logger.error(json.dumps(error_data))

    def get_metrics_summary(self, time_window: int = 3600) -> Dict:
        """
        Get metrics summary for time window.

        Args:
            time_window: Time window in seconds

        Returns:
            Dictionary with aggregated metrics
        """
        # This would query the metrics store
        # For now, return placeholder
        return {
            "time_window": time_window,
            "total_queries": 0,
            "avg_latency_ms": 0,
            "cache_hit_rate": 0,
            "avg_user_rating": 0,
        }

    def _measure_latency(self) -> float:
        """
        Measure query latency.

        Returns:
            Latency in milliseconds
        """
        if self._query_start_time is None:
            return 0.0

        latency_s = time.time() - self._query_start_time
        self._query_start_time = None
        return latency_s * 1000

    def _store_metrics(self, metrics: QueryMetrics):
        """
        Store metrics in backend.

        Args:
            metrics: Query metrics to store
        """
        # Implement based on metrics_store type
        # Could be database, time-series DB, etc.
        pass

    def _update_quality_metrics(self, metrics: QueryMetrics):
        """
        Update retrieval quality metrics.

        Args:
            metrics: Query metrics with user feedback
        """
        # Track quality metrics like precision, recall, MRR
        # Based on user feedback
        pass

    @staticmethod
    def _create_default_logger() -> logging.Logger:
        """
        Create default logger configuration.

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger("rag_monitor")
        logger.setLevel(logging.INFO)

        # Console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        return logger


class MetricsAggregator:
    """Aggregate and analyze RAG metrics."""

    def __init__(self):
        """Initialize metrics aggregator."""
        self.metrics_history: List[QueryMetrics] = []

    def add_metrics(self, metrics: QueryMetrics):
        """
        Add metrics to history.

        Args:
            metrics: Query metrics to add
        """
        self.metrics_history.append(metrics)

    def calculate_cache_hit_rate(self, time_window: int = 3600) -> float:
        """
        Calculate cache hit rate for time window.

        Args:
            time_window: Time window in seconds

        Returns:
            Cache hit rate (0-1)
        """
        cutoff_time = time.time() - time_window
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return 0.0

        cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
        return cache_hits / len(recent_metrics)

    def calculate_avg_latency(self, time_window: int = 3600) -> float:
        """
        Calculate average latency for time window.

        Args:
            time_window: Time window in seconds

        Returns:
            Average latency in milliseconds
        """
        cutoff_time = time.time() - time_window
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return 0.0

        total_latency = sum(m.latency_ms for m in recent_metrics)
        return total_latency / len(recent_metrics)

    def calculate_avg_user_rating(self, time_window: int = 3600) -> float:
        """
        Calculate average user rating for time window.

        Args:
            time_window: Time window in seconds

        Returns:
            Average user rating (1-5)
        """
        cutoff_time = time.time() - time_window
        recent_metrics = [
            m
            for m in self.metrics_history
            if m.timestamp >= cutoff_time and m.user_feedback is not None
        ]

        if not recent_metrics:
            return 0.0

        total_rating = sum(m.user_feedback for m in recent_metrics)
        return total_rating / len(recent_metrics)
