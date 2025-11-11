"""
Semantic caching module using embedding similarity.

This module provides intelligent query caching that matches semantically
similar queries, dramatically improving cache hit rates.
"""

from typing import Optional, List, Dict
import time
import json
import numpy as np


class SemanticCache:
    """
    Semantic query cache using embedding similarity.

    SOLID Principles:
    - Single Responsibility: Manages semantic caching only
    - Open/Closed: Extensible for different similarity metrics
    - Dependency Inversion: Depends on Redis abstraction

    Attributes:
        redis_client: Redis client instance
        similarity_threshold: Minimum similarity for cache hit (0-1)
        ttl: Default time-to-live in seconds
    """

    def __init__(
        self, redis_client, similarity_threshold: float = 0.95, default_ttl: int = 3600
    ):
        """
        Initialize semantic cache.

        Args:
            redis_client: Redis client instance
            similarity_threshold: Minimum similarity for cache hit (0-1)
            default_ttl: Default TTL in seconds
        """
        self.redis = redis_client
        self.threshold = similarity_threshold
        self.default_ttl = default_ttl

    def get(self, query: str, query_embedding: np.ndarray) -> Optional[Dict]:
        """
        Retrieve cached results for semantically similar query.

        Uses Redis SCAN to avoid blocking on large keyspaces.

        Args:
            query: Query string
            query_embedding: Query embedding vector

        Returns:
            Cached results if similar query found, None otherwise
        """
        best_match = None
        highest_similarity = 0.0

        # Use SCAN instead of KEYS to avoid blocking Redis
        # scan_iter handles the cursor management automatically
        for key in self.redis.scan_iter(match="cache:query:*", count=100):
            try:
                cached_data = self.redis.hgetall(key)

                if not cached_data or b"embedding" not in cached_data:
                    continue

                cached_embedding = np.frombuffer(
                    cached_data[b"embedding"], dtype=np.float32
                )

                # Compute cosine similarity
                similarity = self._cosine_similarity(query_embedding, cached_embedding)

                # Track best match above threshold
                if similarity >= self.threshold and similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = {
                        "results": json.loads(cached_data[b"results"].decode()),
                        "cache_hit": True,
                        "similarity": float(similarity),
                        "cached_query": cached_data[b"query"].decode(),
                    }
            except Exception:
                # Skip corrupted or expired keys
                continue

        return best_match

    def set(
        self,
        query: str,
        query_embedding: np.ndarray,
        results: List[Dict],
        ttl: Optional[int] = None,
    ):
        """
        Cache query results with embedding.

        Args:
            query: Query string
            query_embedding: Query embedding vector (numpy array or list)
            results: Retrieved results to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        if ttl is None:
            ttl = self.default_ttl

        # Normalize embedding to numpy array to handle both lists and arrays
        embedding = np.asarray(query_embedding, dtype=np.float32)

        cache_key = f"cache:query:{hash(query)}"

        self.redis.hset(
            cache_key,
            mapping={
                "query": query,
                "embedding": embedding.tobytes(),
                "results": json.dumps(results),
                "timestamp": time.time(),
            },
        )

        self.redis.expire(cache_key, ttl)

    def invalidate(self, pattern: str = "cache:query:*"):
        """
        Invalidate cache entries matching pattern.

        Uses Redis SCAN to avoid blocking on large keyspaces.

        Args:
            pattern: Redis key pattern to match
        """
        # Use SCAN to collect keys in batches
        keys_to_delete = []
        batch_size = 1000  # Delete in batches to avoid large commands

        for key in self.redis.scan_iter(match=pattern, count=100):
            keys_to_delete.append(key)

            # Delete in batches
            if len(keys_to_delete) >= batch_size:
                self.redis.delete(*keys_to_delete)
                keys_to_delete = []

        # Delete remaining keys
        if keys_to_delete:
            self.redis.delete(*keys_to_delete)

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Uses Redis SCAN to avoid blocking on large keyspaces.

        Returns:
            Dictionary with cache statistics
        """
        # Count keys using SCAN
        entry_count = 0
        for _ in self.redis.scan_iter(match="cache:query:*", count=100):
            entry_count += 1

        return {
            "total_entries": entry_count,
            "memory_usage": self.redis.info("memory").get("used_memory_human", "N/A"),
        }

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


class CacheConfig:
    """Configuration for semantic cache."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        similarity_threshold: float = 0.95,
        default_ttl: int = 3600,
    ):
        """
        Initialize cache configuration.

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            similarity_threshold: Minimum similarity for cache hit
            default_ttl: Default TTL in seconds
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl

    def create_cache(self) -> SemanticCache:
        """
        Create semantic cache instance.

        Returns:
            Configured SemanticCache instance
        """
        import redis

        redis_client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            decode_responses=False,  # Keep binary for embeddings
        )

        return SemanticCache(
            redis_client=redis_client,
            similarity_threshold=self.similarity_threshold,
            default_ttl=self.default_ttl,
        )
