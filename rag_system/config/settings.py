"""
Configuration settings for RAG system.

This module centralizes all configuration using environment variables
and provides type-safe configuration access.
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class QdrantConfig:
    """Qdrant vector database configuration."""

    url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name: str = os.getenv("QDRANT_COLLECTION", "rag_documents")
    vector_size: int = int(os.getenv("QDRANT_VECTOR_SIZE", "1536"))
    on_disk: bool = os.getenv("QDRANT_ON_DISK", "true").lower() == "true"


@dataclass
class RedisConfig:
    """Redis cache configuration."""

    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    db: int = int(os.getenv("REDIS_DB", "0"))
    password: Optional[str] = os.getenv("REDIS_PASSWORD")


@dataclass
class OpenAIConfig:
    """OpenAI API configuration."""

    api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4-turbo-preview")
    max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
    temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))


@dataclass
class ChunkingConfig:
    """Document chunking configuration."""

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "300"))
    overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    encoding_name: str = os.getenv("CHUNK_ENCODING", "cl100k_base")


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""

    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))
    fusion_k: int = int(os.getenv("FUSION_K", "60"))
    rerank_model: str = os.getenv(
        "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    ce_weight: float = float(os.getenv("CE_WEIGHT", "0.7"))
    fusion_weight: float = float(os.getenv("FUSION_WEIGHT", "0.3"))


@dataclass
class CacheConfig:
    """Semantic cache configuration."""

    enabled: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    similarity_threshold: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95"))
    ttl: int = int(os.getenv("CACHE_TTL", "3600"))


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""

    enabled: bool = os.getenv("MONITORING_ENABLED", "true").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: Optional[str] = os.getenv("LOG_FILE")


class Settings:
    """
    Centralized settings for RAG system.

    SOLID Principles:
    - Single Responsibility: Manages only configuration
    - Open/Closed: Extensible via new config dataclasses
    - Dependency Inversion: Provides abstractions for configuration
    """

    def __init__(self):
        """Initialize settings from environment variables."""
        self.qdrant = QdrantConfig()
        self.redis = RedisConfig()
        self.openai = OpenAIConfig()
        self.chunking = ChunkingConfig()
        self.retrieval = RetrievalConfig()
        self.cache = CacheConfig()
        self.monitoring = MonitoringConfig()

        # Validate critical settings
        self._validate()

    def _validate(self):
        """Validate critical configuration."""
        if not self.openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")

        if self.chunking.chunk_size <= self.chunking.overlap:
            raise ValueError("Chunk size must be greater than overlap")

        if not (0 <= self.cache.similarity_threshold <= 1):
            raise ValueError("Cache similarity threshold must be between 0 and 1")

    @classmethod
    def from_env_file(cls, env_file: Path):
        """
        Load settings from .env file.

        Args:
            env_file: Path to .env file

        Returns:
            Settings instance
        """
        from dotenv import load_dotenv

        load_dotenv(env_file)
        return cls()

    def to_dict(self) -> dict:
        """
        Convert settings to dictionary.

        Returns:
            Dictionary representation of settings
        """
        from dataclasses import asdict

        return {
            "qdrant": asdict(self.qdrant),
            "redis": asdict(self.redis),
            "openai": {
                **asdict(self.openai),
                "api_key": "***",  # Mask API key
            },
            "chunking": asdict(self.chunking),
            "retrieval": asdict(self.retrieval),
            "cache": asdict(self.cache),
            "monitoring": asdict(self.monitoring),
        }


# Global settings instance
settings = Settings()
