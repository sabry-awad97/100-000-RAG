"""
Configuration management for RAG system.

Uses Pydantic for validation and environment variables for configuration.
"""

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from rag_system directory
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


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
    password: str | None = os.getenv("REDIS_PASSWORD")


@dataclass
class LLMConfig:
    """LLM provider configuration."""

    provider: str = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "gemini"


@dataclass
class OpenAIConfig:
    """OpenAI API configuration."""

    api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4-turbo-preview")
    max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
    temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))


@dataclass
class GeminiConfig:
    """Google Gemini API configuration."""

    api_key: str = os.getenv("GEMINI_API_KEY", "")
    chat_model: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    max_tokens: int = int(os.getenv("GEMINI_MAX_TOKENS", "1000"))
    temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))


@dataclass
class EmbeddingConfig:
    """Local embedding service configuration."""

    service_url: str = os.getenv(
        "LOCAL_EMBEDDING_URL", "http://localhost:11434/v1/embeddings"
    )
    model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


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
class CacheSettings:
    """Semantic cache settings configuration."""

    enabled: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    similarity_threshold: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95"))
    ttl: int = int(os.getenv("CACHE_TTL", "3600"))


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""

    enabled: bool = os.getenv("MONITORING_ENABLED", "true").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str | None = os.getenv("LOG_FILE")


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
        self.llm = LLMConfig()
        self.qdrant = QdrantConfig()
        self.redis = RedisConfig()
        self.openai = OpenAIConfig()
        self.gemini = GeminiConfig()
        self.embedding = EmbeddingConfig()
        self.chunking = ChunkingConfig()
        self.retrieval = RetrievalConfig()
        self.cache = CacheSettings()
        self.monitoring = MonitoringConfig()

        # Validate critical settings
        self._validate()

    def _validate(self):
        """Validate critical configuration."""
        # Only validate the active LLM provider
        provider = self.llm.provider.lower()

        if provider == "gemini":
            if not self.gemini.api_key:
                raise ValueError(
                    "GEMINI_API_KEY environment variable must be set when using Gemini"
                )
        elif provider == "openai":
            if not self.openai.api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable must be set when using OpenAI"
                )
        else:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. Use 'openai' or 'gemini'"
            )

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
        return {
            "llm": asdict(self.llm),
            "qdrant": asdict(self.qdrant),
            "redis": asdict(self.redis),
            "openai": {
                **asdict(self.openai),
                "api_key": "***",  # Mask API key
            },
            "gemini": {
                **asdict(self.gemini),
                "api_key": "***",  # Mask API key
            },
            "embedding": asdict(self.embedding),
            "chunking": asdict(self.chunking),
            "retrieval": asdict(self.retrieval),
            "cache": asdict(self.cache),
            "monitoring": asdict(self.monitoring),
        }


# Lazy initialization of settings
_settings_instance: Settings | None = None


def get_settings(validate: bool = False) -> Settings:
    """
    Get the global settings instance (lazy initialization).

    This function creates the Settings instance on first call and caches it.
    This prevents validation errors at import time.

    Args:
        validate: If True, validates that required settings are present

    Returns:
        Settings instance

    Raises:
        ValueError: If validate=True and required settings are missing
    """
    global _settings_instance

    if _settings_instance is None:
        _settings_instance = Settings()

    if validate:
        # Validate required settings based on LLM provider
        provider = _settings_instance.llm.provider.lower()

        if provider == "openai" and not _settings_instance.openai.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required when using OpenAI. "
                "Please set it in your .env file or environment."
            )
        elif provider == "gemini" and not _settings_instance.gemini.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required when using Gemini. "
                "Please set it in your .env file or environment."
            )

    return _settings_instance


def reset_settings():
    """Reset the settings instance (useful for testing)."""
    global _settings_instance
    _settings_instance = None


# For backward compatibility, create a lazy property-like access
# This allows `from config import settings` to still work
class _SettingsProxy:
    """Proxy object that lazily initializes settings on attribute access."""

    def __getattr__(self, name):
        return getattr(get_settings(), name)

    def __repr__(self):
        return repr(get_settings())


settings = _SettingsProxy()
