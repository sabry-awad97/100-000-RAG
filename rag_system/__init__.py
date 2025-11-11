"""
RAG System - Production-ready Retrieval-Augmented Generation

A scalable RAG system designed for 100,000+ documents with:
- Semantic chunking
- Hybrid retrieval (dense + sparse + reranking)
- Intelligent caching
- Comprehensive monitoring
- Production-ready architecture following SOLID principles
"""

__version__ = "0.1.0"
__author__ = "RAG System Team"

from .core import (
    SemanticChunker,
    HybridRetriever,
    DocumentIndexer,
    RAGGenerator,
    SemanticCache,
    RAGMonitor,
)

from .pipelines import IngestPipeline, RetrievalPipeline, GenerationPipeline

from .config import settings, get_settings

__all__ = [
    "SemanticChunker",
    "HybridRetriever",
    "DocumentIndexer",
    "RAGGenerator",
    "SemanticCache",
    "RAGMonitor",
    "IngestPipeline",
    "RetrievalPipeline",
    "GenerationPipeline",
    "settings",
    "get_settings",
]
