"""
Document indexing module for Qdrant vector database.

This module provides production-ready document indexing with optimizations
for large-scale deployments (100K+ documents).
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import uuid


@dataclass
class IndexConfig:
    """
    Configuration for document indexing.

    Attributes:
        collection_name: Name of the collection
        vector_size: Embedding dimension
        distance_metric: Distance metric (COSINE, EUCLIDEAN, DOT)
        on_disk: Store vectors on disk to reduce RAM
        quantization_type: Quantization type (int8, uint8, binary)
        indexing_threshold: Optimize after N documents
    """

    collection_name: str
    vector_size: int = 1536
    distance_metric: str = "COSINE"
    on_disk: bool = True
    quantization_type: str = "int8"
    indexing_threshold: int = 20000


class DocumentIndexer:
    """
    Production-ready document indexing for Qdrant.

    SOLID Principles:
    - Single Responsibility: Handles only indexing operations
    - Open/Closed: Extensible via IndexConfig
    - Dependency Inversion: Depends on QdrantClient abstraction

    Attributes:
        client: Qdrant client instance
        config: Indexing configuration
    """

    def __init__(self, qdrant_url: str, config: Optional[IndexConfig] = None):
        """
        Initialize document indexer.

        Args:
            qdrant_url: Qdrant server URL
            config: Optional indexing configuration
        """
        self.qdrant_url = qdrant_url
        self.config = config
        self._client = None  # Lazy loading

    @property
    def client(self):
        """Lazy load Qdrant client."""
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self.qdrant_url)
        return self._client

    def create_collection(self, config: IndexConfig):
        """
        Create optimized collection for large-scale document storage.

        Args:
            config: Collection configuration
        """
        from qdrant_client.models import Distance, VectorParams

        # Map distance metric string to enum
        distance_map = {
            "COSINE": Distance.COSINE,
            "EUCLIDEAN": Distance.EUCLID,
            "DOT": Distance.DOT,
        }

        self.client.create_collection(
            collection_name=config.collection_name,
            vectors_config=VectorParams(
                size=config.vector_size,
                distance=distance_map[config.distance_metric],
                on_disk=config.on_disk,  # Critical for 100K+ docs
            ),
            optimizers_config={
                "indexing_threshold": config.indexing_threshold,
            },
            quantization_config={
                "scalar": {
                    "type": config.quantization_type,
                    "quantile": 0.99,
                    "always_ram": True,  # Keep quantized vectors in RAM
                }
            },
        )

        self.config = config

    def index_documents(self, documents: List[Dict], batch_size: int = 100):
        """
        Batch index documents for optimal performance.

        Args:
            documents: List of document dictionaries with embeddings
            batch_size: Number of documents per batch insert

        Raises:
            ValueError: If config is not set
        """
        if not self.config:
            raise ValueError("Config must be set before indexing")

        from qdrant_client.models import PointStruct

        points = []

        for doc in documents:
            point = PointStruct(
                id=doc.get("id", str(uuid.uuid4())),
                vector=doc["embedding"],
                payload={
                    "text": doc["text"],
                    "source": doc.get("source", ""),
                    "page": doc.get("page", 0),
                    "doc_type": doc.get("type", ""),
                    "timestamp": doc.get("created_at", ""),
                },
            )
            points.append(point)

            # Batch insert for performance
            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=self.config.collection_name, points=points
                )
                points = []

        # Insert remaining documents
        if points:
            self.client.upsert(
                collection_name=self.config.collection_name, points=points
            )

    def delete_collection(self, collection_name: str):
        """
        Delete a collection.

        Args:
            collection_name: Name of collection to delete
        """
        self.client.delete_collection(collection_name=collection_name)

    def get_collection_info(self, collection_name: str) -> Dict:
        """
        Get collection information.

        Args:
            collection_name: Name of collection

        Returns:
            Collection information dictionary
        """
        return self.client.get_collection(collection_name=collection_name)
