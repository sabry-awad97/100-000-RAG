"""
Local embedding client for Docker AI models.

This module provides an interface to the local embedding service
running in Docker using Gemma embeddings.
"""

import requests
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """
    Client for local Docker AI embedding service.

    This provides an alternative to OpenAI embeddings by using
    a local model running in Docker.
    """

    def __init__(self, service_url: str = "http://localhost:8000"):
        """
        Initialize local embedder.

        Args:
            service_url: URL of the embedding service
        """
        self.service_url = service_url
        self._verify_service()

    def _verify_service(self):
        """Verify that the embedding service is accessible."""
        try:
            response = requests.get(f"{self.service_url}/health", timeout=5)
            response.raise_for_status()
            logger.info("Local embedding service is healthy")
        except Exception as e:
            logger.warning(f"Local embedding service not accessible: {e}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        embeddings = self.embed_texts([text])
        return embeddings[0]

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            response = requests.post(
                f"{self.service_url}/embed", json={"texts": texts}, timeout=30
            )
            response.raise_for_status()

            data = response.json()
            embeddings = [np.array(emb, dtype=np.float32) for emb in data["embeddings"]]

            logger.info(f"Generated {len(embeddings)} embeddings using {data['model']}")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.

        Returns:
            Embedding dimension
        """
        # Gemma produces 768-dimensional embeddings
        return 768
