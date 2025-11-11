"""
Document ingestion pipeline for batch processing.

This module orchestrates the complete document ingestion workflow:
preprocessing, chunking, embedding, and indexing.
"""

from typing import List, Dict, Optional
from pathlib import Path
import logging

from ..core.chunking import SemanticChunker, SectionDetector
from ..core.indexing import DocumentIndexer, IndexConfig
from ..config import settings


logger = logging.getLogger(__name__)


class IngestPipeline:
    """
    Batch document ingestion workflow.

    SOLID Principles:
    - Single Responsibility: Orchestrates ingestion workflow
    - Open/Closed: Extensible via dependency injection
    - Dependency Inversion: Depends on abstractions

    Attributes:
        chunker: Document chunker instance
        indexer: Document indexer instance
        embedding_model: Embedding model for vector generation
    """

    def __init__(
        self,
        chunker: SemanticChunker,
        indexer: DocumentIndexer,
        embedding_model: object,
    ):
        """
        Initialize ingestion pipeline.

        Args:
            chunker: Document chunker
            indexer: Document indexer
            embedding_model: Embedding model
        """
        self.chunker = chunker
        self.indexer = indexer
        self.embedding_model = embedding_model

    def ingest_documents(
        self,
        documents: List[Dict],
        section_detector: Optional[SectionDetector] = None,
        batch_size: int = 100,
    ) -> Dict:
        """
        Ingest batch of documents.

        Args:
            documents: List of document dictionaries
            section_detector: Optional custom section detector
            batch_size: Batch size for indexing

        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Starting ingestion of {len(documents)} documents")

        stats = {
            "total_documents": len(documents),
            "total_chunks": 0,
            "successful": 0,
            "failed": 0,
            "errors": [],
        }

        all_chunks = []

        for doc in documents:
            try:
                # Chunk document
                chunks = self.chunker.chunk_document(
                    text=doc["text"],
                    metadata={
                        "source": doc.get("source", ""),
                        "page": doc.get("page", 0),
                        "type": doc.get("type", ""),
                        "created_at": doc.get("created_at", ""),
                    },
                    section_detector=section_detector,
                )

                # Generate embeddings
                chunk_texts = [chunk["text"] for chunk in chunks]
                embeddings = self.embedding_model.embed_batch(chunk_texts)

                # Add embeddings to chunks
                for chunk, embedding in zip(chunks, embeddings):
                    chunk["embedding"] = embedding

                all_chunks.extend(chunks)
                stats["successful"] += 1
                stats["total_chunks"] += len(chunks)

            except Exception as e:
                logger.error(
                    f"Error processing document {doc.get('source', 'unknown')}: {e}"
                )
                stats["failed"] += 1
                stats["errors"].append(
                    {"document": doc.get("source", "unknown"), "error": str(e)}
                )

        # Index all chunks
        if all_chunks:
            try:
                self.indexer.index_documents(all_chunks, batch_size=batch_size)
                logger.info(f"Successfully indexed {len(all_chunks)} chunks")
            except Exception as e:
                logger.error(f"Error indexing chunks: {e}")
                stats["errors"].append({"stage": "indexing", "error": str(e)})

        logger.info(f"Ingestion complete: {stats}")
        return stats

    def ingest_from_directory(
        self,
        directory: Path,
        file_pattern: str = "*.txt",
        section_detector: Optional[SectionDetector] = None,
    ) -> Dict:
        """
        Ingest all documents from directory.

        Args:
            directory: Directory containing documents
            file_pattern: Glob pattern for files
            section_detector: Optional section detector

        Returns:
            Ingestion statistics
        """
        documents = []

        for file_path in Path(directory).glob(file_pattern):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                documents.append(
                    {
                        "text": text,
                        "source": str(file_path),
                        "type": file_path.suffix[1:],  # Remove dot
                        "created_at": file_path.stat().st_ctime,
                    }
                )
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")

        return self.ingest_documents(documents, section_detector)


class DocumentPreprocessor:
    """Preprocess documents before chunking."""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = " ".join(text.split())

        # Normalize line breaks
        text = text.replace("\r\n", "\n")

        return text

    @staticmethod
    def extract_metadata(file_path: Path) -> Dict:
        """
        Extract metadata from file.

        Args:
            file_path: Path to file

        Returns:
            Metadata dictionary
        """
        stat = file_path.stat()

        return {
            "source": str(file_path),
            "filename": file_path.name,
            "extension": file_path.suffix[1:],
            "size_bytes": stat.st_size,
            "created_at": stat.st_ctime,
            "modified_at": stat.st_mtime,
        }
