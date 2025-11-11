"""
Semantic document chunking module.

This module provides context-aware text segmentation that respects document structure
and semantic boundaries, implementing the Single Responsibility Principle.
"""

from typing import List, Dict, Protocol
import tiktoken


class SectionDetector(Protocol):
    """Protocol for document section detection strategies."""

    def detect_sections(self, text: str) -> List[str]:
        """Detect semantic sections in document."""
        ...


class SemanticChunker:
    """
    Adaptive document chunking that respects semantic boundaries.

    SOLID Principles Applied:
    - Single Responsibility: Handles only document chunking logic
    - Open/Closed: Extensible for new document types via section detectors
    - Liskov Substitution: Any SectionDetector implementation can be used
    - Interface Segregation: Clean separation via Protocol
    - Dependency Inversion: Depends on SectionDetector abstraction

    Attributes:
        chunk_size: Target chunk size in tokens
        overlap: Token overlap between chunks for context preservation
        encoder: Token encoder for text tokenization
    """

    def __init__(
        self,
        chunk_size: int = 300,
        overlap: int = 50,
        encoding_name: str = "cl100k_base",
    ):
        """
        Initialize the semantic chunker.

        Args:
            chunk_size: Target chunk size in tokens
            overlap: Token overlap between chunks
            encoding_name: Tiktoken encoding name
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoder = tiktoken.get_encoding(encoding_name)

    def chunk_document(
        self, text: str, metadata: Dict, section_detector: SectionDetector = None
    ) -> List[Dict]:
        """
        Chunk document while preserving semantic structure.

        Args:
            text: Raw document text
            metadata: Document metadata (source, type, etc.)
            section_detector: Optional custom section detector

        Returns:
            List of chunk dictionaries with text and enriched metadata
        """
        # Detect document structure
        if section_detector:
            sections = section_detector.detect_sections(text)
        else:
            sections = self._default_section_detection(text)

        chunks = []

        for section in sections:
            # Respect semantic boundaries
            if self._is_atomic_section(section):
                chunks.append(self._create_chunk(section, metadata))
            else:
                # Split large sections with overlap
                sub_chunks = self._split_with_overlap(
                    section, self.chunk_size, self.overlap
                )
                chunks.extend(
                    [self._create_chunk(chunk, metadata) for chunk in sub_chunks]
                )

        return chunks

    def _split_with_overlap(self, text: str, size: int, overlap: int) -> List[str]:
        """
        Split text into overlapping chunks at token level.

        Args:
            text: Text to split
            size: Chunk size in tokens
            overlap: Overlap size in tokens

        Returns:
            List of text chunks with specified overlap
        """
        tokens = self.encoder.encode(text)
        chunks = []

        for i in range(0, len(tokens), size - overlap):
            chunk_tokens = tokens[i : i + size]
            chunks.append(self.encoder.decode(chunk_tokens))

        return chunks

    def _create_chunk(self, text: str, metadata: Dict) -> Dict:
        """
        Create chunk dictionary with enriched metadata.

        Args:
            text: Chunk text
            metadata: Base metadata to extend

        Returns:
            Chunk dictionary with text and metadata
        """
        return {
            "text": text,
            "metadata": {
                **metadata,
                "chunk_size": len(self.encoder.encode(text)),
                "preview": text[:100] + "..." if len(text) > 100 else text,
            },
        }

    def _is_atomic_section(self, section: str) -> bool:
        """
        Determine if section should remain intact.

        Args:
            section: Text section to evaluate

        Returns:
            True if section should not be split
        """
        token_count = len(self.encoder.encode(section))
        return token_count <= self.chunk_size

    def _default_section_detection(self, text: str) -> List[str]:
        """
        Default section detection using paragraph boundaries.

        Args:
            text: Document text

        Returns:
            List of detected sections
        """
        # Simple paragraph-based splitting
        # Override with custom section detector for domain-specific logic
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]


class LegalDocumentDetector:
    """Section detector specialized for legal documents."""

    def detect_sections(self, text: str) -> List[str]:
        """
        Detect sections in legal documents based on common patterns.

        Args:
            text: Legal document text

        Returns:
            List of detected sections
        """
        # Implement legal-specific section detection
        # E.g., detect clauses, articles, sections by numbering patterns
        import re

        # Pattern for legal section markers (e.g., "Section 1.", "Article I", etc.)
        section_pattern = r"(?:Section|Article|Clause)\s+[\dIVXivx]+[.:]"

        sections = []
        current_section = []

        for line in text.split("\n"):
            if re.match(section_pattern, line.strip()):
                if current_section:
                    sections.append("\n".join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        if current_section:
            sections.append("\n".join(current_section))

        return sections if sections else [text]


class TechnicalManualDetector:
    """Section detector specialized for technical manuals."""

    def detect_sections(self, text: str) -> List[str]:
        """
        Detect sections in technical manuals based on heading patterns.

        Args:
            text: Technical manual text

        Returns:
            List of detected sections
        """
        # Implement technical manual section detection
        # E.g., detect by markdown headings, numbered procedures, etc.
        import re

        # Pattern for headings (e.g., "1.2.3 Title", "### Heading")
        heading_pattern = r"^(?:#{1,6}\s+|\d+(?:\.\d+)*\s+)[A-Z]"

        sections = []
        current_section = []

        for line in text.split("\n"):
            if re.match(heading_pattern, line.strip()):
                if current_section:
                    sections.append("\n".join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        if current_section:
            sections.append("\n".join(current_section))

        return sections if sections else [text]
