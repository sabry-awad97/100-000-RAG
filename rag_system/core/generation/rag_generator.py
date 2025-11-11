"""
RAG generation module with intelligent context packing and citation.

This module handles LLM prompt orchestration, context management,
and citation extraction for production RAG systems.
"""

from typing import List, Dict, Optional, Protocol
from dataclasses import dataclass


class LLMClient(Protocol):
    """Protocol for LLM client implementations."""

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> str:
        """Generate response from messages."""
        ...


@dataclass
class GenerationConfig:
    """
    Configuration for RAG generation.

    Attributes:
        model: LLM model identifier
        max_context_tokens: Maximum tokens for context
        temperature: Generation temperature
        max_response_tokens: Maximum tokens for response
        citation_format: Format for citations
    """

    model: str = "gpt-4-turbo-preview"
    max_context_tokens: int = 6000
    temperature: float = 0.1
    max_response_tokens: int = 1000
    citation_format: str = "[Document {doc_num}, Page {page}]"


class RAGGenerator:
    """
    RAG generation with intelligent context packing and citation.

    SOLID Principles:
    - Single Responsibility: Handles only generation logic
    - Open/Closed: Extensible via GenerationConfig
    - Liskov Substitution: Can work with any LLMClient
    - Dependency Inversion: Depends on LLMClient abstraction

    Attributes:
        llm_client: LLM client implementation
        config: Generation configuration
    """

    def __init__(
        self, llm_client: LLMClient, config: Optional[GenerationConfig] = None
    ):
        """
        Initialize RAG generator.

        Args:
            llm_client: LLM client implementation
            config: Optional generation configuration
        """
        self.llm_client = llm_client
        self.config = config or GenerationConfig()
        self._encoder = None  # Lazy loading

    @property
    def encoder(self):
        """Lazy load token encoder."""
        if self._encoder is None:
            import tiktoken

            self._encoder = tiktoken.encoding_for_model(self.config.model)
        return self._encoder

    def generate(self, query: str, retrieved_docs: List[Dict]) -> Dict:
        """
        Generate answer from retrieved documents with citations.

        Args:
            query: User query
            retrieved_docs: Retrieved document chunks with scores

        Returns:
            Dictionary with answer, sources, and context used
        """
        # Pack context intelligently
        context = self._pack_context(retrieved_docs, self.config.max_context_tokens)

        # Build prompt
        prompt = self._build_prompt(query, context)

        # Generate with citations
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt},
        ]

        response = self.llm_client.generate(
            messages=messages,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_response_tokens,
        )

        return {
            "answer": response,
            "sources": self._extract_citations(response),
            "context_used": context,
            "num_context_docs": len(context),
        }

    def _pack_context(self, docs: List[Dict], max_tokens: int) -> List[Dict]:
        """
        Pack highest-scoring documents within token budget.

        Args:
            docs: Documents sorted by relevance score
            max_tokens: Maximum tokens for context

        Returns:
            Subset of documents fitting within token budget
        """
        sorted_docs = sorted(docs, key=lambda x: x.get("score", 0), reverse=True)
        packed = []
        token_count = 0

        for doc in sorted_docs:
            doc_tokens = len(self.encoder.encode(doc["text"]))

            if token_count + doc_tokens > max_tokens:
                break

            packed.append(doc)
            token_count += doc_tokens

        return packed

    def _build_prompt(self, query: str, context: List[Dict]) -> str:
        """
        Build prompt with structured context and citation instructions.

        Args:
            query: User query
            context: Packed context documents

        Returns:
            Formatted prompt string
        """
        context_text = "\n\n".join(
            [
                f"[Document {i + 1}] (Source: {doc.get('source', 'Unknown')}, "
                f"Page: {doc.get('page', 'N/A')})\n{doc['text']}"
                for i, doc in enumerate(context)
            ]
        )

        return f"""Context Documents:
{context_text}

Question: {query}

Provide a comprehensive answer based ONLY on the context above. 
Cite sources using [Document X, Page Y] format after each claim."""

    def _get_system_prompt(self) -> str:
        """
        Get system prompt with strict grounding rules.

        Returns:
            System prompt string
        """
        return """You are a document analysis assistant.

Rules:
1. Answer ONLY using information from provided context
2. Cite every claim with [Document X, Page Y]
3. If information isn't in context, say "The provided documents don't contain information about [topic]"
4. Never make assumptions or use external knowledge
5. Maintain professional, precise language"""

    def _extract_citations(self, text: str) -> List[str]:
        """
        Extract citation references from generated text.

        Args:
            text: Generated text with citations

        Returns:
            List of unique citations
        """
        import re

        citations = re.findall(r"\[Document \d+, Page [^\]]+\]", text)
        return list(set(citations))


class OpenAIClient:
    """OpenAI LLM client implementation."""

    def __init__(self, api_key: str):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            import openai

            openai.api_key = self.api_key
            self._client = openai
        return self._client

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> str:
        """
        Generate response using OpenAI API.

        Args:
            messages: List of message dictionaries
            model: Model name to use for generation
            temperature: Generation temperature
            max_tokens: Maximum response tokens

        Returns:
            Generated text
        """
        response = self.client.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content
