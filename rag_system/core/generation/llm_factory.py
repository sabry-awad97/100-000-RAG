"""
LLM client factory for creating appropriate LLM clients.

This module provides a factory pattern for creating LLM clients
based on configuration, following the Dependency Inversion Principle.
"""

from .rag_generator import OpenAIClient, GeminiClient


class LLMClientFactory:
    """
    Factory for creating LLM clients.

    SOLID Principles:
    - Open/Closed: Extensible for new providers without modification
    - Dependency Inversion: Returns protocol-compliant clients
    - Single Responsibility: Only responsible for client creation
    """

    @staticmethod
    def create_client(provider: str, api_key: str):
        """
        Create an LLM client based on provider.

        Args:
            provider: LLM provider name ("openai" or "gemini")
            api_key: API key for the provider

        Returns:
            LLM client implementing the LLMClient protocol

        Raises:
            ValueError: If provider is not supported
        """
        provider = provider.lower()

        if provider == "openai":
            return OpenAIClient(api_key=api_key)
        elif provider == "gemini":
            return GeminiClient(api_key=api_key)
        else:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers: openai, gemini"
            )

    @staticmethod
    def create_from_settings(settings):
        """
        Create an LLM client from settings configuration.

        Args:
            settings: Settings object with LLM configuration

        Returns:
            Configured LLM client

        Raises:
            ValueError: If provider is not configured or unsupported
        """
        provider = settings.llm.provider.lower()

        if provider == "openai":
            return OpenAIClient(api_key=settings.openai.api_key)
        elif provider == "gemini":
            return GeminiClient(api_key=settings.gemini.api_key)
        else:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Please set LLM_PROVIDER to 'openai' or 'gemini' in your .env file."
            )
