"""
Local embedding service using Docker AI models.

This service provides a REST API for generating embeddings using
the Gemma embedding model from Docker AI.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
import requests


class EmbeddingHandler(BaseHTTPRequestHandler):
    """HTTP handler for embedding requests."""

    MODEL_NAME = os.getenv("MODEL_NAME", "gemma")

    def do_POST(self):
        """Handle POST requests for embeddings."""
        if self.path == "/embed":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode("utf-8"))
                texts = data.get("texts", [])

                if not texts:
                    self.send_error(400, "No texts provided")
                    return

                # Generate embeddings using Docker AI model
                embeddings = self.generate_embeddings(texts)

                response = {
                    "embeddings": embeddings,
                    "model": self.MODEL_NAME,
                    "dimension": len(embeddings[0]) if embeddings else 0,
                }

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))

            except Exception as e:
                self.send_error(500, f"Error generating embeddings: {str(e)}")

        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode("utf-8"))

        else:
            self.send_error(404, "Endpoint not found")

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint not found")

    def generate_embeddings(self, texts):
        """
        Generate embeddings using Gemma Docker AI model.

        Model Specifications:
        - Context length: Up to 2K tokens (longer texts are truncated)
        - Embedding dimension: 768 (normalized by default)
        - Language: Primarily English (performance varies on other languages)
        - Normalization: Embeddings are pre-normalized for cosine similarity

        Args:
            texts: List of text strings (will be truncated if > 2K tokens)

        Returns:
            List of 768-dimensional embedding vectors (normalized)

        Raises:
            Exception: If the Docker AI API request fails
        """
        # Docker AI embedding API endpoint
        api_url = os.getenv(
            "DOCKER_AI_EMBEDDING_URL",
            "http://localhost:12434/engines/llama.cpp/v1/embeddings",
        )

        # Model supports up to 2K tokens - approximately 1500 words
        # Truncate texts that are too long to prevent errors
        MAX_CHARS = 8000  # Conservative estimate (~2K tokens)

        embeddings = []

        # Process each text (batch processing could be added for efficiency)
        for text in texts:
            try:
                # Truncate text if too long (model supports up to 2K tokens)
                if len(text) > MAX_CHARS:
                    text = text[:MAX_CHARS]
                    print(
                        f"Warning: Text truncated to {MAX_CHARS} chars (2K token limit)"
                    )

                response = requests.post(
                    api_url,
                    headers={"Content-Type": "application/json"},
                    json={"model": "ai/embeddinggemma", "input": text},
                    timeout=30,
                )
                response.raise_for_status()

                # Extract embedding from response
                data = response.json()
                embedding = data.get("data", [{}])[0].get("embedding", [])

                if not embedding:
                    raise ValueError(f"No embedding returned for text: {text[:50]}...")

                # Verify embedding dimension (should be 768 for Gemma)
                if len(embedding) != 768:
                    print(f"Warning: Expected 768 dimensions, got {len(embedding)}")

                embeddings.append(embedding)

            except requests.exceptions.RequestException as e:
                # Log error and raise
                print(f"Error calling Docker AI API: {e}")
                raise Exception(f"Failed to generate embedding: {e}")

        return embeddings


def run_server(port=8000):
    """Run the embedding service."""
    server_address = ("", port)
    httpd = HTTPServer(server_address, EmbeddingHandler)
    print(f"Embedding service running on port {port}")
    print(f"Model: {EmbeddingHandler.MODEL_NAME}")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
