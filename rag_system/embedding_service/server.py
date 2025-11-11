"""
Local embedding service using Docker AI models.

This service provides a REST API for generating embeddings using
the Gemma embedding model from Docker AI.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os


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

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        # Docker AI models are accessible via the models service
        # This is a placeholder - adjust based on actual Docker AI API
        embeddings = []

        for text in texts:
            # In production, this would call the Docker AI model API
            # For now, return placeholder with correct dimensions
            embedding = [0.0] * 768  # Gemma produces 768-dim embeddings
            embeddings.append(embedding)

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
