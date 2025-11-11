# Docker Setup for RAG System

This guide explains how to set up Qdrant and Redis using Docker for the RAG system.

## Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed (included with Docker Desktop)

## Quick Start

### 1. Start Services

```bash
# Start Qdrant and Redis
docker-compose up -d

# Check if services are running
docker-compose ps
```

### 2. Verify Services

**Qdrant:**

```bash
# Check Qdrant health
curl http://localhost:6333/healthz

# Access Qdrant Web UI
# Open browser: http://localhost:6333/dashboard
```

**Redis:**

```bash
# Test Redis connection
docker exec -it rag_redis redis-cli ping
# Expected output: PONG
```

### 3. Stop Services

```bash
# Stop services
docker-compose down

# Stop and remove volumes (WARNING: deletes all data)
docker-compose down -v
```

## Service Details

### Qdrant Vector Database

- **REST API**: `http://localhost:6333`
- **gRPC API**: `http://localhost:6334`
- **Web Dashboard**: `http://localhost:6333/dashboard`
- **Storage**: Persistent volume `qdrant_storage`

### Redis Cache

- **Host**: `localhost`
- **Port**: `6379`
- **Storage**: Persistent volume `redis_data` with AOF enabled

### Local Embedding Service (Docker AI - Gemma)

- **REST API**: `http://localhost:8000`
- **Model**: `embeddinggemma:latest` (768-dimensional embeddings)
- **Health Check**: `http://localhost:8000/health`
- **Embedding Endpoint**: `POST http://localhost:8000/embed`

This service uses the Gemma embedding model from Docker AI to generate embeddings locally, eliminating the need for external API calls.

## Configuration

Update your `.env` file to use Docker services:

```bash
# Qdrant Configuration
QDRANT_URL=http://localhost:6333

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Local Embedding Service (Docker AI - Gemma)
LOCAL_EMBEDDING_URL=http://localhost:8000
EMBEDDING_MODEL=gemma
```

### Using Local Embeddings

The system uses the Gemma embedding model from Docker AI:

```python
from rag_system.core.retrieval import LocalEmbedder

# Initialize local embedder
embedder = LocalEmbedder(service_url="http://localhost:8000")

# Generate embeddings
embedding = embedder.embed_text("Your text here")
embeddings = embedder.embed_texts(["Text 1", "Text 2"])
```

**Benefits of Gemma Embeddings:**

- No API costs for embeddings
- Faster response times (no network latency)
- Data privacy (embeddings generated locally)
- No rate limits
- High-quality 768-dimensional embeddings

**Model Specifications:**

- **Model**: `embeddinggemma:latest`
- **Dimensions**: 768
- **Provider**: Docker AI
- **License**: Open source

## Advanced Usage

### View Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f qdrant
docker-compose logs -f redis
```

### Backup Data

```bash
# Backup Qdrant data
docker run --rm -v qdrant_storage:/data -v $(pwd):/backup alpine tar czf /backup/qdrant_backup.tar.gz -C /data .

# Backup Redis data
docker run --rm -v redis_data:/data -v $(pwd):/backup alpine tar czf /backup/redis_backup.tar.gz -C /data .
```

### Restore Data

```bash
# Restore Qdrant data
docker run --rm -v qdrant_storage:/data -v $(pwd):/backup alpine sh -c "cd /data && tar xzf /backup/qdrant_backup.tar.gz"

# Restore Redis data
docker run --rm -v redis_data:/data -v $(pwd):/backup alpine sh -c "cd /data && tar xzf /backup/redis_backup.tar.gz"
```

### Production Configuration

For production, consider:

1. **Resource Limits**:

```yaml
services:
  qdrant:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

2. **Security**:

```yaml
services:
  redis:
    command: redis-server --requirepass your_secure_password
```

3. **Network Isolation**:

```yaml
networks:
  rag_network:
    driver: bridge
    internal: true  # Prevent external access
```

## Troubleshooting

### Port Already in Use

```bash
# Check what's using the port
netstat -ano | findstr :6333  # Windows
lsof -i :6333                  # Linux/Mac

# Change port in docker-compose.yml
ports:
  - "6335:6333"  # Use different host port
```

### Permission Issues

```bash
# Fix volume permissions (Linux)
sudo chown -R 1000:1000 /var/lib/docker/volumes/
```

### Reset Everything

```bash
# Stop all containers and remove volumes
docker-compose down -v

# Remove all unused Docker resources
docker system prune -a --volumes
```

## Monitoring

### Qdrant Metrics

Access Qdrant metrics at: `http://localhost:6333/metrics`

### Redis Monitoring

```bash
# Connect to Redis CLI
docker exec -it rag_redis redis-cli

# Check memory usage
INFO memory

# Monitor commands in real-time
MONITOR
```

## Alternative: Qdrant Cloud

For production, consider [Qdrant Cloud](https://cloud.qdrant.io/):

```bash
# Update .env for Qdrant Cloud
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
```

## Next Steps

1. Start services: `docker-compose up -d`
2. Run migrations: `python -m rag_system.scripts.migrate_qdrant create rag_documents`
3. Test connection: `python -m rag_system.scripts.migrate_qdrant info rag_documents`
4. Start ingesting documents with your RAG pipeline
