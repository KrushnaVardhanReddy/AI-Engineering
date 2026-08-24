# Day 15: Qdrant Local Spin-up & Collection Initialization

## Concise Summaries
Qdrant is a high-performance, Rust-based vector search engine. For local development and testing, you can instantly spin it up via Docker. Initializing a collection involves specifying a vector configuration (most notably, the size/dimensions of your embeddings and the distance metric like Cosine). Proper initialization and dependency management are critical for agentic systems relying on semantic retrieval.

## Docker Spin-up Snippet
To run Qdrant locally, execute the following Docker command. This will map the HTTP and gRPC ports and persist the data in a local volume.

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

## Code Snippets

```python
import os
import logging
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

# Configure logging for better observability in production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabaseManager:
    """Manages Qdrant vector database connections and collections."""

    def __init__(self, host: str = "localhost", port: int = 6333):
        self.host = host
        self.port = port
        self.client = self._initialize_client()

    def _initialize_client(self) -> QdrantClient:
        """Initializes the Qdrant client."""
        # Note: In production, consider utilizing API keys via os.environ.get("QDRANT_API_KEY")
        # if connecting to a remote instance, raising an error if missing.
        try:
            client = QdrantClient(host=self.host, port=self.port)
            logger.info(f"Successfully connected to Qdrant at {self.host}:{self.port}")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def initialize_collection(self, collection_name: str, vector_size: int, distance_metric: models.Distance = models.Distance.COSINE):
        """Creates a collection if it does not already exist."""
        try:
            # Check if collection exists
            self.client.get_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' already exists.")
        except UnexpectedResponse as e:
            if e.status_code == 404:
                # Collection doesn't exist, create it
                logger.info(f"Collection '{collection_name}' not found. Creating it...")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=distance_metric
                    )
                )
                logger.info(f"Collection '{collection_name}' created successfully with dimension {vector_size}.")
            else:
                logger.error(f"An unexpected error occurred while checking collection: {e}")
                raise

if __name__ == "__main__":
    # Example Usage
    COLLECTION_NAME = "agent_knowledge_base"
    # Example vector size for standard text-embedding-ada-002 model is 1536
    VECTOR_DIMENSION = 1536

    db_manager = VectorDatabaseManager()
    db_manager.initialize_collection(
        collection_name=COLLECTION_NAME,
        vector_size=VECTOR_DIMENSION
    )
```

## Key Concepts
- **QdrantClient**: The primary class used to interface with the Qdrant database.
- **Collection**: A logical container for storing vectors and their associated payloads (metadata), similar to a table in SQL.
- **VectorParams**: Configuration specifying the properties of vectors in a collection.
- **Dimensions (`size`)**: The number of float values in the vector. This MUST exactly match the output dimension of your chosen embedding model (e.g., 1536 for OpenAI's ada-002).
- **Distance Metric (`distance`)**: The mathematical function used to calculate similarity between vectors (Cosine, Dot, or Euclidean). Cosine is standard for most NLP tasks.
- **AI Security (PII)**: Do not store raw sensitive PII in vector database payloads. Implement redaction or anonymization strategies *before* embedding and ingestion.
- **Graceful Fallbacks**: Ensure your system can gracefully degrade or retry if the vector database is temporarily unreachable, to prevent agentic loops from failing catastrophically.

## Common Gotchas
- **Dimension Mismatch:** The most common error is trying to upsert a vector of size 768 (e.g., standard BERT) into a collection initialized for 1536 (e.g., OpenAI). The insertion will fail.
- **Docker Port Conflicts:** Ensure port 6333 (HTTP) and 6334 (gRPC) are free on your host machine before starting the Docker container.

## Reference Links
- [Qdrant Quickstart Documentation](https://qdrant.tech/documentation/quick-start/)
- [Qdrant Python Client](https://github.com/qdrant/qdrant-client)
