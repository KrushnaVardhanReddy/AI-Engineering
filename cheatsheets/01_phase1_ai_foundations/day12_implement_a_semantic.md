# Semantic Search Cheatsheet

## Day 12: Implement a Semantic Search Script

### 1. Concise Summary
Semantic search uses vector embeddings to retrieve documents based on their underlying meaning rather than exact keyword matches. This enables search that understands context, synonyms, and intent.

### 2. Code Snippets

```python
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_openai import OpenAIEmbeddings

# 1. Initialize Open AI Embeddings
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ.get("OPENAI_API_KEY", "sk-dummy")
)

# 2. Local Text Batch
documents = [
    {"id": 1, "text": "The quick brown fox jumps over the lazy dog."},
    {"id": 2, "text": "Python is a popular programming language for AI."},
    {"id": 3, "text": "Qdrant is an open-source vector database."}
]

# 3. Create Embeddings
vectors = embeddings_model.embed_documents([doc["text"] for doc in documents])

# 4. Initialize Qdrant Client (In-memory for testing)
client = QdrantClient(":memory:")

# 5. Create Collection
client.create_collection(
    collection_name="my_documents",
    vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
)

# 6. Upsert Documents
points = [
    PointStruct(id=doc["id"], vector=vectors[i], payload={"text": doc["text"]})
    for i, doc in enumerate(documents)
]
client.upsert(
    collection_name="my_documents",
    points=points
)

# 7. Perform Semantic Search
query = "What is a good language for machine learning?"
query_vector = embeddings_model.embed_query(query)

# Use query_points for search (v1.19.0+), accessing .points
try:
    search_result = client.query_points(
        collection_name="my_documents",
        query=query_vector,
        limit=1
    )

    # Check if results exist
    if search_result.points:
        top_match = search_result.points[0]
        print(f"Top Match: {top_match.payload['text']}")
        print(f"Score: {top_match.score}")
    else:
        print("No matches found.")
except Exception as e:
    print(f"Search failed: {e}")

```

### 3. Key Concepts
- **Embeddings:** Numerical representations (vectors) of text where similar meanings are located close to each other in vector space.
- **Cosine Distance/Similarity:** A common metric for measuring the angle between two vectors to determine how similar they are.
- **`query_points()`:** The updated method in `qdrant-client` (v1.19.0+) used to retrieve the nearest vectors to a given query vector.

### 4. Common Gotchas
- **Using deprecated methods:** Using `client.search()` instead of `client.query_points()` in newer versions of `qdrant-client` (1.19.0+) will cause issues. Always access results via `.points`.
- **Mismatched Vector Dimensions:** The dimension size configured in `VectorParams` must exactly match the output dimension size of your chosen embedding model (e.g., `text-embedding-3-small` is 1536).

### 5. Reference Links
- [Qdrant Python Client Documentation](https://qdrant.tech/documentation/interfaces/python/)
- [LangChain OpenAI Embeddings](https://python.langchain.com/docs/integrations/text_embedding/openai/)

### 6. Production-Ready OOP Implementation

```python
import os
import logging
import re
from typing import List, Dict, Any

from pydantic import BaseModel, Field, SecretStr
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Document(BaseModel):
    id: int
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SemanticSearcher:
    """
    A production-ready Semantic Searcher demonstrating clean OOP design and AI security.
    """
    def __init__(self, collection_name: str, embedding_model_name: str = "text-embedding-3-small"):
        self.collection_name = collection_name

        # Security: Fetch API key securely and avoid hardcoded fallbacks in production logic
        api_key_str = os.environ.get("OPENAI_API_KEY")
        if not api_key_str:
            raise ValueError("OPENAI_API_KEY environment variable is not set. It is required for production.")

        self._api_key = SecretStr(api_key_str)

        try:
            self.embeddings_model = OpenAIEmbeddings(
                model=embedding_model_name,
                api_key=self._api_key
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings model: {e}")
            raise

        # Initialize Qdrant Client (In-memory for testing, use URL/API key in production)
        self.client = QdrantClient(":memory:")
        self._initialize_collection()

    def _initialize_collection(self) -> None:
        """Initializes the Qdrant collection if it doesn't exist."""
        # For text-embedding-3-small, the dimension is 1536
        dimension = 1536
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            logger.info(f"Collection '{self.collection_name}' initialized.")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def _sanitize_pii(self, text: str) -> str:
        """Sanitizes PII from text before embedding. Redacts email addresses."""
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return re.sub(email_pattern, "[REDACTED_EMAIL]", text)

    def index_documents(self, documents: List[Document]) -> None:
        """Embeds and indexes a list of documents."""
        if not documents:
            logger.warning("No documents provided to index.")
            return

        # Security: Sanitize PII here before embedding
        texts = [self._sanitize_pii(doc.text) for doc in documents]

        try:
            vectors = self.embeddings_model.embed_documents(texts)
            points = [
                PointStruct(id=doc.id, vector=vectors[i], payload={"text": texts[i], **doc.metadata})
                for i, doc in enumerate(documents)
            ]
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Successfully indexed {len(documents)} documents.")
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")

    def search(self, query: str, limit: int = 1) -> List[Dict[str, Any]]:
        """Searches for the most relevant documents based on semantic similarity."""
        try:
            query_vector = self.embeddings_model.embed_query(query)
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit
            )

            results = []
            if search_result.points:
                for point in search_result.points:
                    results.append({
                        "score": point.score,
                        "text": point.payload.get("text", ""),
                        "metadata": {k: v for k, v in point.payload.items() if k != "text"}
                    })
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            # Graceful fallback: return empty list on failure
            return []

if __name__ == "__main__":
    # Example Usage
    searcher = SemanticSearcher(collection_name="knowledge_base")

    docs = [
        Document(id=1, text="The quick brown fox jumps over the lazy dog."),
        Document(id=2, text="Python is a popular programming language for AI."),
        Document(id=3, text="Qdrant is an open-source vector database."),
        Document(id=4, text="Contact us at secret@example.com for more info.") # Contains PII
    ]

    searcher.index_documents(docs)

    query_str = "What is a good language for machine learning?"
    print(f"\nQuery: {query_str}")

    results = searcher.search(query=query_str, limit=1)
    if results:
        for res in results:
            print(f"Top Match: {res['text']} (Score: {res['score']:.4f})")
    else:
        print("No matches found.")
```
