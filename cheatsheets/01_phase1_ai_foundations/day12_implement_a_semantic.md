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
