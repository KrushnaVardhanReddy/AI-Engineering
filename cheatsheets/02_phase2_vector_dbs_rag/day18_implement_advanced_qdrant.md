# Day 18: Advanced Qdrant Operations (Filtered Searches & Scroll APIs)

## Concise Summaries
Advanced vector database operations are critical for managing scale and improving search precision. **Filtered searches** allow you to apply traditional metadata filtering alongside vector similarity, dramatically reducing search space and improving relevance. **Scroll APIs** enable pagination over entire datasets without the limitations of standard search limits, essential for data extraction, migrations, and bulk updates.

## Code Snippets

```python
import os
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

class AdvancedQdrantManager:
    """
    Manages advanced Qdrant operations such as filtered searches and scrolling.
    Demonstrates clean OOP principles and secure client initialization.
    """

    def __init__(self, collection_name: str, host: str = "localhost", port: int = 6333):
        self.collection_name = collection_name
        self._initialize_client(host, port)

    def _initialize_client(self, host: str, port: int) -> None:
        """
        Securely initializes the Qdrant client. Avoids hardcoding secrets.
        Falls back to in-memory mode for local testing if API key is not present.
        """
        api_key = os.environ.get("QDRANT_API_KEY")
        if api_key:
            self.client = QdrantClient(url=f"https://{host}:{port}", api_key=api_key)
        else:
            # Fallback to local memory for safe execution in missing-key scenarios
            self.client = QdrantClient(location=":memory:")
            self._setup_dummy_collection()

    def _setup_dummy_collection(self) -> None:
        """Sets up a temporary collection for demonstration purposes."""
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
        )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=1,
                    vector=[0.1, 0.2, 0.3, 0.4],
                    payload={"category": "A", "status": "active", "user_id": "u_123"}
                ),
                models.PointStruct(
                    id=2,
                    vector=[0.5, 0.6, 0.7, 0.8],
                    payload={"category": "B", "status": "inactive", "user_id": "u_456"}
                ),
                models.PointStruct(
                    id=3,
                    vector=[0.9, 0.1, 0.2, 0.3],
                    payload={"category": "A", "status": "active", "user_id": "u_789"}
                ),
            ]
        )

    def search_with_filters(self, query_vector: List[float], category: str, limit: int = 5) -> Any:
        """
        Performs a vector search filtered by metadata constraints.
        Applies a 'must' condition on both category and active status.
        """
        try:
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="category",
                        match=models.MatchValue(value=category)
                    ),
                    models.FieldCondition(
                        key="status",
                        match=models.MatchValue(value="active")
                    )
                ]
            )

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=filter_condition,
                limit=limit
            )
            return results
        except Exception as e:
            # Fallback mechanism to handle search errors gracefully
            print(f"Error performing filtered search: {e}")
            return []

    def scroll_dataset(self, batch_size: int = 100) -> List[models.Record]:
        """
        Retrieves all points from a collection using the scroll API.
        Optimized by excluding heavy vectors if only metadata is needed.
        """
        all_records: List[models.Record] = []
        next_page_offset: Optional[Any] = None

        try:
            while True:
                records, next_page_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=next_page_offset,
                    with_payload=True,
                    with_vectors=False  # Secure/Efficient: skip vectors to save bandwidth
                )

                all_records.extend(records)

                if next_page_offset is None:
                    break

            return all_records
        except Exception as e:
            print(f"Error scrolling dataset: {e}")
            return all_records

# --- Execution Example ---
if __name__ == "__main__":
    manager = AdvancedQdrantManager(collection_name="secure_documents")

    # 1. Filtered Search
    print("--- Filtered Search Results ---")
    search_results = manager.search_with_filters(
        query_vector=[0.1, 0.2, 0.3, 0.4],
        category="A"
    )

    points = getattr(search_results, "points", search_results)
    for result in points:
        print(f"ID: {result.id}, Score: {result.score:.4f}, Payload: {result.payload}")

    # 2. Scrolling Dataset
    print("\n--- Scrolled Records ---")
    scrolled_records = manager.scroll_dataset(batch_size=2)
    for record in scrolled_records:
        print(f"ID: {record.id}, Payload: {record.payload}")
```

## Key Concepts
- **`models.Filter`:** The core mechanism for defining search conditions in Qdrant.
- **`models.FieldCondition`:** Specifies the matching rule for a specific payload field (e.g., `MatchValue`, `MatchAny`).
- **`must`, `should`, `must_not`:** Logical operators within a Filter used to combine multiple conditions.
- **`scroll()` API:** A method designed for iterating over an entire collection systematically, circumventing maximum `limit` restrictions found in standard search APIs.
- **`offset`:** The cursor marker returned by `scroll()` indicating where the next batch of results should begin.
- **Security/PII Best Practice:** When scrolling for metadata extraction or audits, set `with_vectors=False` to minimize exposure and transfer overhead. Ensure sensitive metadata fields (like `user_id`) are adequately protected or masked before broad dissemination.

## Common Gotchas
1. **Ignoring Payload Indexes:** Applying filters on unindexed payload fields will lead to a full scan of the dataset, causing severe performance degradation at scale. Always index fields frequently used in `models.FieldCondition`.
2. **Scrolling Without Offsets:** Attempting to retrieve a massive dataset using `client.search` (or `query_points`) with an extremely high limit will crash or timeout. You must use the `scroll()` API with proper offset iteration for bulk retrieval.

## Reference Links
- [Qdrant Documentation: Filtering](https://qdrant.tech/documentation/concepts/filtering/)
- [Qdrant Documentation: Payload indexing](https://qdrant.tech/documentation/concepts/indexing/#payload-index)
- [Qdrant Documentation: Scroll API](https://qdrant.tech/documentation/concepts/points/#scroll-points)
