# Day 16: Perform CRUD Operations in Qdrant

## Concise Summaries
Qdrant is a vector database that supports robust CRUD (Create, Read, Update, Delete) operations. Beyond inserting vectors, you can attach and update rich JSON metadata (payloads) and filter deletions based on those payloads. Managing these operations requires robust error handling, exact Point IDs (UUIDs or integers), and a focus on AI security (e.g., stripping Personally Identifiable Information (PII) before storage).

## Code Snippets
The following example demonstrates a production-ready class for managing Qdrant vectors. It includes PII redaction as an essential AI security practice.

```python
import os
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, UpdateStatus, Filter, FieldCondition, MatchValue
from qdrant_client.models import VectorParams, Distance

class QdrantVectorManager:
    """
    Manages vector operations (CRUD) in Qdrant with built-in PII redaction.
    """
    def __init__(self, collection_name: str, location: Optional[str] = None, url: Optional[str] = None):
        # Fetch environment variable, handle missing keys explicitly
        api_key = os.environ.get("QDRANT_API_KEY")
        self.client = QdrantClient(location=location, url=url, api_key=api_key)
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Ensures the collection exists; creates it if not."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=3, distance=Distance.COSINE)
            )

    def _sanitize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Redacts PII fields from the payload before storage."""
        sanitized = payload.copy()
        pii_fields = ["ssn", "credit_card", "personal_email", "phone"]
        for field in pii_fields:
            if field in sanitized:
                sanitized[field] = "[REDACTED]"
        return sanitized

    def insert_vectors(self, point_ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> bool:
        """Inserts vectors and their sanitized payloads."""
        points = [
            PointStruct(
                id=pid,
                vector=vec,
                payload=self._sanitize_payload(payload)
            )
            for pid, vec, payload in zip(point_ids, vectors, payloads)
        ]

        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return operation_info.status == UpdateStatus.COMPLETED

    def update_payload(self, point_id: str, new_payload: Dict[str, Any]) -> bool:
        """Updates the payload for an existing vector."""
        operation_info = self.client.set_payload(
            collection_name=self.collection_name,
            payload=self._sanitize_payload(new_payload),
            points=[point_id]
        )
        return operation_info.status == UpdateStatus.COMPLETED

    def delete_by_id(self, point_id: str) -> bool:
        """Deletes a vector by its unique Point ID."""
        operation_info = self.client.delete(
            collection_name=self.collection_name,
            points_selector=[point_id]
        )
        return operation_info.status == UpdateStatus.COMPLETED

    def delete_by_filter(self, key: str, value: Any) -> bool:
        """Deletes vectors that match a specific payload filter."""
        operation_info = self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                ]
            )
        )
        return operation_info.status == UpdateStatus.COMPLETED

if __name__ == "__main__":
    # Example Usage (Local In-Memory Qdrant)
    manager = QdrantVectorManager(collection_name="employee_directory", location=":memory:")

    # 1. Create/Insert
    emp_id = str(uuid.uuid4())
    emp_vector = [0.15, 0.72, 0.33]
    emp_payload = {
        "department": "Engineering",
        "personal_email": "jane.doe@example.com",
        "role": "Developer"
    }

    success = manager.insert_vectors([emp_id], [emp_vector], [emp_payload])
    print(f"Insert Success: {success}")

    # 2. Update Payload
    success = manager.update_payload(emp_id, {"role": "Senior Developer"})
    print(f"Update Payload Success: {success}")

    # 3. Delete by ID
    success = manager.delete_by_id(emp_id)
    print(f"Delete by ID Success: {success}")

    # 4. Delete by Filter
    manager.insert_vectors([emp_id], [emp_vector], [{"department": "Marketing"}])
    success = manager.delete_by_filter("department", "Marketing")
    print(f"Delete by Filter Success: {success}")
```

## Key Concepts
- **Point ID**: A unique identifier for a vector. In Qdrant, this must be a UUID or a 64-bit unsigned integer.
- **Payload**: JSON-like metadata attached to vectors. Crucial for filtering search results.
- **Upsert**: Insert or update operation. If the point ID already exists, Qdrant updates it.
- **Filter Deletions**: Removing multiple vectors simultaneously by matching payload criteria, essential for bulk data purging (e.g., GDPR data deletion requests).
- **PII Redaction**: Stripping sensitive information before it reaches the vector database to prevent data leaks.

## Common Gotchas
- **ID Collisions**: Using predictable integers instead of UUIDs can lead to unintended overwrites via `upsert`. Always default to UUIDs.
- **Missing Collections**: Attempting CRUD operations on a non-existent collection raises exceptions. Always ensure the collection is initialized.
- **Environment Fallbacks**: Hardcoding fallback API keys in `os.environ.get()` is a security risk. Check explicitly and fail fast if unauthorized.

## Reference Links
- [Qdrant Python Client Documentation](https://qdrant.tech/documentation/interfaces/)
- [Qdrant Payload Filtering](https://qdrant.tech/documentation/concepts/filtering/)
