import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def create_notebook():
    nb = new_notebook()

    # ---------------------------------------------------------
    # Core Theory
    # ---------------------------------------------------------
    theory_md = """# Day 18: Advanced Qdrant Operations - Filtered Searches & Scroll APIs

Welcome to Day 18! Today we tackle a critical component of real-world AI engineering: **Managing and querying large vector datasets.**

When building production RAG systems or semantic search engines, simply returning the top K nearest neighbors based purely on vector similarity is rarely sufficient. Real applications require:
1.  **Metadata Filtering:** Filtering out documents based on structured metadata (e.g., date, author, permissions, category) before or during the similarity search. This ensures users only see data they are authorized to see or that is contextually relevant.
2.  **Pagination and Scrolling:** When you have thousands of matches or need to iterate over large datasets for batch processing (like migrations or offline analytics), you need a reliable way to cursor through data without loading everything into memory.

## The "Why" and "How"

**Filtering (Payload Filtering):**
Qdrant allows you to attach JSON payloads (metadata) to your vectors. Filtered searches apply conditions to these payloads. Qdrant handles this efficiently at the engine level, combining dense vector indices (like HNSW) with payload indices. This is crucial for *multi-tenant* applications where data separation is required.

**Scroll API:**
The Scroll API is designed to sequentially iterate over all (or filtered) records in a collection. It uses an `offset` (a point ID) to fetch the next batch. This is fundamentally different from a vector search; it's a structural traversal. It's the standard pattern for data dumps, re-indexing, and background batch jobs.

Let's dive into the production-grade implementation of these operations."""

    nb.cells.append(new_markdown_cell(theory_md))

    # ---------------------------------------------------------
    # Code Implementation: Setup
    # ---------------------------------------------------------
    setup_md = """## 1. Setup and Data Ingestion
First, let's initialize a Qdrant client (in-memory for this module) and populate it with some dummy records containing structured payloads."""
    nb.cells.append(new_markdown_cell(setup_md))

    setup_code = """from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue

# Type hints for our records
RecordPayload = Dict[str, Any]

def initialize_qdrant() -> QdrantClient:
    \"\"\"Initializes an in-memory Qdrant client and creates a collection.\"\"\"
    client = QdrantClient(":memory:")

    client.create_collection(
        collection_name="tech_articles",
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    return client

def ingest_data(client: QdrantClient) -> None:
    \"\"\"Ingests sample points with payloads into the collection.\"\"\"
    points = [
        PointStruct(id=1, vector=[0.1, 0.2, 0.3, 0.4], payload={"category": "AI", "author": "Alice", "year": 2023}),
        PointStruct(id=2, vector=[0.9, 0.8, 0.7, 0.6], payload={"category": "WebDev", "author": "Bob", "year": 2022}),
        PointStruct(id=3, vector=[0.2, 0.1, 0.4, 0.3], payload={"category": "AI", "author": "Charlie", "year": 2024}),
        PointStruct(id=4, vector=[0.5, 0.5, 0.5, 0.5], payload={"category": "DataEng", "author": "Alice", "year": 2024}),
        PointStruct(id=5, vector=[0.8, 0.9, 0.1, 0.2], payload={"category": "WebDev", "author": "Charlie", "year": 2021}),
    ]

    client.upsert(
        collection_name="tech_articles",
        points=points
    )
    print(f"Successfully ingested {len(points)} records.")

# Initialize and ingest
client = initialize_qdrant()
ingest_data(client)
"""
    nb.cells.append(new_code_cell(setup_code))

    # ---------------------------------------------------------
    # Code Implementation: Filtered Search
    # ---------------------------------------------------------
    filter_md = """## 2. Advanced Filtered Search
Here we demonstrate how to perform a vector search while enforcing metadata constraints. This is a common pattern for multi-tenant architectures or granular querying."""
    nb.cells.append(new_markdown_cell(filter_md))

    filter_code = """def search_with_filter(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    category_filter: str,
    limit: int = 2
) -> None:
    \"\"\"
    Performs a vector similarity search filtered by a specific category.

    Args:
        client: The Qdrant client instance.
        collection_name: Name of the collection to search.
        query_vector: The dense vector to search against.
        category_filter: The category value to filter on.
        limit: Max number of results to return.
    \"\"\"
    # Define the filter using Qdrant's Filter and FieldCondition models
    query_filter = Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value=category_filter)
            )
        ]
    )

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=limit
    )

    print(f"\\n--- Search Results for Category: {category_filter} ---")
    for hit in results.points:
        print(f"ID: {hit.id}, Score: {hit.score:.4f}, Payload: {hit.payload}")

# Execute filtered search
search_with_filter(client, "tech_articles", [0.1, 0.2, 0.3, 0.4], "AI")
"""
    nb.cells.append(new_code_cell(filter_code))

    # ---------------------------------------------------------
    # Code Implementation: Scroll API
    # ---------------------------------------------------------
    scroll_md = """## 3. Using the Scroll API for Batch Processing
The Scroll API is used to iterate over all records, optionally applying a filter. It returns a tuple containing the batch of records and the ID of the next record to continue scrolling from."""
    nb.cells.append(new_markdown_cell(scroll_md))

    scroll_code = """def scroll_all_records(client: QdrantClient, collection_name: str, batch_size: int = 2) -> None:
    \"\"\"
    Iterates through all records in a collection using the Scroll API.

    Args:
        client: The Qdrant client instance.
        collection_name: Name of the collection.
        batch_size: Number of records to fetch per scroll iteration.
    \"\"\"
    print(f"\\n--- Scrolling through all records (Batch Size: {batch_size}) ---")
    next_page_offset = None
    iteration = 1

    while True:
        records, next_page_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=next_page_offset,
            with_payload=True,
            with_vectors=False # Often we don't need vectors for batch processing metadata
        )

        print(f"Batch {iteration}: Retrieved {len(records)} records.")
        for record in records:
             print(f"  Record ID: {record.id}, Payload: {record.payload}")

        if next_page_offset is None:
            print("Reached the end of the collection.")
            break

        iteration += 1

# Execute scroll
scroll_all_records(client, "tech_articles")
"""
    nb.cells.append(new_code_cell(scroll_code))

    # ---------------------------------------------------------
    # Common Pitfalls
    # ---------------------------------------------------------
    pitfalls_md = """## Common Pitfalls in Production

1.  **Missing Payload Indices:** Filtering on unindexed payload fields requires a full scan of the dataset, which is disastrous for performance. *Always* create payload indices (`client.create_payload_index(...)`) for fields you frequently filter by.
2.  **Using Search for Batch Export:** Developers sometimes try to use `client.search` with a massive `limit` to export data. This overloads the memory and the HNSW index. Use `client.scroll` instead.
3.  **Complex Filter Logic:** Deeply nested `must`, `should`, and `must_not` clauses can become difficult to debug. Keep filter logic as flat as possible, or handle complex business logic upstream before querying the database.
4.  **Offset Mismanagement:** When implementing pagination on the frontend, relying on the `offset` parameter in `search` for deep pagination is inefficient. It forces the engine to calculate all results up to `offset + limit`. For deep pagination or full traversal, use the Scroll API."""
    nb.cells.append(new_markdown_cell(pitfalls_md))

    # ---------------------------------------------------------
    # Practical Lab / Homework
    # ---------------------------------------------------------
    lab_md = """## Practical Lab: Multi-Tenant Batch Processor

**Your Task:**
You are building an administrative tool for a multi-tenant platform.

1.  Create a function `process_tenant_data(client, collection_name, author_name)` that uses the **Scroll API** to retrieve *all* records authored by a specific `author_name`.
2.  The scroll should fetch records in batches of 2.
3.  Print the total number of records found for that author.

*Hint: You will need to combine the `scroll` method with a `Filter` object, similar to how we used it in the `search` method.*"""
    nb.cells.append(new_markdown_cell(lab_md))

    lab_code = """# Lab Implementation
def process_tenant_data(client: QdrantClient, collection_name: str, author_name: str, batch_size: int = 2) -> None:
    \"\"\"
    Uses the Scroll API to retrieve all records for a specific author in batches.
    \"\"\"
    # Define the filter for the specific author
    author_filter = Filter(
        must=[
            FieldCondition(
                key="author",
                match=MatchValue(value=author_name)
            )
        ]
    )

    print(f"\\n--- Processing Data for Tenant/Author: {author_name} ---")
    next_page_offset = None
    total_records = 0
    iteration = 1

    while True:
        records, next_page_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=author_filter,
            limit=batch_size,
            offset=next_page_offset,
            with_payload=True,
            with_vectors=False
        )

        batch_count = len(records)
        total_records += batch_count
        print(f"Batch {iteration}: Retrieved {batch_count} records.")
        for record in records:
             print(f"  Record ID: {record.id}, Payload: {record.payload}")

        if next_page_offset is None:
            break

        iteration += 1

    print(f"Total records processed for {author_name}: {total_records}")

# Execute Lab task for author "Alice"
process_tenant_data(client, "tech_articles", "Alice")
"""
    nb.cells.append(new_code_cell(lab_code))

    # Write notebook to file
    with open('modules/day_18.ipynb', 'w') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    create_notebook()
