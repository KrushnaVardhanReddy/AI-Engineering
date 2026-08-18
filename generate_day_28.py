import json
import os

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_markdown(source: str):
        lines = [line + "\n" for line in source.split("\n")]
        # Remove trailing newline from the last element to match Jupyter format
        if lines:
            lines[-1] = lines[-1].rstrip("\n")

        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": lines
        })

    def add_code(source: str):
        lines = [line + "\n" for line in source.split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")

        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        })

    # Core Theory
    add_markdown("""# Day 28: Finalize the "Smart Chunking" Project

Welcome to Day 28! Today we finalize our "Smart Chunking" pipeline. Traditional chunking methods (like naive character splitting) often destroy the structure of tabular data. When a table is split haphazardly, the LLM loses the row-column relationship, leading to hallucinations or incorrect answers during retrieval.

To solve this, we need a pipeline that explicitly recognizes tables and chunks them intelligently—often by keeping rows intact and repeating column headers, or by converting the table into a text-based format (like Markdown or HTML) before embedding.

## The "Why" and "How"

**Why table structure matters:**
Relational data relies on its two-dimensional structure. If a chunk contains "Revenue: $500" but misses the "Q3 2023" column header, the data is useless.

**How we solve it (The Architecture):**
1. **Extraction:** We use standard tools to extract tables in a structured format (e.g., Markdown).
2. **Structural Chunking:** We chunk the text such that table rows remain attached to their headers. Here we implement a robust row-aware approach that ensures each chunk contains the table's context.
3. **Retrieval:** We store the structured chunks in Qdrant and use `query_points()` to retrieve them based on semantic similarity, ensuring the structured context is preserved for the LLM.""")

    # Code Implementation: Setup
    add_markdown("## 1. Setup and Initialization\nLet's initialize our in-memory Qdrant client to store our structured table chunks.")

    add_code("""from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain_core.documents import Document

def initialize_qdrant_for_tables() -> QdrantClient:
    \"\"\"Initializes an in-memory Qdrant client for our smart chunking project.\"\"\"
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="structured_tables",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    return client

client = initialize_qdrant_for_tables()
print("Qdrant initialized successfully.")""")

    # Code Implementation: Smart Chunking
    add_markdown("## 2. Structural Chunking Logic\nHere we implement a parser that reads a Markdown table and splits it while preserving the headers.")

    add_code("""def smart_table_chunker(table_markdown: str, max_chunk_size: int = 150) -> List[Document]:
    \"\"\"
    Chunks a Markdown table intelligently by ensuring headers are preserved.

    Args:
        table_markdown: The table formatted as a Markdown string.
        max_chunk_size: Maximum characters per chunk.

    Returns:
        A list of LangChain Document objects representing the chunks.
    \"\"\"
    lines = table_markdown.strip().split("\\n")
    if len(lines) < 3:
        return [Document(page_content=table_markdown, metadata={"type": "table_fragment"})]

    header = lines[0]
    separator = lines[1]
    data_rows = lines[2:]

    chunks: List[Document] = []
    current_chunk_lines: List[str] = [header, separator]
    current_length = len(header) + len(separator) + 2

    for row in data_rows:
        row_length = len(row) + 1
        if current_length + row_length > max_chunk_size and len(current_chunk_lines) > 2:
            chunk_content = "\\n".join(current_chunk_lines)
            chunks.append(Document(page_content=chunk_content, metadata={"type": "table_chunk"}))
            current_chunk_lines = [header, separator, row]
            current_length = len(header) + len(separator) + len(row) + 3
        else:
            current_chunk_lines.append(row)
            current_length += row_length

    if len(current_chunk_lines) > 2:
        chunk_content = "\\n".join(current_chunk_lines)
        chunks.append(Document(page_content=chunk_content, metadata={"type": "table_chunk"}))

    return chunks

# Example Table
sample_table = \"\"\"| Product | Q1 Revenue | Q2 Revenue |
|---|---|---|
| Widget A | $10,000 | $12,000 |
| Widget B | $15,000 | $14,500 |
| Widget C | $8,000 | $9,000 |
| Widget D | $20,000 | $22,000 |\"\"\"

table_chunks = smart_table_chunker(sample_table, max_chunk_size=100)
for i, chunk in enumerate(table_chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk.page_content)
    print()""")

    # Code Implementation: Ingestion & Retrieval
    add_markdown("## 3. Ingestion and Retrieval Pipeline\nNow we will store these chunks and query them. We use a deterministic mock embedding function for local execution without API dependencies.")

    add_code("""import hashlib

def mock_embed_text(text: str, vector_size: int = 384) -> List[float]:
    \"\"\"
    Generates a deterministic pseudo-random vector for a given text.
    In production, replace this with a real embedding model.
    \"\"\"
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()

    # Generate floats between -1.0 and 1.0
    vector = [(b / 128.0) - 1.0 for b in hash_bytes]

    # Pad or truncate to the desired vector size
    if len(vector) < vector_size:
        vector = (vector * (vector_size // len(vector) + 1))[:vector_size]
    else:
        vector = vector[:vector_size]

    return vector

def ingest_chunks(client: QdrantClient, chunks: List[Document], collection_name: str = "structured_tables") -> None:
    \"\"\"Ingests table chunks into Qdrant.\"\"\"
    points = []
    for i, chunk in enumerate(chunks):
        vector = mock_embed_text(chunk.page_content)
        points.append(
            PointStruct(
                id=i + 1,
                vector=vector,
                payload={"text": chunk.page_content, "type": chunk.metadata.get("type", "unknown")}
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print(f"Successfully ingested {len(points)} chunks into '{collection_name}'.")

def retrieve_table_context(client: QdrantClient, query: str, collection_name: str = "structured_tables", limit: int = 2) -> None:
    \"\"\"Retrieves chunks from Qdrant using the query_points API.\"\"\"
    query_vector = mock_embed_text(query)

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit
    )

    print(f"\\n--- Search Results for: '{query}' in '{collection_name}' ---")
    for hit in results.points:
        print(f"Score: {hit.score:.4f}")
        print(f"Content:\\n{hit.payload.get('text')}")
        print("-" * 30)

ingest_chunks(client, table_chunks)
retrieve_table_context(client, "What is the revenue for Widget B?")""")

    # Common Pitfalls
    add_markdown("""## Common Pitfalls in Production

1. **Stripping Headers Too Early:** Using standard text splitters on HTML or Markdown tables often strips the `<th>` tags or Markdown headers, making subsequent rows meaningless to the LLM.
2. **Context Window Overflow:** If a table is extremely wide (many columns), a single row might exceed the maximum chunk size. In these cases, you must pivot the table (e.g., converting rows into key-value text pairs) before chunking.
3. **Ignoring Cell Spans:** Complex tables with merged cells (`rowspan`/`colspan`) are notoriously difficult to parse into Markdown. Specialized vision models or advanced OCR are often required before text chunking can even begin.
4. **Using Deprecated APIs:** Always use modern API endpoints. For Qdrant, use `client.query_points()` instead of the deprecated `client.search()`. Ensure you access the payload via `.points`.""")

    # Practical Lab
    add_markdown("""## Practical Lab: Key-Value Table Transformation

**Your Task:**
Sometimes, row-based chunking is not enough if columns are too wide. A better approach is to convert each cell into a self-contained sentence or key-value pair.

1. Write a function `table_to_key_value(table_markdown: str)` that takes the provided Markdown table.
2. It should output a list of LangChain `Document` objects where each document represents a single data cell with full context.
   *Example output for a cell:* `Document(page_content="Product: Widget A | Q1 Revenue: $10,000")`
3. Ingest these new chunks into a new Qdrant collection called `"key_value_tables"`.
4. Perform a query using `query_points()` and print the result.""")

    # Lab Implementation
    add_code("""def table_to_key_value(table_markdown: str) -> List[Document]:
    \"\"\"
    Converts a Markdown table into self-contained key-value documents.
    \"\"\"
    lines = table_markdown.strip().split("\\n")
    if len(lines) < 3:
        return []

    headers = [h.strip() for h in lines[0].strip("|").split("|") if h.strip()]
    data_rows = lines[2:]

    documents: List[Document] = []

    for row in data_rows:
        cells = [c.strip() for c in row.strip("|").split("|") if c.strip() or "|" in row]
        if not cells:
            continue

        primary_entity_col = headers[0] if headers else "Entity"
        primary_entity_val = cells[0]

        for i in range(1, len(cells)):
            if i < len(headers):
                content = f"{primary_entity_col}: {primary_entity_val} | {headers[i]}: {cells[i]}"
                documents.append(Document(page_content=content, metadata={"type": "kv_chunk"}))

    return documents

# Execute Lab task
kv_chunks = table_to_key_value(sample_table)
for i, chunk in enumerate(kv_chunks[:3]):  # Print first 3 to verify
    print(f"KV Chunk {i+1}: {chunk.page_content}")

# Initialize new collection
client.create_collection(
    collection_name="key_value_tables",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Ingest and query
ingest_chunks(client, kv_chunks, collection_name="key_value_tables")
retrieve_table_context(client, "Widget C Q2 Revenue", collection_name="key_value_tables")""")

    # Ensure target directory exists
    os.makedirs('modules', exist_ok=True)

    # Write notebook to file
    with open('modules/day_28.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)

if __name__ == "__main__":
    create_notebook()
    print("Notebook modules/day_28.ipynb generated successfully.")
