import json

def create_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Day 29: Hybrid Search in Qdrant (Dense + Sparse Vectors)\n",
                    "\n",
                    "Welcome to Day 29! Today we are moving into Phase 3: Advanced RAG & Agentic AI.\n",
                    "\n",
                    "## Core Theory (Just-in-Time)\n",
                    "\n",
                    "**Why Hybrid Search?**\n",
                    "Standard vector databases rely on dense vectors (embeddings like OpenAI's `text-embedding-3-small`) to find semantic similarity. If a user asks \"Where is the company headquarters?\", dense vectors easily match with \"Our main office is located in Seattle.\"\n",
                    "\n",
                    "However, dense vectors often struggle with **exact keyword matches**, acronyms, part numbers, or specific names. If a user searches for \"Error code 0x80040154\", a dense embedding might match generic error troubleshooting docs, completely missing the specific exact match.\n",
                    "\n",
                    "**Sparse Vectors (BM25) to the Rescue**\n",
                    "Sparse vectors represent text based on term frequency (like BM25, TF-IDF). They are mostly zeros, with non-zero values representing specific words. They excel at exact keyword matching.\n",
                    "\n",
                    "**Hybrid Search** combines both: dense vectors for meaning, and sparse vectors for keywords. Qdrant supports storing both on the same point and using **Reciprocal Rank Fusion (RRF)** to combine the results beautifully."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Code Implementation\n",
                    "\n",
                    "Let's look at a production-grade example of setting up a Qdrant collection for hybrid search, inserting data, and querying it using the modern `query_points` API."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "from typing import List, Dict, Any\n",
                    "from qdrant_client import QdrantClient\n",
                    "from qdrant_client.models import (\n",
                    "    Distance,\n",
                    "    VectorParams,\n",
                    "    SparseVectorParams,\n",
                    "    SparseIndexParams,\n",
                    "    SparseVector,\n",
                    "    PointStruct,\n",
                    "    Prefetch,\n",
                    "    FusionQuery,\n",
                    "    Fusion,\n",
                    ")\n",
                    "# We'll use a local in-memory Qdrant client for this example\n",
                    "client = QdrantClient(\":memory:\")\n",
                    "\n",
                    "COLLECTION_NAME = \"hybrid_documents\"\n",
                    "\n",
                    "def setup_hybrid_collection(client: QdrantClient, collection_name: str) -> None:\n",
                    "    \"\"\"\n",
                    "    Creates a Qdrant collection configured for both dense and sparse vectors.\n",
                    "    \"\"\"\n",
                    "    if client.collection_exists(collection_name):\n",
                    "        client.delete_collection(collection_name)\n",
                    "\n",
                    "    # In Qdrant, we use named vectors when we have multiple vector types\n",
                    "    client.create_collection(\n",
                    "        collection_name=collection_name,\n",
                    "        vectors_config={\n",
                    "            \"dense\": VectorParams(size=384, distance=Distance.COSINE),\n",
                    "        },\n",
                    "        sparse_vectors_config={\n",
                    "            \"sparse\": SparseVectorParams(\n",
                    "                index=SparseIndexParams(\n",
                    "                    on_disk=False,\n",
                    "                )\n",
                    "            )\n",
                    "        }\n",
                    "    )\n",
                    "    print(f\"Collection '{collection_name}' created successfully.\")\n",
                    "\n",
                    "setup_hybrid_collection(client, COLLECTION_NAME)\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Mock Embedding Generation\n",
                    "\n",
                    "To demonstrate, we will mock the generation of dense and sparse vectors. In production, you'd use a model like `text-embedding-3-small` for dense, and `SPLADE` or `FastEmbed` for sparse vectors."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import random\n",
                    "\n",
                    "def mock_dense_embedding() -> List[float]:\n",
                    "    \"\"\"Generates a mock dense embedding (size 384).\"\"\"\n",
                    "    return [random.uniform(-1.0, 1.0) for _ in range(384)]\n",
                    "\n",
                    "def mock_sparse_embedding() -> SparseVector:\n",
                    "    \"\"\"\n",
                    "    Generates a mock sparse embedding.\n",
                    "    Indices represent token IDs, values represent token weights (e.g., TF-IDF).\n",
                    "    \"\"\"\n",
                    "    # Randomly select some token IDs to simulate keyword presence\n",
                    "    indices = sorted(random.sample(range(10000), k=15))\n",
                    "    values = [random.uniform(0.5, 2.0) for _ in indices]\n",
                    "    return SparseVector(indices=indices, values=values)\n",
                    "\n",
                    "def insert_hybrid_data(client: QdrantClient, collection_name: str) -> None:\n",
                    "    \"\"\"Inserts sample documents with both dense and sparse vectors.\"\"\"\n",
                    "    points = []\n",
                    "    docs = [\n",
                    "        \"The quick brown fox jumps over the lazy dog.\",\n",
                    "        \"Error code 0x80040154 usually means class not registered.\",\n",
                    "        \"The corporate headquarters is located in Austin, Texas.\",\n",
                    "    ]\n",
                    "    \n",
                    "    for i, doc in enumerate(docs):\n",
                    "        point = PointStruct(\n",
                    "            id=i + 1,\n",
                    "            vector={\n",
                    "                \"dense\": mock_dense_embedding(),\n",
                    "                \"sparse\": mock_sparse_embedding(),\n",
                    "            },\n",
                    "            payload={\"text\": doc}\n",
                    "        )\n",
                    "        points.append(point)\n",
                    "\n",
                    "    client.upsert(collection_name=collection_name, points=points)\n",
                    "    print(f\"Inserted {len(points)} documents.\")\n",
                    "\n",
                    "insert_hybrid_data(client, COLLECTION_NAME)\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Querying with Hybrid Search\n",
                    "\n",
                    "In modern Qdrant (v1.19.0+), we use `query_points` with `Prefetch` to perform hybrid search and Reciprocal Rank Fusion (RRF) on the database side."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def hybrid_search(client: QdrantClient, collection_name: str, query_text: str) -> None:\n",
                    "    \"\"\"\n",
                    "    Performs a hybrid search combining dense and sparse results.\n",
                    "    \"\"\"\n",
                    "    print(f\"\\n--- Hybrid Search for: '{query_text}' ---\")\n",
                    "    \n",
                    "    # Generate vectors for the query\n",
                    "    query_dense = mock_dense_embedding()\n",
                    "    query_sparse = mock_sparse_embedding()\n",
                    "\n",
                    "    # Use Prefetch to define the two separate searches\n",
                    "    response = client.query_points(\n",
                    "        collection_name=collection_name,\n",
                    "        prefetch=[\n",
                    "            # Dense search prefetch\n",
                    "            Prefetch(\n",
                    "                query=query_dense,\n",
                    "                using=\"dense\",\n",
                    "                limit=5,\n",
                    "            ),\n",
                    "            # Sparse search prefetch\n",
                    "            Prefetch(\n",
                    "                query=query_sparse,\n",
                    "                using=\"sparse\",\n",
                    "                limit=5,\n",
                    "            ),\n",
                    "        ],\n",
                    "        # The outer query defines how to combine the prefetch results.\n",
                    "        # In qdrant-client >= 1.9.0, we use FusionQuery to explicitly perform RRF.\n",
                    "        query=FusionQuery(fusion=Fusion.RRF), \n",
                    "        limit=3,\n",
                    "        with_payload=True\n",
                    "    )\n",
                    "\n",
                    "    # Access results via the .points attribute\n",
                    "    for point in response.points:\n",
                    "        print(f\"Score: {point.score:.4f} | Text: {point.payload.get('text')}\")\n",
                    "\n",
                    "hybrid_search(client, COLLECTION_NAME, \"What is error 0x80040154?\")\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Common Pitfalls\n",
                    "\n",
                    "1. **Using Deprecated Search API:** Using the old `.search()` or `search_batch()` methods for hybrid retrieval instead of the new `query_points()` API. The old methods require complex manual chunking and client-side fusion. `query_points()` handles RRF natively on the server side.\n",
                    "2. **Forgetting Named Vectors:** You must explicitly define `vectors_config` as a dictionary (e.g., `{\"dense\": VectorParams(...) }`) to use named vectors. If you try to mix unnamed default vectors with sparse vectors, the architecture becomes confusing to manage.\n",
                    "3. **Inconsistent Encoders:** Using different tokenizers/models to generate the sparse embeddings during ingestion vs. retrieval. Ensure the exact same model (e.g., BM25 or SPLADE) generates the `SparseVector` objects at both steps."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Practical Lab\n",
                    "\n",
                    "**Your Task:**\n",
                    "Write a Python script that extends the example above. \n",
                    "1. Define a Pydantic model `Document` containing `id: int`, `text: str`, and `category: str`.\n",
                    "2. Create a list of 5 `Document` objects about different topics.\n",
                    "3. Insert these documents into a new Qdrant collection named `lab_hybrid`, storing both dense and sparse vectors.\n",
                    "4. Implement a `query_points()` call that filters the hybrid search to only return documents where `category == 'technical'`.\n",
                    "\n",
                    "*Hint: You'll need to use Qdrant's `Filter` and `FieldCondition` models inside the `query_points()` call.*"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Your Lab Implementation Here\n",
                    "from pydantic import BaseModel\n",
                    "from qdrant_client.models import Filter, FieldCondition, MatchValue\n",
                    "\n",
                    "class Document(BaseModel):\n",
                    "    id: int\n",
                    "    text: str\n",
                    "    category: str\n",
                    "\n",
                    "# Write your lab code below!"
                ]
            }
        ],
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
                "version": "3.9.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open('modules/day_29.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

if __name__ == '__main__':
    create_notebook()
