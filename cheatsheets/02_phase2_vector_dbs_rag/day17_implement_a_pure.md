# Day 17: Implement a "Pure Python RAG" pipeline

## Concise Summaries
A Pure Python RAG (Retrieval-Augmented Generation) pipeline operates without external vector databases or specialized frameworks. It manually retrieves context by computing vector similarity (e.g., cosine similarity) between a query embedding and document embeddings stored in a simple structure (like a JSON file). The retrieved context is then injected into a prompt template and sent to an LLM.

## Code Snippets

```python
import json
import numpy as np

class PurePythonRAG:
    """
    A minimal implementation of a RAG pipeline without external RAG frameworks.
    Demonstrates computing cosine similarity and building context.
    """
    def __init__(self, document_db_path: str):
        self.document_db_path = document_db_path
        self._load_db()

    def _load_db(self) -> None:
        """Loads the document database from a JSON file."""
        try:
            with open(self.document_db_path, "r", encoding="utf-8") as f:
                self.db = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Database file not found at {self.document_db_path}")

    def compute_cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Computes the cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def retrieve_context(self, query_embedding: list[float], top_k: int = 1) -> str:
        """Retrieves the top-k most similar documents based on cosine similarity."""
        similarities = []
        for doc in self.db:
            if "embedding" not in doc or "text" not in doc:
                continue

            score = self.compute_cosine_similarity(query_embedding, doc["embedding"])
            similarities.append((score, doc["text"]))

        # Sort by similarity score descending
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Select top_k and concatenate
        top_docs = similarities[:top_k]
        context = "\n".join([doc_text for score, doc_text in top_docs])
        return context

    def generate_prompt(self, query: str, context: str) -> str:
        """Injects context into a prompt template for the LLM."""
        prompt = f"""You are a helpful assistant. Use the following context to answer the question.
If the answer is not in the context, say "I don't know based on the provided context."

Context:
{context}

Question:
{query}

Answer:"""
        return prompt

if __name__ == "__main__":
    # 1. Create a dummy JSON DB for testing
    dummy_db = [
        {"id": "doc1", "text": "The secret code is 42.", "embedding": [1.0, 0.0, 0.0]},
        {"id": "doc2", "text": "Python is a programming language.", "embedding": [0.0, 1.0, 0.0]},
        {"id": "doc3", "text": "Paris is the capital of France.", "embedding": [0.0, 0.0, 1.0]}
    ]
    with open("dummy_db.json", "w") as f:
        json.dump(dummy_db, f)

    # 2. Initialize RAG
    rag = PurePythonRAG("dummy_db.json")

    # 3. Dummy query and its embedding (let's say it's looking for "secret code")
    query = "What is the secret code?"
    query_embedding = [0.9, 0.1, 0.0]  # Similar to doc1

    # 4. Retrieve context
    context = rag.retrieve_context(query_embedding, top_k=1)

    # 5. Generate prompt
    prompt = rag.generate_prompt(query, context)

    print("Generated Prompt:")
    print(prompt)

    # Clean up dummy DB
    import os
    if os.path.exists("dummy_db.json"):
        os.remove("dummy_db.json")
```

## Key Concepts
- **Cosine Similarity:** A metric used to measure how similar two vectors are. It measures the cosine of the angle between two non-zero vectors.
- **Query Embedding:** The vector representation of the user's input question.
- **Document Embedding:** The vector representation of the context documents stored in the database.
- **Context Injection:** The process of taking the retrieved text and formatting it directly into the final prompt for the LLM.
- **AI Security (PII Protection):** When manually injecting context, ensure the retrieved text from the JSON DB does not inadvertently expose sensitive PII in the generated prompt sent to an external API.

## Common Gotchas
- **Zero Vector Errors:** Forgetting to handle zero vectors when computing cosine similarity (can cause division by zero errors).
- **Scalability Issues:** Loading an entire database of dense vectors into memory (like a JSON file) does not scale for thousands/millions of documents; production requires an actual Vector DB (like Qdrant or Milvus).

## Reference Links
- [NumPy Documentation on Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [Cosine Similarity (Wikipedia)](https://en.wikipedia.org/wiki/Cosine_similarity)
