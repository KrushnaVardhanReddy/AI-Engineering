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
    add_markdown("""# Day 34: Optimizing RAG Pipelines with DSPy

Welcome to Day 34! Today we shift from manual prompt engineering to *programmatic prompt optimization* using **DSPy**.

## Core Theory: The "Why" and "How"

**Why DSPy?**
In traditional RAG pipelines, developers spend a significant amount of time "prompt tweaking"—changing a few words in the prompt, adding examples (Few-Shot), and hoping the LLM's performance improves. This process is brittle and unscalable.
DSPy is a framework that *compiles* declarative language model programs. Instead of writing prompts by hand, you define a **Signature** (input/output requirements) and a **Module** (pipeline flow), and DSPy uses optimizers (teleprompters) to automatically generate optimal prompts and examples based on your data.

**How we solve it (The Architecture):**
1. **Define a Signature:** State clearly what inputs the module receives and what outputs it produces.
2. **Build a Module:** Compose multiple signatures/operations (like retrieving context and generating an answer) into a clear pipeline.
3. **Connect to Vector DB:** Use Qdrant (via custom integration or DSPy's built-in retriever capabilities) to fetch context.
4. **Optimization:** We will build the basic programmatic structure today, which forms the foundation for DSPy's automated optimizers (like `BootstrapFewShot`).""")

    # Code Implementation: Setup
    add_markdown("""## 1. Setup and Initialization

Let's initialize our mock LLM and an in-memory Qdrant client to act as our Vector Database.
For local testing without API keys, we'll configure DSPy with a dummy LM. In production, you would configure it with `dspy.LM('openai/gpt-4o')` or similar.""")

    add_code("""import os
import dspy
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# --- DSPy Mock LLM Setup ---
# We create a simple mock language model for local demonstration
class MockLM(dspy.LM):
    def __init__(self, model="mock"):
        super().__init__(model=model)
        self.provider = "mock"
    def __call__(self, prompt=None, messages=None, **kwargs):
        # Return a simple mock response for testing
        # Mocking dspy.LM in recent DSPy versions requires returning strings
        # that parse to the expected JSON or chat format.
        # The easiest way to mock for tests is to use an adapter or return properly formatted output.
        import json
        if "reasoning" in str(prompt) or (messages and "reasoning" in str(messages)):
             return [json.dumps({"reasoning": "This is mock reasoning.", "answer": "This is a mock answer."})]
        if "search_query" in str(prompt) or (messages and "search_query" in str(messages)):
             return [json.dumps({"search_query": "mock search query"})]

        return [json.dumps({"answer": "This is a mock answer."})]

# Configure DSPy to use our Mock LM
mock_lm = MockLM()
dspy.settings.configure(lm=mock_lm)

# --- Qdrant Setup ---
def initialize_qdrant() -> QdrantClient:
    \"\"\"Initializes an in-memory Qdrant client for document retrieval.\"\"\"
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="knowledge_base",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    return client

qdrant_client = initialize_qdrant()
print("DSPy Mock LM and Qdrant initialized successfully.")""")

    # Code Implementation: Ingestion and Custom Retriever
    add_markdown("""## 2. Ingestion and Custom DSPy Retriever

DSPy allows us to integrate external retrievers. We'll populate our Qdrant DB with some facts and build a function that retrieves them, which our DSPy Module will use.

We use `QdrantClient.query_points()` and access the results via `.points` as per standard Qdrant v1.19.0+ practices.""")

    add_code("""import hashlib

def mock_embed_text(text: str, vector_size: int = 384) -> List[float]:
    \"\"\"
    Generates a deterministic pseudo-random vector for a given text.
    \"\"\"
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()
    vector = [(b / 128.0) - 1.0 for b in hash_bytes]
    if len(vector) < vector_size:
        vector = (vector * (vector_size // len(vector) + 1))[:vector_size]
    else:
        vector = vector[:vector_size]
    return vector

def ingest_documents(client: QdrantClient, documents: List[str], collection_name: str = "knowledge_base") -> None:
    \"\"\"Ingests documents into Qdrant.\"\"\"
    points = []
    for i, doc in enumerate(documents):
        vector = mock_embed_text(doc)
        points.append(
            PointStruct(
                id=i + 1,
                vector=vector,
                payload={"text": doc}
            )
        )
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print(f"Successfully ingested {len(points)} documents.")

# Ingest sample data
sample_docs = [
    "DSPy is a framework for algorithmic prompt optimization.",
    "Qdrant is a powerful vector search engine written in Rust.",
    "LangChain is used for building applications with LLMs.",
    "The capital of France is Paris."
]
ingest_documents(qdrant_client, sample_docs)

# Custom Retriever Function
def retrieve_context(query: str, client: QdrantClient = qdrant_client, collection_name: str = "knowledge_base", top_k: int = 2) -> List[str]:
    \"\"\"
    Retrieves top_k documents from Qdrant based on the query.
    \"\"\"
    query_vector = mock_embed_text(query)
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    )

    # Access payload via .points
    contexts = [hit.payload.get("text", "") for hit in results.points if hit.payload]
    return contexts

print(f"Test Retrieval: {retrieve_context('Tell me about DSPy')}")""")

    # Code Implementation: DSPy Signatures and Modules
    add_markdown("""## 3. DSPy Signatures and Modules

Now we define the declarative structure of our RAG pipeline.
- **Signature (`GenerateAnswer`):** Defines the inputs (context, question) and output (answer).
- **Module (`RAGPipeline`):** Orchestrates the retrieval step and the generation step using `dspy.Predict` or `dspy.ChainOfThought`.""")

    add_code("""class GenerateAnswer(dspy.Signature):
    \"\"\"Answer questions based on the provided context.\"\"\"

    context = dspy.InputField(desc="Relevant facts to answer the question")
    question = dspy.InputField(desc="The user's question")
    answer = dspy.OutputField(desc="A concise, accurate answer")

class RAGPipeline(dspy.Module):
    \"\"\"
    A DSPy Module representing a standard Retrieval-Augmented Generation pipeline.
    \"\"\"
    def __init__(self, retrieve_fn):
        super().__init__()
        self.retrieve_fn = retrieve_fn
        # We use ChainOfThought for better reasoning before outputting the final answer
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str) -> dspy.Prediction:
        \"\"\"
        The execution logic of the RAG pipeline.
        \"\"\"
        # 1. Retrieve relevant contexts
        contexts = self.retrieve_fn(question)
        context_str = "\\n".join(contexts)

        # 2. Generate answer
        prediction = self.generate_answer(context=context_str, question=question)
        return dspy.Prediction(context=contexts, answer=prediction.answer)

# Instantiate the pipeline
rag_pipeline = RAGPipeline(retrieve_fn=retrieve_context)

# Test the pipeline
question = "What is DSPy?"
result = rag_pipeline(question=question)

print(f"Question: {question}")
print(f"Retrieved Context: {result.context}")
print(f"Generated Answer: {result.answer}")""")

    # Common Pitfalls
    add_markdown("""## Common Pitfalls in Production

1. **Treating DSPy Modules like standard Python Functions:** `forward()` methods in DSPy are meant to define computational graphs for the optimizers. Do not put non-deterministic logic (like random sleeps or unstable external API calls not wrapped in DSPy tools) inside them if you plan to use automated optimizers.
2. **Ignoring the `dspy.Prediction` Object:** Always return a `dspy.Prediction` object from your `forward()` methods. It is required for the optimizers to correctly map outputs and calculate metrics during the compilation phase.
3. **Qdrant API Deprecation:** Remember that `client.search()` is deprecated in Qdrant v1.19.0+. Always use `client.query_points()` and extract the values through `.points`.
4. **Poor Context Formatting:** When passing `context` to a DSPy `InputField`, ensure it is properly stringified. Passing raw objects or deeply nested lists can cause the underlying LM prompt builder to format things poorly.""")

    # Practical Lab
    add_markdown("""## Practical Lab: Multi-Hop Retrieval Module

**Your Task:**
Sometimes a single retrieval isn't enough. You might need to retrieve context, generate a sub-query based on what's missing, retrieve *again*, and then answer.

1. Define a new Signature called `GenerateSearchQuery` that takes a `context` and a `question` and outputs a `search_query` (a string).
2. Create a new DSPy Module called `MultiHopRAG` that:
   - Retrieves initial context based on the original `question`.
   - Uses `GenerateSearchQuery` to see what else might be needed.
   - Retrieves *more* context based on the `search_query`.
   - Combines both sets of context and uses `GenerateAnswer` (or a similar signature) to produce the final `answer`.
3. Instantiate and run your `MultiHopRAG` module with a sample question.""")

    # Lab Implementation
    add_code("""class GenerateSearchQuery(dspy.Signature):
    \"\"\"Generate a follow-up search query based on current context to find missing information.\"\"\"
    context = dspy.InputField(desc="The information already known")
    question = dspy.InputField(desc="The ultimate question we are trying to answer")
    search_query = dspy.OutputField(desc="A new search query to find missing details")

class MultiHopRAG(dspy.Module):
    \"\"\"
    A DSPy Module for multi-hop Retrieval-Augmented Generation.
    \"\"\"
    def __init__(self, retrieve_fn):
        super().__init__()
        self.retrieve_fn = retrieve_fn
        self.generate_query = dspy.Predict(GenerateSearchQuery)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str) -> dspy.Prediction:
        # Hop 1: Initial Retrieval
        context_1 = self.retrieve_fn(question)
        context_str_1 = "\\n".join(context_1)

        # Hop 2: Generate follow-up query and retrieve more context
        query_prediction = self.generate_query(context=context_str_1, question=question)
        follow_up_query = query_prediction.search_query

        context_2 = self.retrieve_fn(follow_up_query)

        # Combine Contexts
        combined_context = context_1 + context_2
        # Remove duplicates
        unique_context = list(set(combined_context))
        context_str_final = "\\n".join(unique_context)

        # Final Generation
        final_prediction = self.generate_answer(context=context_str_final, question=question)

        return dspy.Prediction(
            initial_context=context_1,
            follow_up_query=follow_up_query,
            additional_context=context_2,
            answer=final_prediction.answer
        )

# Instantiate and Test the Lab Module
multi_hop_rag = MultiHopRAG(retrieve_fn=retrieve_context)

lab_question = "What is DSPy and what database is written in Rust?"
lab_result = multi_hop_rag(question=lab_question)

print(f"Question: {lab_question}")
print(f"Initial Context: {lab_result.initial_context}")
print(f"Follow-up Query Generated: {lab_result.follow_up_query}")
print(f"Additional Context Retrieved: {lab_result.additional_context}")
print(f"Final Answer: {lab_result.answer}")""")

    # Ensure target directory exists
    os.makedirs('modules', exist_ok=True)

    # Write notebook to file
    with open('modules/day_34.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)

if __name__ == "__main__":
    create_notebook()
    print("Notebook modules/day_34.ipynb generated successfully.")
