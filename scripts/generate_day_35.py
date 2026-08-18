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
    add_markdown("""# Day 35: Audit RAG Performance & "Lost in the Middle"

Welcome to Day 35 of the AI Engineering Mastery program! Today, we dive into auditing Retrieval-Augmented Generation (RAG) performance, focusing specifically on the **"Lost in the Middle"** phenomenon.

## Core Theory: What is "Lost in the Middle"?

When LLMs are presented with long contexts (like multiple retrieved documents in a RAG pipeline), they are remarkably good at extracting information located at the very **beginning** or the very **end** of the context. However, their performance drops significantly when the relevant information is buried in the **middle** of the context.

This happens because of how attention mechanisms in Transformers work. The model tends to heavily weight the start of the prompt (the instructions and the first few docs) and the end of the prompt (the most recent tokens seen before generation).

**Why does this matter in RAG?**
Standard vector search (like Qdrant or FAISS) returns documents sorted by relevance (cosine distance). If you retrieve 10 documents, the most relevant are first, and the least relevant are last. If you inject them into the prompt in this exact order:
1. Doc 1 (Highest relevance) - *Model pays attention*
2. Doc 2 (High relevance)
...
5. Doc 5 (Medium relevance) - *Model loses focus (Lost in the middle)*
...
10. Doc 10 (Lowest relevance) - *Model pays attention*

The LLM is paying the most attention to your best document (good) and your worst document (bad), while ignoring the decent documents in the middle.

**The Solution: Reordering (The "How")**
To mitigate this, we can intercept the retrieved documents and reorder them before sending them to the LLM. A common technique is to alternate placing the most relevant documents at the beginning and the end of the context window.

Example reordering of 5 documents (1 is most relevant, 5 is least):
*Original:* `[1, 2, 3, 4, 5]`
*Reordered:* `[1, 3, 5, 4, 2]`

This ensures the most crucial information is placed at the extremities where the LLM is most likely to "see" it.""")

    # Code Implementation: Setup and Mock Documents
    add_markdown("## 1. Setup & Baseline Retrieval (Mock)\n\nLet's set up a mock scenario where we have retrieved 10 documents, sorted by relevance (1 being the most relevant, 10 being the least).")

    add_code("""from typing import List
from langchain_core.documents import Document

# Simulate documents returned from a Vector DB, sorted by descending relevance.
# doc_1 is the most relevant, doc_10 is the least relevant.
retrieved_docs: List[Document] = [
    Document(page_content=f"Doc {i}", metadata={"relevance_rank": i})
    for i in range(1, 11)
]

print("Original Retrieval Order (Sorted by Relevance):")
for doc in retrieved_docs:
    print(f"- {doc.page_content}")""")

    # Code Implementation: The Reordering Algorithm
    add_markdown("## 2. Implementing `LongContextReorder`\n\nInstead of relying on third-party integrations that might be unstable, we will write a clean, robust Python function to handle the alternating reordering logic.")

    add_code("""def reorder_documents(documents: List[Document]) -> List[Document]:
    \"\"\"
    Reorders documents to mitigate the 'Lost in the Middle' effect.
    Places the most relevant documents at the beginning and end of the list.

    Args:
        documents: A list of Documents, assumed to be sorted by relevance
                   (most relevant first).

    Returns:
        A list of reordered Documents.
    \"\"\"
    if not documents:
        return []

    # Create a copy to avoid mutating the original list
    docs = list(documents)

    # Reverse the list so the least relevant are at the beginning.
    # Original: [1, 2, 3, 4, 5] -> Reversed: [5, 4, 3, 2, 1]
    docs.reverse()

    reordered_docs: List[Document] = []

    # Alternate taking from the front and back of the reversed list
    # For [5, 4, 3, 2, 1]:
    # Iter 1 (i=0, doc=5): append -> [5]
    # Iter 2 (i=1, doc=4): insert(0) -> [4, 5]
    # Iter 3 (i=2, doc=3): append -> [4, 5, 3]
    # Iter 4 (i=3, doc=2): insert(0) -> [2, 4, 5, 3]
    # Iter 5 (i=4, doc=1): append -> [2, 4, 5, 3, 1]

    for i, doc in enumerate(docs):
        if i % 2 == 1:
            # Odd index in reversed list -> Insert at the beginning
            reordered_docs.insert(0, doc)
        else:
            # Even index in reversed list -> Append at the end
            reordered_docs.append(doc)

    return reordered_docs

reordered = reorder_documents(retrieved_docs)

print("\\nReordered Document Order (Best docs at edges):")
for doc in reordered:
    print(f"- {doc.page_content}")""")

    # Common Pitfalls
    add_markdown("""## Common Pitfalls in Production

1. **Blindly applying reordering:** If you only retrieve 2-3 short documents, the context window is small enough that "Lost in the Middle" doesn't occur. Reordering here just adds overhead. Only apply this when context lengths exceed a certain threshold (e.g., >3000 tokens or >5 documents).
2. **Ignoring Document Length:** The reordering algorithm assumes documents are roughly the same length. If `Doc 1` is 50 tokens and `Doc 2` is 4000 tokens, placing `Doc 2` at the end might push `Doc 1` out of the critical "start" zone entirely. You must chunk documents uniformly.
3. **Over-Retrieval:** Trying to solve bad retrieval by retrieving *more* documents (e.g., top 50) and then reordering them often degrades performance regardless of reordering. Always optimize base retrieval quality first.
4. **Not measuring the impact:** You must audit RAG performance (e.g., using Ragas or LLM-as-a-judge) before and after implementing reordering to ensure it actually improves your specific task.""")

    # Practical Lab
    add_markdown("""## Practical Lab: Integrating Reordering with LCEL

**Your Task:**
You need to integrate our `reorder_documents` function into a LangChain Expression Language (LCEL) chain.

1. Create a `RunnableLambda` out of the `reorder_documents` function.
2. Build a simple LCEL chain that takes a list of raw dictionaries representing documents, converts them to LangChain `Document` objects, and then passes them through your reordering runnable.
3. Execute the chain with the provided test data and print the results.""")

    # Lab Implementation
    add_code("""from typing import Dict, Any
from langchain_core.runnables import RunnableLambda

# Test Data
raw_data: List[Dict[str, Any]] = [
    {"text": "Most important info", "score": 0.95},
    {"text": "Very important info", "score": 0.85},
    {"text": "Important info", "score": 0.75},
    {"text": "Somewhat important info", "score": 0.65},
    {"text": "Slightly important info", "score": 0.55},
    {"text": "Least important info", "score": 0.45},
]

def dicts_to_docs(data: List[Dict[str, Any]]) -> List[Document]:
    \"\"\"Converts raw dictionaries into LangChain Documents.\"\"\"
    # Assumes data is already sorted by score descending
    return [Document(page_content=d["text"], metadata={"score": d["score"]}) for d in data]

# 1. Wrap the functions in RunnableLambda
convert_runnable = RunnableLambda(dicts_to_docs)
reorder_runnable = RunnableLambda(reorder_documents)

# 2. Build the LCEL Chain using the pipe operator (|)
processing_chain = convert_runnable | reorder_runnable

# 3. Execute the chain
final_docs = processing_chain.invoke(raw_data)

print("Final Chain Output:")
for i, doc in enumerate(final_docs):
    print(f"{i+1}. {doc.page_content} (Score: {doc.metadata['score']})")""")

    # Ensure target directory exists
    os.makedirs('modules', exist_ok=True)
    os.makedirs('scripts', exist_ok=True)

    # Write notebook to file
    with open('modules/day_35.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)

if __name__ == "__main__":
    create_notebook()
    print("Notebook modules/day_35.ipynb generated successfully.")
