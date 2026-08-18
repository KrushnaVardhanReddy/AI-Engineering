import json
import os

def create_notebook():
    notebook_dict = {
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_markdown(source):
        notebook_dict["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source.splitlines(True)
        })

    def add_code(source):
        notebook_dict["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(True)
        })

    # Generate Dummy Files
    dummy_files_code = """import os

# Create dummy TXT
with open("test.txt", "w") as f:
    f.write("This is a standard text document.\\nIt contains plain text.")

# Create dummy MD
with open("test.md", "w") as f:
    f.write("# Markdown Document\\n\\nThis is a *markdown* file with structured text.")

# Create dummy PDF (minimal valid binary PDF)
minimal_pdf = b"%PDF-1.0\\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\\nxref\\n0 4\\n0000000000 65535 f\\n0000000010 00000 n\\n0000000053 00000 n\\n0000000102 00000 n\\ntrailer<</Size 4/Root 1 0 R>>\\nstartxref\\n149\\n%EOF"
with open("test.pdf", "wb") as f:
    f.write(minimal_pdf)
print("Dummy files test.txt, test.md, and test.pdf generated successfully.")
"""
    add_code(dummy_files_code)

    # Core Theory
    theory_md = """# Day 27: Refactoring the LangChain Pipeline for Multiple Document Types

Welcome to Day 27! Today we elevate our RAG (Retrieval-Augmented Generation) pipeline from a simple text processor to a robust, production-ready document ingestion engine capable of handling varied real-world data sources (PDFs, Markdown, and TXT files).

## The "Why" and "How"

**The Problem:**
Most production systems do not ingest nicely formatted text strings. Enterprise knowledge bases contain unstructured PDFs, documentation in Markdown, and raw log files in TXT. Hardcoding a single loader (like `TextLoader`) breaks the moment a user uploads a different file type.

**The Solution:**
We need to implement a **Factory Pattern** or a **routing mechanism** within our pipeline.
1. **Dynamic Selection:** We inspect the file extension and dynamically instantiate the correct LangChain DocumentLoader.
2. **Unified Output:** Regardless of the underlying loader (`PyPDFLoader`, `UnstructuredMarkdownLoader`, `TextLoader`), the output must always be a standardized list of LangChain `Document` objects containing `page_content` and `metadata`. This ensures the downstream pipeline (chunking, embedding, vector DB ingestion) remains completely agnostic to the source format."""
    add_markdown(theory_md)

    # Code Implementation
    impl_md = """## Code Implementation: Multi-Document Pipeline

Here we implement a `DocumentIngestor` class with strict type hinting and docstrings. It dynamically selects the appropriate loader."""
    add_markdown(impl_md)

    impl_code = """import os
from typing import List, Dict, Type, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.document_loaders.base import BaseLoader

class DocumentIngestor:
    \"\"\"
    A robust pipeline component for ingesting multiple document types into LangChain Documents.
    \"\"\"

    def __init__(self) -> None:
        \"\"\"Initializes the ingestor with a mapping of extensions to LangChain loaders.\"\"\"
        self.loader_mapping: Dict[str, Type[BaseLoader]] = {
            ".txt": TextLoader,
            ".md": TextLoader,  # Using TextLoader for simple MD parsing.
            ".pdf": PyPDFLoader
        }

    def load_document(self, file_path: str) -> List[Document]:
        \"\"\"
        Dynamically selects the appropriate loader based on the file extension and loads the document.

        Args:
            file_path (str): The absolute or relative path to the file.

        Returns:
            List[Document]: A list of LangChain Document objects.

        Raises:
            ValueError: If the file extension is not supported or file does not exist.
        \"\"\"
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        loader_cls = self.loader_mapping.get(ext)
        if not loader_cls:
            raise ValueError(f"Unsupported file extension: {ext}. Supported types: {list(self.loader_mapping.keys())}")

        print(f"Ingesting {file_path} using {loader_cls.__name__}...")
        try:
            loader = loader_cls(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

# Example Usage
if __name__ == "__main__":
    ingestor = DocumentIngestor()

    docs_to_process = ["test.txt", "test.md", "test.pdf"]
    all_documents: List[Document] = []

    for doc_path in docs_to_process:
        try:
            docs = ingestor.load_document(doc_path)
            all_documents.extend(docs)
            print(f"Successfully loaded {len(docs)} chunk(s) from {doc_path}.\\n")
        except ValueError as e:
            print(f"Skipping: {e}\\n")

    print(f"Total documents loaded: {len(all_documents)}")
"""
    add_code(impl_code)

    # Common Pitfalls
    pitfalls_md = """## Common Pitfalls in Production

1. **Ignoring Metadata:** Different loaders extract different metadata. For example, `PyPDFLoader` includes the `page` number, while `TextLoader` does not. Downstream components (like Qdrant payload filters) will crash if they expect a `page` key that doesn't exist. Always normalize metadata after loading.
2. **Memory Overload with Large Files:** Calling `.load()` on a 10,000-page PDF loads the entire document into memory. For large files, use `.lazy_load()` which returns an iterator, preventing OOM (Out Of Memory) exceptions.
3. **Missing System Dependencies:** Loaders like `PyPDFLoader` (requires `pypdf`) rely on underlying system libraries. Forgetting to include these in your `requirements.txt` or Dockerfile is a classic deployment failure."""
    add_markdown(pitfalls_md)

    # Practical Lab
    lab_md = """## Practical Lab / Homework

**Your Task:**
1. Create a function `process_directory(directory_path: str, ingestor: DocumentIngestor) -> List[Document]`.
2. The function should iterate through all files in the given directory.
3. It should attempt to load each file using the `ingestor` instance.
4. It must catch `ValueError` for unsupported files gracefully (print a warning) and continue processing.
5. Return the combined list of all successfully loaded `Document` objects.

*Test it by pointing it to the current directory (`"."`) to pick up the `test.txt`, `test.md`, and `test.pdf` files.*"""
    add_markdown(lab_md)

    lab_code = """# Lab Implementation
def process_directory(directory_path: str, ingestor: DocumentIngestor) -> List[Document]:
    \"\"\"
    Scans a directory and ingests all supported documents.
    \"\"\"
    combined_docs: List[Document] = []

    print(f"Scanning directory: {directory_path}")
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)

        # Skip directories
        if os.path.isdir(filepath):
            continue

        try:
            docs = ingestor.load_document(filepath)
            combined_docs.extend(docs)
        except ValueError as e:
            print(f"Warning: Could not process {filename} - {e}")

    return combined_docs

# Run the Lab
final_docs = process_directory(".", ingestor)
print(f"\\nLab Complete! Total documents ingested from directory: {len(final_docs)}")
"""
    add_code(lab_code)

    os.makedirs('modules', exist_ok=True)
    with open('modules/day_27.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook_dict, f, indent=2)

if __name__ == "__main__":
    create_notebook()
