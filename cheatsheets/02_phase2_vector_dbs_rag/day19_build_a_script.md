# Day 19: RAG Context Overflow & Sliding Window

## Concise Summaries
When passing retrieved documents to an LLM, the combined length can exceed the model's context window, causing truncation or errors. A sliding window mechanism solves "Context Overflow" by iteratively sending fixed-size chunks of context, often carrying over summary state or combining responses, to ensure the LLM processes the full body of retrieved information without exceeding token limits.

## Code Snippets
```python
import logging
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SlidingWindowRAG:
    """
    Implements a basic sliding window approach to handle context overflow
    when processing large numbers of retrieved documents.
    """
    def __init__(self, llm_model: str = "gpt-3.5-turbo", max_docs_per_window: int = 3):
        # We enforce a maximum number of documents per window to prevent token overflow.
        self.max_docs_per_window = max_docs_per_window
        # Fallback mechanism: we use a reliable default model and can handle failure gracefully.
        self.llm = ChatOpenAI(model=llm_model, temperature=0)

    def process_documents(self, query: str, documents: List[Document]) -> str:
        """
        Processes a list of documents in chunks (sliding window) to generate a comprehensive answer.
        """
        if not documents:
            return "No documents provided to answer the query."

        # PII Protection: in a real system, ensure PII is stripped from documents before processing.
        # This example assumes documents are already sanitized.

        combined_answers = []

        # Iterate over documents in chunks (windows)
        for i in range(0, len(documents), self.max_docs_per_window):
            window_docs = documents[i:i + self.max_docs_per_window]
            context_text = "\n\n".join([doc.page_content for doc in window_docs])

            logger.info(f"Processing window {i//self.max_docs_per_window + 1} with {len(window_docs)} documents.")

            system_prompt = SystemMessage(
                content="You are a helpful assistant. Answer the query based ONLY on the provided context."
            )
            human_prompt = HumanMessage(
                content=f"Query: {query}\n\nContext:\n{context_text}"
            )

            try:
                response = self.llm.invoke([system_prompt, human_prompt])
                combined_answers.append(str(response.content))
            except Exception as e:
                logger.error(f"LLM processing failed for window {i//self.max_docs_per_window + 1}: {e}")
                # Graceful fallback: record the error but continue processing other windows
                combined_answers.append(f"[Error processing context window: {e}]")

        # Synthesize final answer if multiple windows were processed
        if len(combined_answers) > 1:
            return self._synthesize_answers(query, combined_answers)
        elif len(combined_answers) == 1:
            return combined_answers[0]
        else:
             return "Failed to generate an answer."

    def _synthesize_answers(self, query: str, answers: List[str]) -> str:
        """
        Combines the answers from multiple sliding windows into a single coherent response.
        """
        logger.info("Synthesizing final answer from multiple windows.")
        combined_text = "\n\n".join([f"Partial Answer {idx+1}: {ans}" for idx, ans in enumerate(answers)])

        system_prompt = SystemMessage(
            content="You are a master synthesizer. Combine the partial answers into a single, cohesive response to the original query."
        )
        human_prompt = HumanMessage(
            content=f"Original Query: {query}\n\nPartial Answers:\n{combined_text}"
        )

        try:
            final_response = self.llm.invoke([system_prompt, human_prompt])
            return str(final_response.content)
        except Exception as e:
            logger.error(f"Failed to synthesize final answer: {e}")
            # Fallback: just return the concatenated partial answers
            return "Error synthesizing final answer. Returning combined partial answers:\n\n" + "\n".join(answers)

# Example Usage
if __name__ == "__main__":
    # Create dummy documents
    docs = [
        Document(page_content="Fact 1: The capital of France is Paris."),
        Document(page_content="Fact 2: The capital of Germany is Berlin."),
        Document(page_content="Fact 3: The capital of Italy is Rome."),
        Document(page_content="Fact 4: The capital of Spain is Madrid."),
        Document(page_content="Fact 5: The capital of Portugal is Lisbon.")
    ]

    # Initialize processor with a small window size to force multiple windows
    processor = SlidingWindowRAG(max_docs_per_window=2)

    # Process query
    final_result = processor.process_documents("What are the capital cities mentioned in the documents?", docs)
    print("\nFinal Result:\n", final_result)
```

## Key Concepts
*   **Context Overflow**: Occurs when the total tokens in the prompt (system message + retrieved context + query) exceed the LLM's maximum context length, leading to token limit errors (`context_length_exceeded`).
*   **Sliding Window**: A chunking technique where data is processed in overlapping or non-overlapping sequential blocks (windows). In RAG, this means passing subsets of retrieved documents to the LLM sequentially.
*   **Map-Reduce / Refine**: LangChain concepts related to sliding windows. "Map" processes chunks individually, "Reduce" combines them. "Refine" iteratively updates an answer by passing the previous answer + new context.
*   **PII Protection (Security)**: When chunking and sending context to external LLM APIs, ensure PII is masked or redacted *before* it enters the prompt, regardless of which sliding window chunk it lands in.
*   **Graceful Fallbacks (Security/Reliability)**: Network issues or API rate limits can cause individual chunk processing to fail. Catch exceptions per chunk and continue, or fallback to a smaller model, rather than failing the entire process.

## Common Gotchas
1.  **Context Fragmentation (Loss of Meaning)**: Simply splitting text into windows can cut important context in half (e.g., splitting a sentence or a related paragraph). Consider using text splitters with overlap (e.g., `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`) before feeding into the windowing logic.
2.  **"Needle in a Haystack" Dilution**: When summarizing multiple partial answers (Map-Reduce), the model might lose the specific detail (the "needle") in the final synthesis step. A pure "Refine" chain can sometimes mitigate this better than map-reduce, though it is slower and sequential.

## Reference Links
*   [LangChain Summarization Strategies (Map-Reduce, Refine)](https://python.langchain.com/docs/use_cases/summarization/)
*   [OpenAI: Managing Context Window](https://platform.openai.com/docs/guides/text-generation/managing-tokens)
