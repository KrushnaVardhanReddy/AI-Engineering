# System Prompt Versioning Cheatsheet

## Day 14: System Prompts and Versioning

### Concise Summaries
A **System Prompt** acts as the overarching instruction manual for an LLM, setting its persona, constraints, and operational context. **Versioning** these prompts is critical to treat them as configurable code artifacts rather than static text, allowing you to track changes, test variations, and ensure deterministic model behavior across application updates.

### Code Snippets

```python
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Setup the model (requires GROQ_API_KEY environment variable)
llm = ChatGroq(model="llama3-8b-8192", request_timeout=5.0)

# A simulated dictionary-based versioning system
SYSTEM_PROMPTS = {
    "v1.0": "You are a helpful assistant.",
    "v1.1": "You are a helpful assistant. Keep your answers under 50 words.",
    "v2.0": "You are an expert Python software engineer. Provide concise code snippets."
}

def get_response(user_input: str, version: str = "v1.0"):
    """Fetches a response using a specific version of the system prompt."""

    # Retrieve the specified version, fallback to v1.0 if not found
    system_instruction = SYSTEM_PROMPTS.get(version, SYSTEM_PROMPTS["v1.0"])

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "{input}")
    ])

    # Compose the LCEL chain
    chain = prompt | llm

    try:
        # Invoke the chain
        response = chain.invoke({"input": user_input})
        return response.content
    except Exception as e:
        return f"Error connecting to LLM: {str(e)}"

# Example Usage
if __name__ == "__main__":
    # In a real app, 'version' could come from an environment variable or database
    print("--- v1.0 ---")
    print(get_response("Explain binary search.", version="v1.0"))

    print("\n--- v1.1 ---")
    print(get_response("Explain binary search.", version="v1.1"))

    print("\n--- v2.0 ---")
    print(get_response("Explain binary search.", version="v2.0"))
```

### Key Concepts
*   **System Prompt / System Message:** The first message in the context window that instructs the model on *how* to behave (tone, style, constraints) rather than *what* specific question to answer.
*   **Prompt Engineering as Code:** Treating prompts like code by versioning them (e.g., v1, v2) and storing them in source control or a database.
*   **A/B Testing:** Versioning enables testing two different system prompts against the same user input to evaluate performance and quality.
*   **`ChatPromptTemplate`:** LangChain's primary class for structuring system and human messages dynamically.

### Common Gotchas
1.  **Hardcoding Prompts:** Hardcoding system prompts directly inside your execution functions makes them difficult to update and test without modifying the core logic. Extract them to dictionaries, external files, or databases.
2.  **Lack of Fallbacks:** When implementing versioning, always provide a default or fallback version in case the requested version identifier is missing or misspelled.

### Reference Links
*   [LangChain Prompts Documentation](https://python.langchain.com/docs/concepts/prompt_templates/)
*   [Groq LangChain Integration](https://python.langchain.com/docs/integrations/chat/groq/)
