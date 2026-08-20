# Day 4: Measure LLM Usage & Latency with a Python Decorator

## 🚀 Core Concept
A Python decorator allows you to wrap functions (like LLM API calls) to execute code before and after the function runs. For AI engineering, decorators are essential for automatically tracking **token usage**, **cost**, and **execution latency** across multiple API calls without cluttering your core business logic.

## 💻 Production-Ready Code
Here is a complete, copy-pasteable implementation using the standard library `time` and `functools.wraps` alongside LangChain's `get_openai_callback`.

```python
import time
from functools import wraps
from typing import Callable, Any
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def track_llm_usage(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator to measure execution latency and OpenAI token usage.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()

        # Use LangChain's context manager to track token usage for this specific call
        with get_openai_callback() as cb:
            result = func(*args, **kwargs)

        end_time = time.perf_counter()
        latency = end_time - start_time

        print(f"--- [LLM Telemetry for '{func.__name__}'] ---")
        print(f"Latency       : {latency:.4f} seconds")
        print(f"Total Tokens  : {cb.total_tokens}")
        print(f"Prompt Tokens : {cb.prompt_tokens}")
        print(f"Completion    : {cb.completion_tokens}")
        print(f"Est. Cost     : ${cb.total_cost:.5f}")
        print("-----------------------------------------")

        return result
    return wrapper

# Example Usage
@track_llm_usage
def generate_summary(text: str) -> str:
    """Generates a summary of the provided text."""
    # Using a dummy API key for local execution to prevent auth errors
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key="sk-dummy-key")
    messages = [HumanMessage(content=f"Summarize this in one sentence: {text}")]

    # Try/except block to handle execution without a real API key gracefully
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Simulation result: (API call failed gracefully due to dummy key. Original text len: {len(text)})"

if __name__ == "__main__":
    # Test the decorated function
    result = generate_summary("Artificial intelligence engineering is the process of designing, building, and deploying AI systems into production environments.")
    print("\nResult:", result)
```

## 🔑 Key Concepts
*   **`@wraps(func)`:** Always use this from `functools` inside your decorators. It preserves the original function's metadata (like `__name__` and `__doc__`).
*   **`time.perf_counter()`:** The preferred method in Python for measuring short durations, as it has the highest available resolution.
*   **`*args, **kwargs`:** Allows your decorator to accept any number of positional and keyword arguments, making it flexible for wrapping different functions.
*   **Context Managers (`with`):** LangChain's `get_openai_callback()` uses this pattern to track usage specifically within that indented block.

## ⚠️ Common Gotchas
1.  **Forgetting `@wraps`:** If you omit this, debugging tools and frameworks (like FastAPI or LangChain) will see the name `wrapper` instead of your actual function name, breaking routing or logging.
2.  **Tracking Asynchronous Calls:** The standard decorator above won't work correctly for `async def` functions. You need an `async def wrapper` and `await func(*args, **kwargs)` for asynchronous endpoints.

## 🔗 Reference Links
*   [Python Official Docs: functools.wraps](https://docs.python.org/3/library/functools.html#functools.wraps)
*   [LangChain Docs: Tracking Token Usage](https://python.langchain.com/v0.1/docs/modules/model_io/llms/token_usage_tracking/)
