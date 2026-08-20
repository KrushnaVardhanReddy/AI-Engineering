# Day 13: Benchmarking LLM Parameters

## Concise Summary
LLM outputs are heavily influenced by generation parameters. **Temperature** controls randomness (higher = more creative/random), **Top-P (nucleus sampling)** limits token selection to a cumulative probability threshold, and **Frequency Penalty** reduces repetition by penalizing tokens based on their frequency in the generated text so far. Benchmarking these variations systematically reveals optimal configurations for specific use cases.

## Key Concepts
- **Temperature (0.0 - 2.0):** Scales the logits before softmax. `0.0` is deterministic (greedy), `>1.0` is highly random.
- **Top-P (0.0 - 1.0):** Nucleus sampling. `0.1` means only tokens comprising the top 10% probability mass are considered.
- **Frequency Penalty (-2.0 - 2.0):** Penalizes tokens based on how many times they've appeared in the text so far. Positive values decrease likelihood of repetition.
- **Presence Penalty (-2.0 - 2.0):** Penalizes tokens simply if they have appeared (boolean), encouraging topic changes.

## Code Snippet: Parameter Benchmarking
This script runs a prompt through multiple iterations varying Temperature, Top-P, and Frequency Penalty using LangChain's `ChatOpenAI`.

```python
import os
import warnings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Suppress the ChatOpenAI deprecation warning about passing parameters via model_kwargs
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_openai.chat_models.base")

# Mock the API key for local execution if not present
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "sk-dummy"

def benchmark_llm_parameters():
    prompt = "Write a haiku about artificial intelligence."

    # Define 10 configurations to benchmark
    configurations = [
        {"temperature": 0.0, "top_p": 1.0, "frequency_penalty": 0.0},
        {"temperature": 0.5, "top_p": 1.0, "frequency_penalty": 0.0},
        {"temperature": 1.0, "top_p": 1.0, "frequency_penalty": 0.0},
        {"temperature": 1.5, "top_p": 1.0, "frequency_penalty": 0.0},
        {"temperature": 0.7, "top_p": 0.5, "frequency_penalty": 0.0},
        {"temperature": 0.7, "top_p": 0.1, "frequency_penalty": 0.0},
        {"temperature": 0.7, "top_p": 1.0, "frequency_penalty": 1.0},
        {"temperature": 0.7, "top_p": 1.0, "frequency_penalty": 2.0},
        {"temperature": 1.2, "top_p": 0.9, "frequency_penalty": 0.5},
        {"temperature": 0.0, "top_p": 0.1, "frequency_penalty": 2.0},
    ]

    print(f"Prompt: '{prompt}'\n")

    for i, config in enumerate(configurations, 1):
        try:
            # Initialize LLM with the specific configuration
            # In Langchain > 0.1, it's recommended to pass kwargs directly to ChatOpenAI rather than model_kwargs
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=config["temperature"],
                top_p=config["top_p"],
                frequency_penalty=config["frequency_penalty"],
                max_retries=0 # Fail fast for benchmarking/dummy key
            )

            print(f"Iteration {i}: {config}")
            # Execute the prompt
            response = llm.invoke([HumanMessage(content=prompt)])
            print(f"Result: {response.content.strip()}\n")

        except Exception as e:
            # Handle API errors gracefully, especially with dummy keys
            print(f"Result: API Call Failed (Expected if using dummy key) - {type(e).__name__}\n")

if __name__ == "__main__":
    benchmark_llm_parameters()
```

## Common Gotchas
- **Altering Both Temperature and Top-P:** OpenAI generally recommends modifying *either* Temperature or Top-P, but not both simultaneously, as their interactions can be unpredictable.
- **Extreme Penalties:** Setting Frequency or Presence Penalties too high (e.g., `2.0`) can cause the model to generate nonsensical words or abruptly stop generating to avoid repeating any common words (like "the" or "and").

## Reference Links
- [OpenAI API Reference - Chat Completions (Parameters)](https://platform.openai.com/docs/api-reference/chat/create)
- [LangChain ChatOpenAI Documentation](https://python.langchain.com/v0.1/docs/integrations/chat/openai/)
