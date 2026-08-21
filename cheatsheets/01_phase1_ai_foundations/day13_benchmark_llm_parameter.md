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

## Advanced Implementation: OOP & Security Best Practices
This section demonstrates how to wrap parameter benchmarking in a robust class structure. This ensures proper separation of concerns, secure configuration loading (e.g., handling missing API keys without exposing secrets), and comprehensive error handling during batch processing.

```python
import os
import logging
import warnings
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress the ChatOpenAI deprecation warning about passing parameters via model_kwargs
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_openai.chat_models.base")

class BenchmarkConfig(BaseModel):
    """Pydantic model for benchmark configuration parameters."""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)

class ParameterBenchmarker:
    """Class to manage and execute LLM parameter benchmarks."""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self._ensure_api_key()

    def _ensure_api_key(self) -> None:
        """Securely check for the API key without exposing it."""
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY environment variable not found. Using dummy key for testing.")
            os.environ["OPENAI_API_KEY"] = "sk-dummy"

    def run_benchmark(self, prompt: str, configs: List[BenchmarkConfig], iterations_per_config: int = 10) -> List[Dict[str, Any]]:
        """Run the provided prompt against multiple parameter configurations, multiple times to observe variance."""
        results = []
        logger.info(f"Starting benchmark for prompt: '{prompt}' with {len(configs)} configurations, {iterations_per_config} iterations each.")

        for config_idx, config in enumerate(configs, 1):
            logger.info(f"--- Testing Configuration {config_idx}/{len(configs)} ---")
            logger.info(f"Config: {config.model_dump_json()}")

            try:
                # Initialize LLM with the specific configuration
                llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    frequency_penalty=config.frequency_penalty,
                    max_retries=0 # Fail fast for benchmarking
                )

                for iter_num in range(1, iterations_per_config + 1):
                    try:
                        logger.info(f"  -> Running iteration {iter_num}/{iterations_per_config}")
                        # Execute the prompt
                        response = llm.invoke([HumanMessage(content=prompt)])
                        result_text = response.content.strip()

                        results.append({
                            "config_id": config_idx,
                            "iteration": iter_num,
                            "config": config.model_dump(),
                            "response": result_text,
                            "status": "success"
                        })
                    except Exception as loop_e:
                         error_msg = f"  -> API Call Failed on iteration {iter_num}: {type(loop_e).__name__} - {str(loop_e)}"
                         logger.error(error_msg)
                         results.append({
                            "config_id": config_idx,
                            "iteration": iter_num,
                            "config": config.model_dump(),
                            "response": None,
                            "status": "failed",
                            "error": type(loop_e).__name__
                        })

            except Exception as e:
                # Graceful fallback mechanisms - critical for production reliability
                error_msg = f"Failed to initialize LLM or fatal error: {type(e).__name__} - {str(e)}"
                logger.error(error_msg)

        return results

if __name__ == "__main__":
    # Define a set of robust configurations to test variance
    test_configs = [
        BenchmarkConfig(temperature=0.2, top_p=1.0, frequency_penalty=0.0), # Low randomness
        BenchmarkConfig(temperature=0.8, top_p=1.0, frequency_penalty=0.0), # Standard randomness
        BenchmarkConfig(temperature=0.8, top_p=0.5, frequency_penalty=0.0), # Standard randomness + nucleus sampling
        BenchmarkConfig(temperature=1.5, top_p=1.0, frequency_penalty=1.0), # High randomness + repetition penalty
    ]

    benchmarker = ParameterBenchmarker()
    test_prompt = "Explain quantum computing in one simple sentence."

    # Execute the benchmarking process with 10 iterations per config to observe variance
    benchmark_results = benchmarker.run_benchmark(prompt=test_prompt, configs=test_configs, iterations_per_config=10)

    # Summary of results
    print("\n--- Benchmark Execution Summary ---")
    for res in benchmark_results:
        print(f"Config {res['config_id']}, Iteration {res['iteration']} ({res['status']}): Temp={res['config']['temperature']}, Top-P={res['config']['top_p']}, Freq Penalty={res['config']['frequency_penalty']}")
```
