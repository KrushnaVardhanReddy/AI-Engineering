# Day 3: Build a Class-Based Python Wrapper for an LLM API

## Concise Summaries
Wrapping an LLM API within a Python class encapsulates configuration, execution logic, and error handling. Implementing robust retry logic and logging at this level prevents transient network issues from cascading into application failures, ensuring reliable inference in production systems.

## Code Snippets

```python
import json
import logging
import time
import urllib.request
from urllib.error import HTTPError, URLError
from typing import Optional, Dict, Any

# Configure standard logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LLMClientWrapper:
    """
    A production-ready Python wrapper for an LLM API with built-in retries and logging.
    """
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1/chat/completions", max_retries: int = 3, backoff_factor: float = 2.0):
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def generate_text(self, prompt: str, model: str = "gpt-3.5-turbo") -> Optional[str]:
        """
        Sends a prompt to the LLM API and returns the generated text, with retries on failure.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.base_url, data=data, headers=headers, method="POST")

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Attempt {attempt}: Sending request to LLM API.")
                with urllib.request.urlopen(req, timeout=10) as response:
                    response_data = json.loads(response.read().decode("utf-8"))
                    logger.info("Request successful.")
                    return response_data["choices"][0]["message"]["content"]

            except HTTPError as e:
                logger.error(f"HTTP Error {e.code}: {e.reason}")
                if e.code in (401, 403, 404):
                    logger.error("Client error - not retrying.")
                    break
            except URLError as e:
                logger.error(f"URL Error: {e.reason}")
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")

            if attempt < self.max_retries:
                sleep_time = self.backoff_factor ** attempt
                logger.warning(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error("Max retries reached. Request failed.")

        return None

# --- Example Usage ---
if __name__ == "__main__":
    # Ensure this is replaced with a real API key in production
    client = LLMClientWrapper(api_key="sk-dummy-key", max_retries=1)

    # Intentionally trigger an error for demonstration by using a mock/dummy key and small timeout
    print("Running LLMClientWrapper test...")
    response = client.generate_text("Explain quantum computing in one sentence.")
    if response:
        print(f"Response: {response}")
    else:
        print("Failed to get a response (expected with dummy key).")
```

## Key Concepts
- **Encapsulation:** Hiding the internal state and functionality of an object (API configuration and request logic) inside a class.
- **Exponential Backoff:** Increasing the wait time between retry attempts exponentially to avoid overwhelming the server.
- **Idempotency:** A property of operations where making multiple identical requests has the same effect as making a single request (crucial for safe retries).
- **Transient Errors:** Temporary issues, such as rate limits or network timeouts, that can often be resolved simply by retrying the operation.

## Common Gotchas
- **Retrying Client Errors (4xx):** Blindly retrying `401 Unauthorized` or `404 Not Found` wastes resources. Only retry server errors (5xx) or rate limits (429).
- **Missing Timeouts:** Failing to specify a timeout in the network request (e.g., in `urlopen`) can cause the application to hang indefinitely if the API server becomes unresponsive.

## Reference Links
- [Python `urllib.request` Documentation](https://docs.python.org/3/library/urllib.request.html)
- [Python `logging` HOWTO](https://docs.python.org/3/howto/logging.html)

## Advanced Concepts: State Management & AI Security
To build production-ready agentic systems, wrappers must also handle conversation state, protect sensitive data (PII), and provide reliable fallbacks when all retries fail.

## Advanced Code Snippets

```python
import json
import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

class SecureConversationalLLM:
    """
    Advanced wrapper demonstrating State Management, PII scrubbing, and Fallback mechanisms.
    Builds upon the LLMClientWrapper to maintain robust network handling.
    """
    def __init__(self, client_wrapper: LLMClientWrapper, fallback_message: str = "Service temporarily unavailable. Please try again later."):
        self.client = client_wrapper
        self.fallback_message = fallback_message
        self.history: List[Dict[str, str]] = []

    def _scrub_pii(self, text: str) -> str:
        """
        Replaces simple email patterns to protect PII before sending to the LLM.
        """
        return re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED_EMAIL]", text)

    def chat(self, user_input: str) -> str:
        """
        Manages conversation state, scrubs PII, and utilizes fallbacks if the API is unreachable.
        """
        safe_input = self._scrub_pii(user_input)
        self.history.append({"role": "user", "content": safe_input})

        # Construct a simple unified prompt from history for the base client
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history])

        response = self.client.generate_text(prompt)

        if response is None:
            logger.warning("LLM call failed after all retries. Returning fallback message.")
            return self.fallback_message

        self.history.append({"role": "assistant", "content": response})
        return response

# --- Advanced Example Usage ---
if __name__ == "__main__":
    print("\nRunning SecureConversationalLLM test...")
    # client is already defined in the previous snippet
    secure_bot = SecureConversationalLLM(client_wrapper=client)

    # Test PII scrubbing and fallback (since dummy key fails)
    safe_response = secure_bot.chat("My email is test@example.com. Explain quantum computing.")
    print(f"Chat Response: {safe_response}")
    print(f"Current History: {json.dumps(secure_bot.history, indent=2)}")
```

## Additional Key Concepts
- **State Management:** Storing the sequence of interactions (e.g., `self.history`) so the LLM has contextual awareness of the conversation.
- **PII Protection:** The practice of intercepting and masking Personally Identifiable Information (like emails or SSNs) at the application layer to avoid leaking sensitive data to external models.
- **Fallback Mechanisms:** Returning a predefined, safe response when the primary service is entirely unreachable, ensuring a degraded but stable user experience.
