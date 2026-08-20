---
# FastAPI & LLM Integration Cheatsheet

## Day 5: Build a FastAPI Endpoint with LLM and Authentication

### 1. Core Concepts
*   **FastAPI:** A modern, fast web framework for building APIs with Python 3.8+ based on standard Python type hints.
*   **Decorators for Auth:** Using FastAPI's dependency injection (`Depends`) combined with security utilities to enforce authentication before endpoint execution.
*   **LLM Integration:** Handling prompt generation and executing LLM calls within a FastAPI route securely.

### 2. Basic Setup & Dependencies
Ensure you have the required libraries installed in your environment:
```bash
uv pip install fastapi uvicorn langchain-openai python-dotenv
```

### 3. Code Example: FastAPI Endpoint with LLM and Basic API Key Auth

This code provides a complete, production-ready snippet for a secure FastAPI endpoint that queries an OpenAI model.

```python
import os
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 1. Initialize FastAPI app
app = FastAPI(title="LLM API", description="Secure FastAPI LLM Integration")

# 2. Security Setup
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Dummy verification - in production, load from env or database
EXPECTED_API_KEY = os.getenv("API_KEY", "super_secret_key_123")

def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    """Dependency to validate API key."""
    if api_key_header == EXPECTED_API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )

# 3. Define Request/Response Schemas
class PromptRequest(BaseModel):
    prompt: str
    temperature: float = 0.7

class LLMResponse(BaseModel):
    reply: str

# 4. LLM Initialization Dependency
# We initialize it outside the route to reuse the connection pool,
# but can also use a dependency to fetch it if needed.
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# 5. Define Endpoint
@app.post("/generate", response_model=LLMResponse)
async def generate_text(
    request: PromptRequest,
    api_key: str = Depends(get_api_key)
) -> LLMResponse:
    """Generate text using an LLM, requires valid API key."""
    try:
        # Initialize LLM with request-specific temperature
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=request.temperature)

        # Invoke LLM
        messages = [HumanMessage(content=request.prompt)]
        response = llm.invoke(messages)

        return LLMResponse(reply=str(response.content))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM Error: {str(e)}"
        )

# Run with: uvicorn filename:app --reload
```

### 4. Key Parameters & Terms
*   `Depends(func)`: FastAPI's dependency injection system, used here to require `get_api_key` to run successfully before the route.
*   `Security(APIKeyHeader)`: Defines the location and name of the API key in the request headers (e.g., `X-API-Key`).
*   `HTTPException`: The correct way to return HTTP errors (like 401 Unauthorized or 500 Server Error) in FastAPI.
*   `response_model`: Used in the `@app.post` decorator to enforce the schema of the returned data and generate OpenAPI docs automatically.

### 5. Common Gotchas
*   **Synchronous LLM Calls Blocking the Event Loop:** If using a synchronous method (like `.invoke` without an async equivalent or offloading), it can block FastAPI's async event loop. For production with high traffic, use `ainvoke` with an async LLM or run sync code in a thread pool.
*   **Hardcoding Secrets:** Never hardcode `EXPECTED_API_KEY` or `OPENAI_API_KEY` in your script. Always use `.env` files or secure secret managers.

### 6. Reference Links
*   [FastAPI Official Documentation: Security](https://fastapi.tiangolo.com/tutorial/security/)
*   [LangChain Documentation: Chat Models](https://python.langchain.com/docs/modules/model_io/chat/)
