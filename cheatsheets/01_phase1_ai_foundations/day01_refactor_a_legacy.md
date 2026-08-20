# Day 1: Refactoring Legacy Python with Pydantic and Advanced Functions

## Concise Summary
Modern Python relies heavily on type safety and data validation. Refactoring legacy scripts involves transitioning from loosely-typed dictionaries or raw arguments to **Pydantic models** for strict validation, and leveraging advanced function features like `**kwargs` unpacking, decorators, and type hints for robust, maintainable code.

---

## Code Snippet: Legacy vs. Refactored

### Legacy Code (Error-Prone)
```python
def process_user(data):
    # No validation, runtime errors waiting to happen
    user_id = int(data.get("id", 0))
    username = data.get("name", "Unknown")
    is_active = bool(data.get("active", False))
    return f"User {username} (ID: {user_id}) is {'active' if is_active else 'inactive'}."

# Hard to know what `data` should contain
result = process_user({"id": "123", "name": "Alice", "active": True})
```

### Refactored Code (Production-Ready)
```python
from pydantic import BaseModel, ValidationError, Field

# 1. Define strict data schemas
class User(BaseModel):
    id: int = Field(..., gt=0, description="Unique user identifier")
    name: str = Field(..., min_length=2, description="User's full name")
    is_active: bool = Field(default=False, alias="active")

# 2. Strict type hints in function signatures
def process_user(user: User) -> str:
    """Processes a validated User model."""
    status = 'active' if user.is_active else 'inactive'
    return f"User {user.name} (ID: {user.id}) is {status}."

def main():
    raw_data = {"id": "123", "name": "Alice", "active": True}

    try:
        # 3. Automatic validation and type coercion
        validated_user = User.model_validate(raw_data)

        # 4. Process safe data
        result = process_user(validated_user)
        print(result)
    except ValidationError as e:
        print(f"Data validation failed: {e.errors()}")

if __name__ == "__main__":
    main()
```

---

## Key Concepts
*   **`BaseModel`:** The core class in Pydantic used to define data schemas.
*   **Type Hinting:** Using standard Python types (`int`, `str`, `List`, `Dict`) to declare expected data types.
*   **`Field(...)`:** Used to define extra validation constraints (e.g., `gt=0`, `min_length=2`) or metadata.
*   **`model_validate()`:** Pydantic v2 method to parse and validate data from a dictionary.
*   **Coercion:** Pydantic automatically converts types where safe (e.g., the string `"123"` to the integer `123`).

---

## Common Gotchas
1.  **Silent Coercion Masking Bugs:** Pydantic will aggressively cast types (like converting `"False"` to boolean `True` in older versions or specific setups). If you need strict types without casting, use `StrictInt`, `StrictStr`, etc.
2.  **Using `.dict()` instead of `.model_dump()`:** In Pydantic V2, methods like `.dict()` and `.parse_obj()` are deprecated. Always use `.model_dump()` and `.model_validate()` for dictionary operations to avoid future breakages.

---

## Reference Links
*   [Pydantic Official Documentation](https://docs.pydantic.dev/latest/)
*   [Python Type Hints (PEP 484)](https://peps.python.org/pep-0484/)

---

## Advanced Pattern: OOP & AI Security Integration

### Concise Summary
When integrating AI (like LLMs) in production, you must adopt strict Object-Oriented principles and apply core AI security measures like **PII (Personally Identifiable Information) Redaction** and **Fallback Mechanisms**. Coupling Pydantic with robust class design ensures you validate inputs, secure sensitive data before it reaches external APIs, and handle failures gracefully.

---

### Code Snippet: Pydantic + Security Fallbacks

```python
from pydantic import BaseModel, ValidationError, Field, field_validator
import re
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Define Strict Input Schema with Built-in Security Logic
class SecurePromptRequest(BaseModel):
    user_id: int = Field(..., gt=0, description="Unique user ID")
    raw_prompt: str = Field(..., min_length=5, description="The raw prompt from the user")

    @field_validator("raw_prompt")
    @classmethod
    def redact_pii(cls, value: str) -> str:
        """
        AI Security: Redact potential PII (e.g., Email addresses)
        before it is processed or sent to an LLM.
        """
        email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        if re.search(email_pattern, value):
            logger.warning("PII detected and redacted in raw_prompt.")
        return re.sub(email_pattern, "[REDACTED_EMAIL]", value)

# 2. OOP Design for the Service layer
class PromptProcessor:
    """Handles prompt processing with fallback mechanisms."""

    def __init__(self, fallback_response: str = "I'm sorry, but I cannot process that request right now."):
        self.fallback_response = fallback_response

    def process(self, data: dict) -> str:
        try:
            # Validate and sanitize input via Pydantic
            validated_req = SecurePromptRequest.model_validate(data)

            # Simulate processing (e.g., LLM call)
            # In a real scenario, you'd call the LLM here.
            logger.info(f"Processing secure prompt for user {validated_req.user_id}: {validated_req.raw_prompt}")
            return f"Processed safely: {validated_req.raw_prompt}"

        except ValidationError as e:
            logger.error(f"Validation Error: {e.errors()}")
            # AI Security: Graceful Fallback instead of crashing or leaking raw errors
            return self.fallback_response
        except Exception as e:
            logger.error(f"Unexpected System Error: {e}")
            return self.fallback_response

def main():
    processor = PromptProcessor()

    # Unsafe input containing PII and invalid fields
    unsafe_data = {
        "user_id": 99,
        "raw_prompt": "Hello, my email is alice@example.com, how do I reset my password?"
    }

    print("--- Attempting processing ---")
    result = processor.process(unsafe_data)
    print(f"Result: {result}")

    print("\n--- Attempting invalid processing ---")
    invalid_data = {"user_id": -5, "raw_prompt": "Hi"}
    result_fail = processor.process(invalid_data)
    print(f"Result: {result_fail}")

if __name__ == "__main__":
    main()
```

---

### Key Concepts (AI Security & OOP)
*   **PII Redaction:** Stripping or obfuscating sensitive data (emails, SSNs) from payloads *before* they are sent to third-party LLM providers.
*   **Graceful Fallbacks:** Ensuring your application returns safe, generic errors to end-users instead of exposing internal stack traces or raw API errors that could be leveraged by attackers.
*   **`@field_validator`:** A Pydantic decorator used to run custom validation logic (like regex replacements for redaction) during model instantiation.
*   **Encapsulation:** Using classes (like `PromptProcessor`) to bundle configuration (e.g., fallbacks) with the logic that operates on the Pydantic models.

---

### Common Gotchas
1.  **Leaking PII via Logging:** Be careful not to log the raw, unvalidated input *before* Pydantic processes and sanitizes it. Always log the validated model's state.
2.  **Over-relying on Regex for Security:** Regex-based PII redaction is a good first step, but it is not foolproof. Attackers can obfuscate PII to bypass regex. Use specialized libraries (like Presidio) for production-grade PII detection.

---

### Reference Links
*   [Pydantic Validation (Validators)](https://docs.pydantic.dev/latest/concepts/validators/)
*   [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
