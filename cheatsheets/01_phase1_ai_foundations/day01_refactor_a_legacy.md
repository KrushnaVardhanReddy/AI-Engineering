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
