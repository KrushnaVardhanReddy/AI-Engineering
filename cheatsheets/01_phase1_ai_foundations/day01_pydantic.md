# Pydantic Cheatsheet

## Day 1: The Basics

### 1. What is Pydantic?
Pydantic is a data validation and settings management library for Python using Python type hints. It enforces type hints at runtime, and provides user-friendly errors when data is invalid.

### 2. Basic Model Definition
To define a schema or a model, inherit from `BaseModel`.

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    is_active: bool
```

### 3. Creating Instances
Pass data as keyword arguments to create an instance.

```python
# Valid data
user = User(id=1, name="Alice", is_active=True)
print(user.name)  # Alice

# Data is cast to the correct type if possible
user2 = User(id="2", name="Bob", is_active="False") 
# id becomes int(2), is_active becomes False
```

### 4. Default Values
Fields with default values are optional when instantiating the model.

```python
from pydantic import BaseModel
from typing import Optional

class User(BaseModel):
    id: int
    name: str
    email: Optional[str] = None  # Optional field
    is_active: bool = True       # Default is True
    
# Can just provide id and name
user = User(id=1, name="Alice") 
```

### 5. Dictionary & JSON Operations
Converting models to dictionaries or JSON is straightforward.

```python
# Model to Dictionary
user_dict = user.model_dump()
# Or user.dict() in Pydantic v1

# Model to JSON string
user_json = user.model_dump_json()
# Or user.json() in Pydantic v1

# Creating Model from Dictionary
data = {"id": 1, "name": "Alice"}
user = User.model_validate(data) 
# Or User(**data) or User.parse_obj(data) in v1
```

---
*Note: We will append future days (Day 2, Day 3, etc.) below this section.*
