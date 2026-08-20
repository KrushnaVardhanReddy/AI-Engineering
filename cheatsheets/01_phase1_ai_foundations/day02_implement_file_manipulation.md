# Day 2: Large-Scale JSONL Data Processing

## Summary
JSON Lines (JSONL) is a text format where each line is a valid JSON object. It is ideal for processing large-scale AI datasets incrementally, as it allows streaming reading and writing without loading the entire dataset into memory.

## Code Snippets

### Reading and Processing JSONL in Chunks
Use generators to read large files line-by-line to minimize memory consumption.

```python
import json
from typing import Iterator, Dict, Any

def stream_jsonl(filepath: str) -> Iterator[Dict[str, Any]]:
    """Yields parsed JSON objects from a JSONL file one line at a time."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def process_dataset(filepath: str):
    """Processes large JSONL dataset efficiently."""
    for record in stream_jsonl(filepath):
        # Process each record individually
        print(record.get('text', ''))
```

### Writing to JSONL
Append to JSONL files line-by-line to avoid out-of-memory errors.

```python
import json
from typing import List, Dict, Any

def append_to_jsonl(filepath: str, records: List[Dict[str, Any]]):
    """Appends a list of records to a JSONL file."""
    with open(filepath, 'a', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')

# Example usage
data = [{"prompt": "Hello", "completion": "World"}, {"prompt": "AI", "completion": "Engineering"}]
append_to_jsonl("dataset.jsonl", data)
```

## Key Concepts
- **JSONL (JSON Lines):** A structured data format where each line represents a single JSON object.
- **Generator (`yield`):** A Python function that returns an iterator, computing values on-the-fly rather than storing them in memory.
- **Streaming:** Processing data incrementally as it is read, rather than loading the full payload upfront.
- **File Modes:** `'r'` for reading, `'a'` for appending, `'w'` for overwriting.

## Common Gotchas
- **Loading entire file into memory:** Avoid `json.loads(f.read())` or `readlines()` with large JSONL files; it will cause memory exhaustion. Always iterate over the file object directly (`for line in f:`).
- **Missing newline characters:** When writing to a JSONL file, ensure you explicitly append `\n` after each `json.dumps()` output.

## Reference Links
- [Python `json` library documentation](https://docs.python.org/3/library/json.html)
- [JSON Lines Specification](https://jsonlines.org/)
