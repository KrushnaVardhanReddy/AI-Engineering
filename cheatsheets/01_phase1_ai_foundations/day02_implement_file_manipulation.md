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

### Secure OOP JSONL Processor
An object-oriented approach for processing JSONL files, incorporating AI security best practices such as PII redaction and error fallbacks.

```python
import json
import re
from typing import Iterator, Dict, Any

class SecureJSONLProcessor:
    """Processes JSONL data securely with PII redaction and error handling."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        # Simple regex for matching common email patterns to demonstrate PII redaction
        self.email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

    def _redact_pii(self, text: str) -> str:
        """Redacts PII (like emails) from the provided text."""
        if not text:
            return text
        return self.email_pattern.sub("[REDACTED_EMAIL]", text)

    def stream_secure(self) -> Iterator[Dict[str, Any]]:
        """Safely streams JSONL records with redaction and fallback handling."""
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                        # Sanitize sensitive fields if they exist
                        if 'prompt' in record and isinstance(record['prompt'], str):
                            record['prompt'] = self._redact_pii(record['prompt'])
                        yield record
                    except json.JSONDecodeError as e:
                        # Fallback mechanism: log error and skip invalid record
                        print(f"Warning: Failed to decode line - {e}")
                        continue
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {self.filepath}")

# Example usage
if __name__ == "__main__":
    # Create dummy file for example
    with open("secure_test.jsonl", "w", encoding="utf-8") as f:
        f.write('{"prompt": "Contact me at user@example.com", "completion": "Sure!"}\n')
        f.write('{"prompt": "Invalid JSON line" \n')

    processor = SecureJSONLProcessor("secure_test.jsonl")
    for safe_record in processor.stream_secure():
        print(f"Processed: {safe_record}")
```

## Additional Key Concepts
- **PII Protection:** Redacting Personally Identifiable Information (PII) like emails or SSNs before feeding data into LLMs to prevent privacy leaks.
- **Fallback Mechanisms:** Implementing robust error handling (e.g., catching `JSONDecodeError`) to ensure one corrupted line doesn't halt the entire data pipeline.
- **OOP Design:** Encapsulating state (like compiled regex patterns for redaction) and behaviors within classes for cleaner, more maintainable code.

## Additional Common Gotchas
- **Leaking PII into AI Models:** Failing to sanitize data before sending it to public LLM APIs, violating compliance and privacy rules. Always sanitize at the file processing layer.
- **Brittle Parsing Logic:** Not wrapping JSON decoding in `try/except` blocks, causing large bulk processing jobs to crash completely over a single malformed line.
