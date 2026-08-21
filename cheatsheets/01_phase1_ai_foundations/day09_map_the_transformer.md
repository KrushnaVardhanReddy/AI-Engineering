# Day 9: Map the Transformer Architecture (Positional Encoding & Multi-Head Attention)

## Concise Summaries
The Transformer architecture revolutionized AI by removing sequence dependency (RNNs) in favor of parallel processing via the **Self-Attention** mechanism. To maintain the order of the sequence, it injects **Positional Encoding** into the embeddings. **Multi-Head Attention** allows the model to jointly attend to information from different representation subspaces at different positions.

## Code Snippets
Here is a NumPy implementation of Positional Encoding and a simplified Scaled Dot-Product Attention (the core of Multi-Head Attention).

```python
import numpy as np

# 1. Positional Encoding
def get_positional_encoding(seq_len, d_model):
    """
    Generate positional encodings for a given sequence length and model dimension.
    """
    pe = np.zeros((seq_len, d_model))
    position = np.arange(0, seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

# 2. Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute 'Scaled Dot-Product Attention'.
    Q, K, V are matrices of shape (seq_len, d_k)
    """
    d_k = Q.shape[-1]
    # Compute attention scores
    scores = np.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

    # Apply mask if provided (e.g., for causal masking)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Softmax to get probabilities (attention weights)
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True)) # stability
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Multiply by values
    output = np.matmul(attention_weights, V)
    return output, attention_weights

# --- Example Usage ---
seq_length = 4
d_model = 8

# Generate positional encodings
pe = get_positional_encoding(seq_length, d_model)
print("Positional Encoding Shape:", pe.shape)

# Simulate Q, K, V for a single head
np.random.seed(42)
Q = np.random.randn(seq_length, d_model)
K = np.random.randn(seq_length, d_model)
V = np.random.randn(seq_length, d_model)

# Compute attention
output, weights = scaled_dot_product_attention(Q, K, V)
print("Attention Output Shape:", output.shape)
```

## Key Concepts
*   **Positional Encoding:** Sinusoidal functions (sine and cosine) added to word embeddings to inject information about the relative or absolute position of the tokens.
*   **Self-Attention:** A mechanism relating different positions of a single sequence in order to compute a representation of the sequence.
*   **Query (Q), Key (K), Value (V):** Abstract matrices derived from the input sequence. The Query is what we are looking for, the Key is what we match against, and the Value is the actual content.
*   **Scaled Dot-Product Attention:** The formula: $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$. Scaling by $\sqrt{d_k}$ prevents the dot products from growing too large and pushing the softmax function into regions with extremely small gradients.
*   **Multi-Head Attention:** Running several attention mechanisms in parallel (heads) and concatenating their outputs, allowing the model to focus on different parts of the sequence simultaneously.

## Common Gotchas
*   **Forgetting to Scale in Attention:** Omitting the division by $\sqrt{d_k}$ can lead to vanishing gradients during training because the softmax function saturates.
*   **Incorrect Matrix Shapes:** When implementing Multi-Head Attention, reshaping and transposing the Q, K, and V tensors correctly is notoriously error-prone. Ensure you keep track of `(batch_size, num_heads, seq_len, head_dim)`.

## Reference Links
*   [Attention Is All You Need (Original Paper)](https://arxiv.org/abs/1706.03762)
*   [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)

### Production-Ready OOP Implementation

The following code demonstrates a robust, object-oriented approach with fallback mechanisms to prevent Out-Of-Memory (OOM) errors and input validation for tensor dimensions.

```python
import numpy as np
import logging
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PositionalEncoding:
    """
    Positional Encoding generator using clean OOP principles.
    Includes fallback mechanisms for extreme sequence lengths to prevent OOM errors.
    """
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pe_matrix = self._generate_pe_matrix()

    def _generate_pe_matrix(self) -> np.ndarray:
        pe = np.zeros((self.max_seq_len, self.d_model))
        position = np.arange(0, self.max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def get_encoding(self, seq_len: int) -> np.ndarray:
        if seq_len > self.max_seq_len:
            logger.warning(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}. Truncating to prevent memory exhaustion.")
            # Fallback mechanism: return max length encoding
            return self.pe_matrix[:self.max_seq_len, :]
        return self.pe_matrix[:seq_len, :]

class MultiHeadAttention:
    """
    Multi-Head Attention block with input validation and safe processing.
    """
    def __init__(self, d_model: int, num_heads: int):
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        # Input validation for security and stability
        if not (Q.shape == K.shape == V.shape):
            logger.error("Dimension mismatch between Q, K, and V tensors.")
            raise ValueError("Q, K, V tensors must have the same shape.")

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        # Stable softmax
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        output = np.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        batch_size, seq_len, _ = Q.shape

        # Simulate projection and reshaping for multiple heads
        Q_split = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K_split = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V_split = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        # Compute attention per head
        head_outputs, _ = self.scaled_dot_product_attention(Q_split, K_split, V_split, mask)

        # Concatenate heads
        concat_output = head_outputs.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        return concat_output

# --- Example Usage ---
if __name__ == "__main__":
    logger.info("Initializing Transformer components...")

    seq_length = 4
    d_model = 8
    num_heads = 2
    batch_size = 1

    # 1. Positional Encoding
    pe_layer = PositionalEncoding(d_model=d_model, max_seq_len=10)
    pe = pe_layer.get_encoding(seq_length)
    logger.info(f"Positional Encoding Shape: {pe.shape}")

    # 2. Multi-Head Attention
    mha_layer = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    np.random.seed(42)
    Q = np.random.randn(batch_size, seq_length, d_model)
    K = np.random.randn(batch_size, seq_length, d_model)
    V = np.random.randn(batch_size, seq_length, d_model)

    try:
        output = mha_layer.forward(Q, K, V)
        logger.info(f"Multi-Head Attention Output Shape: {output.shape}")
    except Exception as e:
        logger.error(f"Attention computation failed: {e}")
```

### Additional Key Concepts (Production & Security)
*   **Input Validation:** Always validate the shapes of `Q`, `K`, and `V` tensors before processing. Dimension mismatches can cause silent failures or unexpected resource exhaustion.
*   **Fallback Mechanisms:** Implement fail-safes for sequence length constraints. If an input exceeds `max_seq_len`, truncating the sequence or returning the maximum available encoding is safer than crashing the service with an Out-Of-Memory (OOM) error.

### Additional Common Gotchas (Production)
*   **Unstable Softmax:** A naive implementation of softmax ($e^x / \sum e^x$) can lead to numerical instability (NaNs) if scores are very large. Subtracting the maximum score before exponentiating is a crucial production safeguard.
*   **Lack of Memory Bounds:** In production, dynamic sequence lengths can lead to unpredictable memory usage. Pre-allocating matrices up to a `max_seq_len` prevents dynamic reallocation overhead and bounds the memory footprint.
