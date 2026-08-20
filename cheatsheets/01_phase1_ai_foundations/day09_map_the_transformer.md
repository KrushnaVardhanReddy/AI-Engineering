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
