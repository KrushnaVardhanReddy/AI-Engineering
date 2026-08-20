# Scaled Dot-Product Attention Cheatsheet

## Day 8: Scaled Dot-Product Attention

### Concise Summaries
Scaled Dot-Product Attention is the core mechanism behind Transformer models. It computes attention scores by taking the dot product of a Query matrix with a Key matrix, scaling the result by the square root of the key dimension to prevent vanishing gradients during softmax, and then multiplying by a Value matrix. This allows the model to dynamically focus on relevant parts of the input sequence.

### Code Snippets
Here is a production-ready, numpy-based implementation of scaled dot-product attention:

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Computes scaled dot-product attention.

    Args:
        Q (np.ndarray): Query matrix of shape (..., seq_len_q, d_k)
        K (np.ndarray): Key matrix of shape (..., seq_len_k, d_k)
        V (np.ndarray): Value matrix of shape (..., seq_len_k, d_v)
        mask (np.ndarray, optional): Mask matrix of shape (..., seq_len_q, seq_len_k)

    Returns:
        np.ndarray: Attention output
        np.ndarray: Attention weights
    """
    d_k = Q.shape[-1]

    # 1. Compute dot product of Q and K^T
    # We transpose the last two dimensions of K
    scores = np.matmul(Q, K.swapaxes(-1, -2))

    # 2. Scale by sqrt(d_k)
    scaled_scores = scores / np.sqrt(d_k)

    # 3. Apply mask (if provided)
    if mask is not None:
        # Using a very large negative number to represent -inf
        scaled_scores = np.where(mask == 0, -1e9, scaled_scores)

    # 4. Apply softmax to get attention weights
    # Subtracting max for numerical stability
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 5. Multiply by Values
    output = np.matmul(attention_weights, V)

    return output, attention_weights

# Example usage
np.random.seed(42)
seq_len = 4
d_k = 8
d_v = 8

# Generate random Q, K, V
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)

# Compute attention without mask
output, weights = scaled_dot_product_attention(Q, K, V)
print("Output shape:", output.shape)
print("Output matrix:\n", output)
```

### Key Concepts
- **Q (Queries):** The items we are looking for (e.g., current word representations).
- **K (Keys):** The items we match against the queries to compute relevance.
- **V (Values):** The actual content or features that will be aggregated based on the attention weights.
- **$d_k$ (Dimension of Keys):** Used to scale the dot products to ensure the softmax outputs don't fall into regions with extremely small gradients.
- **Softmax:** Normalizes the attention scores into probabilities that sum to 1.
- **Masking:** Used to prevent the model from attending to certain positions (e.g., future tokens in autoregressive decoding).

### Common Gotchas
- **Forgetting to Scale:** Omitting the division by $\sqrt{d_k}$ can lead to extremely large dot products. This pushes the softmax function into regions where gradients approach zero, stalling training.
- **Incorrect Masking Value:** When masking, subtract a very large number (e.g., `-1e9`) instead of multiplying by zero before the softmax. Multiplying by zero after softmax is incorrect because the masked positions would still contribute to the denominator of the softmax.

### Reference Links
- [Attention Is All You Need (Original Paper)](https://arxiv.org/abs/1706.03762)
- [Illustrated Transformer by Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
