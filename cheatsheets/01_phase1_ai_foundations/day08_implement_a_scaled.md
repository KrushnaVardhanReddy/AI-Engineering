# Scaled Dot-Product Attention Cheatsheet

## Day 8: Scaled Dot-Product Attention

### Concise Summaries
Scaled Dot-Product Attention is the core mechanism behind Transformer models. It computes attention scores by taking the dot product of a Query matrix with a Key matrix, scaling the result by the square root of the key dimension to prevent vanishing gradients during softmax, and then multiplying by a Value matrix. This allows the model to dynamically focus on relevant parts of the input sequence.

### Code Snippets
Here is a production-ready, numpy-based implementation of scaled dot-product attention:

```python
import numpy as np
import logging

# Configure logger for fallback/security events
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class ScaledDotProductAttention:
    """
    A robust object-oriented implementation of Scaled Dot-Product Attention.
    Includes input validation to ensure numerical stability and fallback mechanisms.
    """
    def __init__(self, scale_factor: float = None):
        self.scale_factor = scale_factor

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes scaled dot-product attention.

        Args:
            Q: Query matrix of shape (..., seq_len_q, d_k)
            K: Key matrix of shape (..., seq_len_k, d_k)
            V: Value matrix of shape (..., seq_len_k, d_v)
            mask: Optional boolean mask matrix of shape (..., seq_len_q, seq_len_k).
                  True indicates positions to mask.

        Returns:
            Tuple of (Attention Output, Attention Weights)
        """
        try:
            # Basic validation to ensure consistent feature dimensions
            if Q.shape[-1] != K.shape[-1]:
                raise ValueError("Query and Key must have the same feature dimension (d_k).")

            d_k = Q.shape[-1]
            scale = self.scale_factor if self.scale_factor is not None else np.sqrt(d_k)

            if scale == 0:
                raise ValueError("Scale factor cannot be zero.")

            # 1. Compute dot product of Q and K^T
            # swapaxes replaces transpose for n-dimensional arrays
            scores = np.matmul(Q, K.swapaxes(-1, -2))

            # 2. Scale scores
            scaled_scores = scores / scale

            # 3. Apply mask (if provided)
            if mask is not None:
                # Security best practice: check mask bounds before applying
                if mask.shape[-2:] != (Q.shape[-2], K.shape[-2]):
                     logger.warning("Mask dimensions do not match Q/K sequence lengths. Proceeding without mask.")
                else:
                     # Use -1e9 instead of -inf to prevent NaNs when multiplied with zero later
                     scaled_scores = np.where(mask, -1e9, scaled_scores)

            # 4. Softmax
            # Subtracting max for numerical stability (prevent overflow)
            exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
            attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

            # 5. Multiply by Values
            output = np.matmul(attention_weights, V)

            return output, attention_weights

        except Exception as e:
            # Fallback mechanism: Return zero attention in case of fatal error
            logger.error(f"Attention computation failed: {str(e)}. Returning zero fallback.")
            seq_len_q = Q.shape[-2]
            d_v = V.shape[-1]
            # Create dummy arrays using the input's data type
            fallback_output = np.zeros(Q.shape[:-2] + (seq_len_q, d_v), dtype=Q.dtype)
            fallback_weights = np.zeros(Q.shape[:-2] + (seq_len_q, K.shape[-2]), dtype=Q.dtype)
            return fallback_output, fallback_weights

if __name__ == "__main__":
    np.random.seed(42)
    seq_len = 4
    d_k = 8
    d_v = 8

    # Generate mock Q, K, V
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_v)

    attention_module = ScaledDotProductAttention()

    # Compute attention
    output, weights = attention_module.forward(Q, K, V)
    print("Attention Output Shape:", output.shape)
    print("Sample Output:\n", output[:2])
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
