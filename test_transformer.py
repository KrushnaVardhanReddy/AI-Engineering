import numpy as np
import logging
from typing import Optional, Tuple, Any

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
def main():
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

if __name__ == "__main__":
    main()
