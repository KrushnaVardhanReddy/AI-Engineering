# Day 10: Visualize Attention Weights (GPT-2)

## Concise Summaries
Attention weights are matrices representing how much focus each token in a sequence places on every other token when generating its representation. Visualizing these weights helps demystify how models like GPT-2 build contextual understanding, revealing patterns like looking at adjacent tokens, previous words, or specific syntax rules.

## Code Snippets

```python
import os
import logging
import re
from typing import Optional, List, Tuple
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2Tokenizer, GPT2Model

# Configure logging for production
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AttentionVisualizer:
    """
    A production-ready class for extracting and visualizing attention weights from GPT-2.
    Incorporates basic PII sanitization and robust error handling.
    """
    def __init__(self, model_name: str = 'gpt2'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        try:
            logger.info(f"Loading tokenizer and model: {self.model_name}")
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            # CRITICAL: output_attentions=True is required to extract attention weights
            self.model = GPT2Model.from_pretrained(self.model_name, output_attentions=True)
            self.model.eval() # Set to evaluation mode
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise

    def _sanitize_input(self, text: str) -> str:
        """
        Basic PII sanitization: redacting potential email addresses and phone numbers.
        In a real production system, use a dedicated PII detection service (e.g., Presidio).
        """
        # Fallback regex for emails and simple phone numbers
        sanitized = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL_REDACTED]', text)
        sanitized = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REDACTED]', sanitized)
        if text != sanitized:
            logger.info("PII redaction applied to input text.")
        return sanitized

    def get_attention_weights(self, text: str) -> Optional[Tuple[torch.Tensor, List[str]]]:
        """
        Performs a forward pass and returns attention matrices and tokenized words.
        """
        try:
            sanitized_text = self._sanitize_input(text)

            # Tokenize
            inputs = self.tokenizer(sanitized_text, return_tensors="pt")
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

            # Forward pass (no gradient computation needed)
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Extracted attentions (tuple of layers, each [batch_size, num_heads, seq_len, seq_len])
            attentions = outputs.attentions

            if not attentions:
                logger.warning("No attention weights found. Was output_attentions=True set?")
                return None

            return attentions, tokens

        except Exception as e:
            logger.error(f"Error during attention extraction: {str(e)}")
            return None

    def plot_attention_head(
        self,
        attentions: Tuple[torch.Tensor, ...],
        tokens: List[str],
        layer: int = 0,
        head: int = 0,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualizes a specific attention head using a seaborn heatmap.
        """
        try:
            # Validate indices
            num_layers = len(attentions)
            num_heads = attentions[0].shape[1]

            if layer >= num_layers or head >= num_heads:
                logger.error(f"Invalid layer/head index. Max layer: {num_layers-1}, Max head: {num_heads-1}")
                return

            # Extract specific matrix: [batch(0), head, seq_len, seq_len]
            attn_matrix = attentions[layer][0, head].numpy()

            # Plotting
            plt.figure(figsize=(10, 8))
            sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
            plt.title(f"Attention Weights (Layer {layer}, Head {head})\nModel: {self.model_name}")
            plt.xlabel("Key (Attending To)")
            plt.ylabel("Query (Attending From)")

            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)

            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")

            # Fallback: close the plot in non-interactive environments to prevent memory leaks
            plt.close()
            logger.info(f"Successfully processed visualization for layer {layer}, head {head}")

        except Exception as e:
            logger.error(f"Error during plotting: {str(e)}")


if __name__ == "__main__":
    # 1. Initialize Visualizer
    visualizer = AttentionVisualizer(model_name="gpt2")

    # 2. Provide input (including mock PII to test sanitization)
    sample_text = "Alice (alice@example.com) told the cat to sit on the mat."

    # 3. Extract Weights
    result = visualizer.get_attention_weights(sample_text)

    if result:
        attentions, tokens = result

        # 4. Plot specific head (Layer 0, Head 0 usually captures local syntax)
        visualizer.plot_attention_head(
            attentions=attentions,
            tokens=tokens,
            layer=0,
            head=0,
            save_path="attention_l0_h0.png"
        )
```

## Key Concepts
*   **Query, Key, Value (QKV):** Attention is computed using Query, Key, and Value matrices. The score between a query and a key determines how much "attention" is given.
*   **Attention Matrix:** A square matrix (sequence_length x sequence_length) where element (i, j) represents how much token *i* attends to token *j*.
*   **Attention Heads:** Transformers use multi-head attention. Different heads learn to focus on different linguistic aspects (e.g., subject-verb relationships, punctuation).
*   **Layers:** Deeper layers often capture more complex, semantic relationships, while early layers capture local, syntactic patterns.
*   `output_attentions=True`: The crucial parameter in Hugging Face models required to return the attention tensors alongside the standard hidden states.


*   **AI Security: PII Sanitization:** When deploying models that process user text, especially in visualization or logging tools, always sanitize Personal Identifiable Information (PII) like emails or phone numbers *before* tokenization.
*   **AI Security: Fallback Mechanisms:** Robust AI engineering requires `try/except` blocks around model inference. If the forward pass fails (e.g., OOM errors, malformed tokens), gracefully degrade rather than crashing the application.

## Common Gotchas
1.  **Forgetting `output_attentions=True`:** If you don't explicitly pass this when loading or calling the model, the `attentions` attribute in the output will be `None`.
2.  **Misinterpreting the Matrix Shape:** Attention outputs are highly dimensional tuples. Extracting them requires careful indexing: `attentions[layer_idx][batch_idx, head_idx]`.

3.  **Missing Error Handling / Fallbacks:** Raw PyTorch/Transformers inference can throw cryptic errors. Always wrap inference calls in `try/except` blocks in production to prevent unexpected downtime.
4.  **Logging PII:** Visualizing attention matrices inherently reveals the input tokens. Ensure PII is redacted before generating heatmaps or logging token arrays.

## Reference Links
*   [Hugging Face Transformers: Model Outputs (Attentions)](https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions)
*   [Seaborn Heatmap Documentation](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
