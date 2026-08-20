# Day 10: Visualize Attention Weights (GPT-2)

## Concise Summaries
Attention weights are matrices representing how much focus each token in a sequence places on every other token when generating its representation. Visualizing these weights helps demystify how models like GPT-2 build contextual understanding, revealing patterns like looking at adjacent tokens, previous words, or specific syntax rules.

## Code Snippets

```python
# Setup: uv pip install transformers matplotlib seaborn torch
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2Tokenizer, GPT2Model

# 1. Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Enable output_attentions=True to extract the weights!
model = GPT2Model.from_pretrained('gpt2', output_attentions=True)

# 2. Prepare the input
text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# 3. Perform a forward pass
with torch.no_grad():
    outputs = model(**inputs)

# outputs.attentions is a tuple of (layer) tensors of shape (batch, heads, seq_len, seq_len)
attentions = outputs.attentions

# 4. Extract attention for visualization
layer = 0   # First layer
head = 0    # First attention head
# Squeeze batch dimension and extract the specific head
attn_matrix = attentions[layer][0, head].numpy()

# 5. Visualize with Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
plt.title(f"Attention Weights - Layer {layer}, Head {head}")
plt.xlabel("Key (Attending To)")
plt.ylabel("Query (Attending From)")
plt.show()
```

## Key Concepts
*   **Query, Key, Value (QKV):** Attention is computed using Query, Key, and Value matrices. The score between a query and a key determines how much "attention" is given.
*   **Attention Matrix:** A square matrix (sequence_length x sequence_length) where element (i, j) represents how much token *i* attends to token *j*.
*   **Attention Heads:** Transformers use multi-head attention. Different heads learn to focus on different linguistic aspects (e.g., subject-verb relationships, punctuation).
*   **Layers:** Deeper layers often capture more complex, semantic relationships, while early layers capture local, syntactic patterns.
*   `output_attentions=True`: The crucial parameter in Hugging Face models required to return the attention tensors alongside the standard hidden states.

## Common Gotchas
1.  **Forgetting `output_attentions=True`:** If you don't explicitly pass this when loading or calling the model, the `attentions` attribute in the output will be `None`.
2.  **Misinterpreting the Matrix Shape:** Attention outputs are highly dimensional tuples. Extracting them requires careful indexing: `attentions[layer_idx][batch_idx, head_idx]`.

## Reference Links
*   [Hugging Face Transformers: Model Outputs (Attentions)](https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions)
*   [Seaborn Heatmap Documentation](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
