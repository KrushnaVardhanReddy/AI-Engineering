import os

PROMPTS_DIR = "/home/krushna/Project/AI-Engineering/prompts/manim"

# Default template for all standard topics
TEMPLATE = """<PASTE MANDATORY RULES BLOCK>

You are tasked with generating a Manim animation script for our AI/ML interview prep series.

### Setup & Requirements
- **Framework:** Manim Community Edition (`manim` package on PyPI, version >= 0.18.0). Do NOT use the legacy `manimlib` (3b1b) version.
- **Audio:** `manim-voiceover[gtts]` (install with `pip install manim-voiceover[gtts]`). Use `GTTSService` — it is free and requires no API key.
- **Python Version:** Python 3.10+
- **Output Location:** Create the script inside the `animations/` directory.

### The Task: Day {day_num}
Today's Topic: **{topic}**

Write a Python script using the Manim Community API. The animation must:

1. **Follow this exact narrative structure** (each section displayed as animated text + diagrams, each section narrated aloud):
   - **What is it?** — A clear 1-sentence definition (spoken + shown on screen).
   - **Why do we need it?** — The specific problem it solves, with a "before vs. after" visual if applicable (spoken + shown).
   - **Use Cases** — 2 real-world examples, e.g., "ChatGPT uses X, Spotify uses Y" (spoken + shown).
   - **Key Interview Insight** — The most common "gotcha" or tradeoff interviewers test (spoken + shown in a callout box).

2. **Audio Narration (Required):**
   - Inherit from `VoiceoverScene` instead of `Scene`.
   - Call `self.set_speech_service(GTTSService())` at the start of `construct()`.
   - Wrap every animation block in a `with self.voiceover(text="...") as tracker:` context manager.
   - The spoken script must match what is shown on screen.
   - Required imports:
     ```python
     from manim import *
     from manim_voiceover import VoiceoverScene
     from manim_voiceover.services.gtts import GTTSService
     ```

3. **Duration:** Target ~90 seconds of animation. Let `tracker.duration` drive `self.wait(tracker.duration)` so audio and visuals stay in sync.

4. **Aesthetic (Whiteboard Style):**
   - Set background to white: `self.camera.background_color = WHITE`
   - Use BLACK for all primary text and outlines.
   - Use vibrant accent colors (BLUE, RED, GREEN, PURPLE) for highlights and emphasis. Avoid light colors (YELLOW, LIGHT_GRAY) that are hard to see on white.
   - Use `Tex` or `MathTex` for all mathematical expressions.

5. **Technical:**
   - The script must contain a single `VoiceoverScene` subclass.
   - Use only non-deprecated Manim Community >= 0.18.0 APIs.
   - The scene must be fully self-contained with no external assets.
   - Smooth transitions only: `FadeIn`, `FadeOut`, `Write`, `Transform`, `MoveToTarget`.

6. Name the file: `animations/day_{day_num:02d}_{safe_topic}.py`

Commit: "jules: add manim animation for day {day_num} - {topic}"
"""

# Special deep-dive template for Transformer data-flow topics
TRANSFORMER_TEMPLATE = """<PASTE MANDATORY RULES BLOCK>

You are tasked with generating a Manim animation script for our AI/ML interview prep series.

### Setup & Requirements
- **Framework:** Manim Community Edition (`manim` package on PyPI, version >= 0.18.0). Do NOT use the legacy `manimlib` (3b1b) version.
- **Audio:** `manim-voiceover[gtts]` (install with `pip install manim-voiceover[gtts]`). Use `GTTSService` — it is free and requires no API key.
- **Python Version:** Python 3.10+
- **Output Location:** Create the script inside the `animations/` directory.

### The Task: Day {day_num}
Today's Topic: **{topic}**

This is a **data-flow deep-dive** animation. You must use the example sentence **"The cat sat"** and trace how a real token travels through this specific stage of the Transformer — with narration explaining every step.

Write a Python script using the Manim Community API. The animation must show:

1. **Audio Narration (Required):**
   - Inherit from `VoiceoverScene` instead of `Scene`.
   - Call `self.set_speech_service(GTTSService())` at the start of `construct()`.
   - Wrap every animation block in a `with self.voiceover(text="...") as tracker:` context manager.
   - The spoken script must narrate exactly what is being shown (e.g., "Here, the token 'cat' is multiplied by the Query weight matrix W_Q to produce the query vector q").
   - Required imports:
     ```python
     from manim import *
     from manim_voiceover import VoiceoverScene
     from manim_voiceover.services.gtts import GTTSService
     ```

2. **Input Setup** — Display the 3 tokens ["The", "cat", "sat"] as labeled boxes. Highlight the active token in BLUE. Narrate: "We start with three tokens..."

3. **Data Flow — Step by Step (each step narrated):**
   - Show the input vector/matrix arriving as an arrow into the current stage.
   - Animate the internal computation (e.g., matrix multiply with W_Q/W_K/W_V, softmax) with small example numbers (2D or 3x3 matrices).
   - Show which values **get updated** — highlight changed cells in GREEN.
   - Show the output flowing out as an arrow to the next stage.

4. **Forward Pass vs. Reverse (Backpropagation if applicable, narrated):**
   - Show the forward pass first with solid arrows (narrate: "In the forward pass...").
   - Then show the gradient/error signal flowing backward with dashed RED arrows (narrate: "During backprop, the gradient flows back...").
   - Label which weights (W_Q, W_K, W_V, W_O, FFN weights) get updated.

5. **Key Interview Insight** — End with a callout box and spoken summary of the most common interview "gotcha" for this stage.

6. **Duration:** Use `self.wait(tracker.duration)` inside each `with self.voiceover(...)` block to sync audio and animation.

7. **Aesthetic (Whiteboard Style):**
   - Set background to white: `self.camera.background_color = WHITE`
   - BLACK for labels and outlines. BLUE for input tokens, GREEN for forward-pass outputs, RED for gradients, PURPLE for weight matrices.
   - Use `MathTex` for all matrix expressions.

8. **Technical:**
   - The script must contain a single `VoiceoverScene` subclass.
   - Use only non-deprecated Manim Community >= 0.18.0 APIs.
   - Smooth transitions only: `FadeIn`, `FadeOut`, `Write`, `Transform`, `MoveToTarget`.

9. Name the file: `animations/day_{day_num:02d}_{safe_topic}.py`

Commit: "jules: add manim animation for day {day_num} - {topic}"
"""

# Tags to identify which topics need the deep-dive Transformer template
TRANSFORMER_DAYS = {
    "Transformer Architecture Overview - End-to-End Data Flow",
    "Tokenization and Input Embedding - How Text Becomes Numbers",
    "Positional Encoding - Injecting Order into Embeddings",
    "Self-Attention - How Q, K, V Are Computed from One Token",
    "Attention Score Calculation - Softmax and Weighted Sum",
    "Multi-Head Attention - Parallel Heads and Concatenation",
    "Feed-Forward Layer Inside the Transformer Block",
    "Encoder Stack - How Data Flows Through All 6 Layers",
    "Decoder Stack - Masked Attention and Cross-Attention Explained",
    "Encoder vs Decoder - What Changes Between BERT and GPT",
    "Transformer Training - Forward Pass, Loss, and Backprop",
}

# 60 micro-topics — sorted from math foundations → classical ML → deep learning → Transformers → LLM/GenAI
TOPICS = [
    # ── Math & Vector Foundations (Days 1–6) ─────────────────────────────────
    "Vector Basics and Coordinates",
    "Dot Product Visualization",
    "Cosine Similarity in 3D",
    "Euclidean Distance vs Cosine Similarity",
    "When to Use Which Distance Metric",
    "Cross-Entropy Loss - The Math Behind Training",

    # ── Embeddings & Semantic Search (Days 7–12) ─────────────────────────────
    "Introduction to Word Embeddings",
    "Semantic Space Mapping (King - Man + Woman = Queen)",
    "Sentence Transformers vs Word2Vec",
    "What is a Vector Database?",
    "Embedding Models and Dimensionality",
    "Chunking Strategies for RAG",

    # ── Retrieval & Vector Indexing (Days 13–18) ──────────────────────────────
    "K-Nearest Neighbors (KNN)",
    "HNSW Graph Indexing - Layer 0",
    "HNSW Graph Indexing - Multi-Layer Search",
    "Inverted File Index (IVF) and Voronoi Cells",
    "Retrieval-Augmented Generation (RAG) Flow",
    "Attention Masking - Padding Masks vs Causal Masks",

    # ── Classical ML Foundations (Days 19–26) ────────────────────────────────
    "Bias-Variance Tradeoff",
    "Overfitting vs Underfitting",
    "Regularization: L1 (Lasso) vs L2 (Ridge)",
    "Decision Trees and Recursive Splitting",
    "Random Forests and Bagging",
    "Support Vector Machines and the Kernel Trick",
    "PCA - Principal Component Analysis",
    "K-Means Clustering",

    # ── Evaluation & Metrics (Days 27–29) ────────────────────────────────────
    "Precision, Recall, and F1 Score",
    "AUC-ROC Curve Explained",
    "t-SNE for High-Dimensional Visualization",

    # ── Core Deep Learning (Days 30–38) ──────────────────────────────────────
    "Introduction to Neural Networks",
    "Activation Functions (ReLU, Sigmoid, Tanh)",
    "Vanishing Gradient Problem",
    "Gradient Descent - SGD vs Adam vs RMSProp",
    "Learning Rate Schedulers - Warmup and Cosine Decay",
    "Backpropagation Chain Rule",
    "Batch Normalization vs Layer Normalization",
    "Dropout Regularization",
    "CNN Architecture - Conv Layer Forward Pass",

    # ── Sequence Models (Days 39–40) ─────────────────────────────────────────
    "RNN Sequential Bottleneck",
    "LSTM and the Gating Mechanism",

    # ── Transformer Deep-Dive: Data Flow (Days 41–51) ─────────────────────────
    "Transformer Architecture Overview - End-to-End Data Flow",
    "Tokenization and Input Embedding - How Text Becomes Numbers",
    "Positional Encoding - Injecting Order into Embeddings",
    "Self-Attention - How Q, K, V Are Computed from One Token",
    "Attention Score Calculation - Softmax and Weighted Sum",
    "Multi-Head Attention - Parallel Heads and Concatenation",
    "Feed-Forward Layer Inside the Transformer Block",
    "Encoder Stack - How Data Flows Through All 6 Layers",
    "Decoder Stack - Masked Attention and Cross-Attention Explained",
    "Encoder vs Decoder - What Changes Between BERT and GPT",
    "Transformer Training - Forward Pass, Loss, and Backprop",

    # ── LLM Inference & Sampling (Days 52–54) ────────────────────────────────
    "Token Sampling - Temperature, Top-K, and Top-P",
    "Prompt Engineering - Zero-Shot, Few-Shot, Chain-of-Thought",
    "Context Window and KV Cache",

    # ── Generative AI & Fine-Tuning (Days 55–60) ─────────────────────────────
    "LoRA (Low-Rank Adaptation) Fine-Tuning",
    "RLHF - Reinforcement Learning from Human Feedback",
    "Autoencoder and Latent Space",
    "Variational Autoencoder (VAE) Sampling",
    "Diffusion Models - Forward and Reverse Process",
    "Convolutional Filter Sliding and Max Pooling (CNN)",
]


def generate_prompts():
    if not os.path.exists(PROMPTS_DIR):
        os.makedirs(PROMPTS_DIR)

    for idx, topic in enumerate(TOPICS):
        day_num = idx + 1
        safe_topic = (
            topic.lower()
            .replace(' ', '_')
            .replace('(', '').replace(')', '')
            .replace('-', '_')
            .replace('+', 'plus')
            .replace('=', 'equals')
            .replace('/', '_')
            .replace(',', '')
            .replace(':', '')
        )

        # Use the richer Transformer template for deep-dive days
        template = TRANSFORMER_TEMPLATE if topic in TRANSFORMER_DAYS else TEMPLATE
        prompt_content = template.format(day_num=day_num, topic=topic, safe_topic=safe_topic)
        file_path = os.path.join(PROMPTS_DIR, f"day_{day_num:02d}.txt")

        with open(file_path, 'w', encoding='utf-8') as pf:
            pf.write(prompt_content)

    print(f"✅ Generated {len(TOPICS)} Manim daily prompts in {PROMPTS_DIR}.")
    print(f"   {len(TRANSFORMER_DAYS)} of those use the deep-dive Transformer data-flow template.")


if __name__ == "__main__":
    generate_prompts()
