import os

PROMPTS_DIR = "/home/krushna/Project/AI-Engineering/prompts/manim"

TEMPLATE = """<PASTE MANDATORY RULES BLOCK>

You are tasked with generating a Manim animation script for our AI/ML interview prep series.

### Setup & Requirements
- **Framework:** Manim Community Edition (`manim` package on PyPI, version >= 0.18.0). Do NOT use the legacy `manimlib` (3b1b) version.
- **Python Version:** Python 3.10+
- **Output Location:** Create the script inside the `animations/` directory.

### The Task: Day {day_num}
Today's Topic: **{topic}**

1. Write a Python script using the Manim Community API to visually explain this specific concept.
2. The script must contain a `Scene` class.
3. **Aesthetic Requirements (Whiteboard Style):** 
   - Set the background color to pure white (`self.camera.background_color = WHITE`).
   - Use black for primary text, equations, and shapes (`color=BLACK`).
   - Use vibrant secondary colors (e.g., blue, red, green) for emphasis, avoiding light colors like yellow that are hard to see on white.
4. Ensure the scene is clean, well-commented, and uses smooth transitions (`Transform`, `FadeIn`).
5. Name the file logically: `animations/day_{day_num:02d}_{safe_topic}.py`.

### Code Constraints
- Must include `from manim import *`
- Do not use any deprecated functions from older manim versions.
- Ensure the scene is self-contained.

Commit: "jules: add manim animation script for day {day_num} - {topic}"
"""

# A curated list of micro-topics mapped to days
TOPICS = [
    "Vector Basics and Coordinates",
    "Dot Product Visualization",
    "Cosine Similarity in 3D",
    "Euclidean Distance vs Cosine Similarity",
    "Introduction to Word Embeddings",
    "Semantic Space Mapping (King - Man + Woman = Queen)",
    "What is a Vector Database?",
    "Chunking Strategies for RAG",
    "K-Nearest Neighbors (KNN)",
    "HNSW Graph Indexing - Layer 0",
    "HNSW Graph Indexing - Multi-Layer Search",
    "Inverted File Index (IVF)",
    "Retrieval-Augmented Generation (RAG) Flow",
    "Introduction to Neural Networks",
    "Activation Functions (ReLU, Sigmoid)",
    "Gradient Descent Landscape",
    "Backpropagation Chain Rule",
    "Dropout Regularization",
    "RNN Sequential Bottleneck",
    "Transformer Architecture Overview",
    "Positional Encoding Waves",
    "Self-Attention QKV Matrices",
    "Attention Score Calculation",
    "Multi-Head Attention Parallelism",
    "Encoder vs Decoder Structure",
    "LoRA (Low-Rank Adaptation)",
    "Diffusion Forward Process",
    "Diffusion Reverse Process",
    "Convolutional Filter Sliding (CNN)",
    "Max Pooling Operation",
]

def generate_prompts():
    if not os.path.exists(PROMPTS_DIR):
        os.makedirs(PROMPTS_DIR)
        
    for idx, topic in enumerate(TOPICS):
        day_num = idx + 1
        safe_topic = topic.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').replace('+', 'plus').replace('=', 'equals')
        
        prompt_content = TEMPLATE.format(day_num=day_num, topic=topic, safe_topic=safe_topic)
        file_path = os.path.join(PROMPTS_DIR, f"day_{day_num:02d}.txt")
        
        with open(file_path, 'w', encoding='utf-8') as pf:
            pf.write(prompt_content)
            
    print(f"✅ Generated {len(TOPICS)} Manim daily prompts in {PROMPTS_DIR}.")

if __name__ == "__main__":
    generate_prompts()
