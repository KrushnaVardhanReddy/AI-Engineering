import os
import re

TRACKER_FILE = "/home/krushna/Project/AI-Engineering/90-Day AI Engineering Mastery Tracker_ From Software Engineer to AI Specialist.md"
PROMPTS_DIR = "/home/krushna/Project/AI-Engineering/prompts/daily"

TEMPLATE = """System Role: Act as a Senior AI Engineering Instructor. I am an experienced software engineer transitioning into AI Engineering through a rigorous 90-day self-study program.

My Learning Style: I learn through a "production-first" mindset. I do not want just surface-level theory; I need to understand the underlying architecture and immediately apply it through code.

Your Task: I want you to act as my daily tutor for Day {day_num}. Generate a comprehensive study module that includes:
- **Core Theory (Just-in-Time):** Explain the "why" and "how" behind the day's concept.
- **Code Implementation:** Provide a tiered progression of Python code examples (Basic, Medium, Advanced).
  - **Basic:** Isolate the core concept with minimal boilerplate.
  - **Medium:** Show how multiple concepts interact.
  - **Advanced:** Provide production-grade implementation with strict type hinting, docstrings, and error handling.
- **Practical Lab / Homework:** Give me a specific, actionable coding task to complete for today.
- **Common Pitfalls:** Tell me what goes wrong in production regarding this topic.
- **Reference Links:** Provide 2-3 links to official documentation or highly reputable sources for further reading.

Anti-Hallucination Guardrails:
1. **No Fictitious APIs:** Do not invent Python libraries, APIs, or class methods. Only use officially supported and well-documented libraries (e.g., LangChain, Qdrant, Streamlit, Pydantic).
2. **Stable Syntax:** For rapidly changing frameworks (like LangGraph or vLLM), strictly use stable, widely accepted design patterns. Do not guess or hallucinate parameters.
3. **Working Code:** Your code examples must be logically sound and syntactically valid. Provide fully working snippets, not pseudo-code.
4. **Strict Scoping:** Do not deviate from or over-explain concepts outside of "Today's Topic".

Context:
This task is part of a 90-day curriculum. You are currently generating material for:
**{phase_name}**

Today's Topic:
Day {day_num}: {topic}

Output Instructions:
Please generate the complete learning material as a **Jupyter Notebook (.ipynb)**.
Save the file at exactly: `modules/day_{day_num:02d}.ipynb`

If the notebook already exists:
1. READ the existing file first to understand its current structure.
2. Do NOT recreate it from scratch. 
3. Locate the existing code sections and elegantly integrate the Basic, Medium, and Advanced examples.
4. Ensure you do not break the underlying JSON structure of the `.ipynb` file.

If it does not exist, create it from scratch with cleanly separated Markdown cells (for theory/instructions) and Python Code cells (for implementations/lab).

Please generate the material right now.
"""

def generate_prompts():
    if not os.path.exists(PROMPTS_DIR):
        os.makedirs(PROMPTS_DIR)
        
    with open(TRACKER_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
        
    chunks = content.split("#### ")
    
    count = 0
    index_md = "## 📚 Modules Index\n\n"
    
    for chunk in chunks:
        if "PHASE" not in chunk:
            continue
            
        lines = chunk.split('\n')
        phase_name = lines[0].strip()
        # Remove the leading "2\. " or similar numbers
        phase_name = re.sub(r'^\d+\\.\s*', '', phase_name)
        
        index_md += f"### {phase_name}\n"
        
        pattern = r'\*\*Day\s+(\d+):\*\*\s+(.*?)(?=\*\*Day\s+\d+:\*\*|\*\*Foundational Checklist|\*\*Project Milestone|\*\*AI Security|\*\*Personal Branding|\Z)'
        matches = re.finditer(pattern, chunk, re.DOTALL)
        
        for match in matches:
            day_num = int(match.group(1))
            topic = match.group(2).strip()
            
            prompt_content = TEMPLATE.format(day_num=day_num, topic=topic, phase_name=phase_name)
            file_path = os.path.join(PROMPTS_DIR, f"day_{day_num:02d}.txt")
            
            with open(file_path, 'w', encoding='utf-8') as pf:
                pf.write(prompt_content)
            count += 1
            
            # Append to index
            index_md += f"- [Day {day_num:02d}: {topic}](day_{day_num:02d}.ipynb)\n"
            
        index_md += "\n"
            
    print(f"Generated {count} daily prompts in {PROMPTS_DIR} with context.")
    
    # Generate index.ipynb
    import json
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in index_md.split("\n")]
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    if not os.path.exists("modules"):
        os.makedirs("modules")
        
    with open("modules/index.ipynb", 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
        
    print("Successfully generated modules/index.ipynb with the Modules Index.")

if __name__ == "__main__":
    generate_prompts()
