import os
import re

TRACKER_FILE = "/home/krushna/Project/AI-Engineering/90-Day AI Engineering Mastery Tracker_ From Software Engineer to AI Specialist.md"
PROMPTS_DIR = "/home/krushna/Project/AI-Engineering/prompts/cheatsheets"

PHASE_FOLDERS = {
    "PHASE 1": "01_phase1_ai_foundations",
    "PHASE 2": "02_phase2_vector_dbs_rag",
    "PHASE 3": "03_phase3_advanced_rag_agents",
    "PHASE 4": "04_phase4_agent_orchestration",
    "PHASE 5": "05_phase5_eval_multimodal",
    "PHASE 6": "06_phase6_cloud_mcp_finetuning",
    "PHASE 7": "07_phase7_system_design_voice"
}

TEMPLATE = """System Role: Act as a Senior AI Engineering Instructor. I am an experienced software engineer transitioning into AI Engineering through a rigorous 90-day self-study program, focusing on practical systems, agentic design, and AI security.

Your Task: I want you to act as my daily tutor for Day {day_num}. Generate a comprehensive cheat sheet for today's topic in Markdown format.

Content Guidelines:
- **Concise Summaries:** Provide a highly condensed summary of the core concept. No fluff.
- **Code Snippets:** Provide copy-pasteable, production-ready code examples with clean OOP design and exact import syntax (crucial for IDE-less interviews).
- **Key Concepts:** A quick reference list of essential terms or parameters, including any relevant AI security concepts (e.g., PII, jailbreaks).
- **Common Gotchas:** 1-2 common mistakes to watch out for in production.
- **Reference Links:** Provide 1-2 authoritative links to official documentation.

Anti-Hallucination Guardrails:
1. **No Fictitious APIs:** Only use officially supported and well-documented libraries.
2. **Stable Syntax:** Strictly use stable, widely accepted design patterns.
3. **Working Code:** Your code examples must be logically sound and syntactically valid.

Context:
This task is part of a 90-day curriculum. You are currently generating material for:
**{phase_name}**

Today's Topic:
Day {day_num}: {topic}

Output Instructions:
Please generate the complete cheat sheet as a **Markdown file (.md)**.
Save the file at exactly: `cheatsheets/{phase_folder}/day{day_num:02d}_{topic_slug}.md`

If the file already exists, carefully append new sections to it without breaking the existing structure.

Please generate the material right now.
"""

def generate_prompts():
    if not os.path.exists(PROMPTS_DIR):
        os.makedirs(PROMPTS_DIR)
        
    with open(TRACKER_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
        
    chunks = content.split("#### ")
    count = 0
    
    for chunk in chunks:
        if "PHASE" not in chunk:
            continue
            
        lines = chunk.split('\n')
        phase_raw = lines[0].strip()
        phase_name = re.sub(r'^\d+\\.\s*', '', phase_raw)
        
        # Match "PHASE X" to get the correct folder
        phase_match = re.search(r'(PHASE \d+)', phase_name)
        phase_key = phase_match.group(1) if phase_match else "PHASE 1"
        phase_folder = PHASE_FOLDERS.get(phase_key, "01_phase1_ai_foundations")
        
        pattern = r'\*\*Day\s+(\d+):\*\*\s+(.*?)(?=\*\*Day\s+\d+:\*\*|\*\*Foundational Checklist|\*\*Project Milestone|\*\*AI Security|\*\*Personal Branding|\Z)'
        matches = re.finditer(pattern, chunk, re.DOTALL)
        
        for match in matches:
            day_num = int(match.group(1))
            topic = match.group(2).strip()
            
            # Create a simple slug from the topic description (first few words)
            slug_words = [w.lower() for w in re.findall(r'[a-zA-Z0-9]+', topic)][:3]
            topic_slug = "_".join(slug_words)
            
            prompt_content = TEMPLATE.format(
                day_num=day_num, 
                topic=topic, 
                phase_name=phase_name,
                phase_folder=phase_folder,
                topic_slug=topic_slug
            )
            
            file_path = os.path.join(PROMPTS_DIR, f"day_{day_num:02d}.txt")
            with open(file_path, 'w', encoding='utf-8') as pf:
                pf.write(prompt_content)
            count += 1
            
    print(f"Generated {count} cheat sheet prompts in {PROMPTS_DIR}.")

if __name__ == "__main__":
    generate_prompts()
