#!/usr/bin/env python3
"""
# Jules Batch Submitter for AI Engineering
# Sends tasks to Jules API to create async coding sessions → GitHub PRs.

Usage:
  python3 scripts/jules_submit.py --day 1      # Submit Day 1 prompt
  python3 scripts/jules_submit.py --week 1     # Submit Week 1 (Days 1-7) prompts
  python3 scripts/jules_submit.py --type cheatsheets --day 1 # Submit cheatsheet prompt
  python3 scripts/jules_submit.py --file path  # Submit a custom prompt from a file
  python3 scripts/jules_submit.py --branch feat# Target a specific branch
"""

import json
import urllib.request
import sys
import os

# ──────────────────────────────────────────────────────────────────────────────
# Config — loads API key from .env.local or .env (never hardcode secrets)
# ──────────────────────────────────────────────────────────────────────────────

def _load_api_key():
    """Read JULES_API_KEY from environment, .env.local, or .env."""
    key = os.environ.get("JULES_API_KEY")
    if key:
        return key
    for envfile in [".env.local", ".env"]:
        # Look in the repo root, not the scripts/ directory
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, envfile)
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("JULES_API_KEY="):
                        return line.split("=", 1)[1].strip()
    print("❌ JULES_API_KEY not found in environment, .env.local, or .env")
    sys.exit(1)

API_KEY = _load_api_key()
API_URL = "https://jules.googleapis.com/v1alpha/sessions"

# ── Repo root (one level up from scripts/) ────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Update this once the GitHub repo is created ──────────────────────────────
REPO_SOURCE = "sources/github/KrushnaVardhanReddy/AI-Engineering"

# Parse branch from args if provided
BRANCH = "main"
if "--branch" in sys.argv:
    idx = sys.argv.index("--branch")
    if idx + 1 < len(sys.argv):
        BRANCH = sys.argv[idx + 1]

# ──────────────────────────────────────────────────────────────────────────────
# Mandatory safety rules (prepended to every prompt)
# ──────────────────────────────────────────────────────────────────────────────

SAFETY_RULES = """
MANDATORY RULES — VIOLATION = REJECTED PR:
1. NEVER stub, mock, or TODO existing implementation code.
2. Commit message must start with "jules: " prefix.

Project: AI-Engineering
Tech stack: Python, LangChain, Qdrant, Streamlit, LangGraph
""".strip()

# ──────────────────────────────────────────────────────────────────────────────
# Submission logic
# ──────────────────────────────────────────────────────────────────────────────

def submit_prompt(full_prompt, task_name="Task"):
    payload = json.dumps({
        "prompt": full_prompt,
        "sourceContext": {
            "source": REPO_SOURCE,
            "githubRepoContext": {
                "startingBranch": BRANCH
            }
        }
    }).encode()

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": API_KEY
        },
        method="POST"
    )

    print(f"🚀 Submitting: {task_name} → branch: {BRANCH}")
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            session_id = result.get("name", "unknown").split("/")[-1]
            print(f"✅ Session created: {session_id}")
            print(f"   View at: https://jules.google.com/session/{session_id}")
    except urllib.error.HTTPError as e:
        print(f"❌ HTTP {e.code}: {e.read().decode()}")
        sys.exit(1)

def submit_day(day_num, prompt_type="daily"):
    prompt_file = f"prompts/{prompt_type}/day_{day_num:02d}.txt"
    full_path = os.path.join(REPO_ROOT, prompt_file)
    if not os.path.exists(full_path):
        print(f"❌ Day {day_num} ({prompt_type}) prompt not found at {prompt_file}")
        sys.exit(1)
        
    with open(full_path) as f:
        prompt_content = f.read()

    full_prompt = SAFETY_RULES + "\n\n---\n\n" + prompt_content
    submit_prompt(full_prompt, task_name=f"Day {day_num} ({prompt_type})")

def submit_file(filepath):
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        sys.exit(1)
    with open(filepath) as f:
        prompt_content = f.read()

    full_prompt = SAFETY_RULES + "\n\n---\n\n" + prompt_content
    submit_prompt(full_prompt, task_name=f"Custom file ({filepath})")

def main():
    args = sys.argv[1:]

    if not args or "--help" in args or "-h" in args:
        print(__doc__)
        sys.exit(0)

    prompt_type = "daily"
    if "--type" in args:
        idx = args.index("--type")
        if idx + 1 < len(args):
            prompt_type = args[idx + 1]

    if "--file" in args:
        idx = args.index("--file")
        if idx + 1 >= len(args):
            print("❌ Please specify a file path after --file.")
            sys.exit(1)
        submit_file(args[idx + 1])
        sys.exit(0)

    if "--day" in args:
        idx = args.index("--day")
        if idx + 1 >= len(args):
            print("❌ Please specify a day number after --day.")
            sys.exit(1)
        submit_day(int(args[idx + 1]), prompt_type)
        sys.exit(0)

    if "--week" in args:
        idx = args.index("--week")
        if idx + 1 >= len(args):
            print("❌ Please specify a week number after --week.")
            sys.exit(1)
        week_num = int(args[idx + 1])
        start_day = (week_num - 1) * 7 + 1
        end_day = min(week_num * 7, 90) # Cap at 90
        print(f"📅 Submitting {prompt_type} prompts for Week {week_num} (Days {start_day} to {end_day})")
        for day in range(start_day, end_day + 1):
            submit_day(day, prompt_type)
        sys.exit(0)

    # Default: print help
    print(__doc__)

if __name__ == "__main__":
    main()
