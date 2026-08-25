.PHONY: help setup jupyter generate generate-cheatsheets submit-week submit-day submit-cheatsheet-week submit-cheatsheet-day merge sync generate-manim submit-manim-week submit-manim-day

help:
	@echo "AI Engineering Mastery - Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make setup               - Create virtual env and install base dependencies"
	@echo "  make jupyter             - Start the Jupyter Lab server locally"
	@echo "  make generate            - Regenerate all 90 daily prompt files from the tracker"
	@echo "  make generate-cheatsheets- Regenerate all 90 daily cheatsheet prompts"
	@echo "  make submit-week W=1     - Submit all 7 days of a specific week to Jules"
	@echo "  make submit-day D=1      - Submit a specific day's prompt to Jules"
	@echo "  make submit-cheatsheet-week W=1 - Submit week's cheatsheet prompts to Jules"
	@echo "  make submit-cheatsheet-day D=1  - Submit day's cheatsheet prompt to Jules"
	@echo "  make merge S=1 E=7       - Merge a batch of PRs (Start to End) from Jules"
	@echo "  make sync                - Pull latest changes from origin main"
	@echo ""
	@echo "Manim Animations:"
	@echo "  make generate-manim       - Regenerate all 65 Manim prompt files"
	@echo "  make submit-manim-week W=1 - Submit a week of Manim prompts to Jules"
	@echo "  make submit-manim-day D=1  - Submit a specific Manim day to Jules"

setup:
	uv venv
	uv pip install jupyterlab pydantic fastapi uvicorn streamlit groq ipykernel manim manim-voiceover gTTS

jupyter:
	.venv/bin/jupyter lab modules/index.ipynb

generate:
	.venv/bin/python3 scripts/generate_daily_prompts.py

generate-cheatsheets:
	.venv/bin/python3 scripts/generate_cheatsheet_prompts.py

submit-week:
	@if [ -z "$(W)" ]; then echo "❌ Error: Week number (W) is required. Usage: make submit-week W=1"; exit 1; fi
	.venv/bin/python3 scripts/jules_submit.py --week $(W)

submit-day:
	@if [ -z "$(D)" ]; then echo "❌ Error: Day number (D) is required. Usage: make submit-day D=1"; exit 1; fi
	.venv/bin/python3 scripts/jules_submit.py --day $(D)

submit-cheatsheet-week:
	@if [ -z "$(W)" ]; then echo "❌ Error: Week number (W) is required. Usage: make submit-cheatsheet-week W=1"; exit 1; fi
	.venv/bin/python3 scripts/jules_submit.py --type cheatsheets --week $(W)

submit-cheatsheet-day:
	@if [ -z "$(D)" ]; then echo "❌ Error: Day number (D) is required. Usage: make submit-cheatsheet-day D=1"; exit 1; fi
	.venv/bin/python3 scripts/jules_submit.py --type cheatsheets --day $(D)

merge:
	@if [ -z "$(S)" ] || [ -z "$(E)" ]; then echo "❌ Error: Start (S) and End (E) are required. Usage: make merge S=1 E=7"; exit 1; fi
	bash scripts/merge_prs.sh $(S) $(E)

sync:
	git pull origin main

generate-manim:
	.venv/bin/python3 scripts/generate_manim_prompts.py

submit-manim-week:
	@if [ -z "$(W)" ]; then echo "❌ Error: Week number (W) is required. Usage: make submit-manim-week W=1"; exit 1; fi
	.venv/bin/python3 scripts/jules_submit.py --type manim --week $(W)

submit-manim-day:
	@if [ -z "$(D)" ]; then echo "❌ Error: Day number (D) is required. Usage: make submit-manim-day D=1"; exit 1; fi
	.venv/bin/python3 scripts/jules_submit.py --type manim --day $(D)

run-manim:
	@if [ -n "$(D)" ]; then \
		DAY=$$(printf "%02d" $(D)); \
		FILE=$$(ls animations/day_$${DAY}_*.py 2>/dev/null); \
		if [ -z "$$FILE" ]; then echo "❌ Error: Day $(D) not found in animations/"; exit 1; fi; \
		.venv/bin/manim -ql $$FILE; \
	else \
		for file in animations/day_*.py; do \
			if [ -f "$$file" ]; then \
				.venv/bin/manim -ql $$file; \
			fi \
		done; \
	fi
