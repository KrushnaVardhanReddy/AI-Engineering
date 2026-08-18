.PHONY: help generate submit-week submit-day merge sync

help:
	@echo "AI Engineering Mastery - Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make generate            - Regenerate all 90 daily prompt files from the tracker"
	@echo "  make submit-week W=1     - Submit all 7 days of a specific week to Jules"
	@echo "  make submit-day D=1      - Submit a specific day's prompt to Jules"
	@echo "  make merge S=1 E=7       - Merge a batch of PRs (Start to End) from Jules"
	@echo "  make sync                - Pull latest changes from origin main"

generate:
	python3 scripts/generate_daily_prompts.py

submit-week:
	@if [ -z "$(W)" ]; then echo "❌ Error: Week number (W) is required. Usage: make submit-week W=1"; exit 1; fi
	python3 scripts/jules_submit.py --week $(W)

submit-day:
	@if [ -z "$(D)" ]; then echo "❌ Error: Day number (D) is required. Usage: make submit-day D=1"; exit 1; fi
	python3 scripts/jules_submit.py --day $(D)

merge:
	@if [ -z "$(S)" ] || [ -z "$(E)" ]; then echo "❌ Error: Start (S) and End (E) are required. Usage: make merge S=1 E=7"; exit 1; fi
	bash scripts/merge_prs.sh $(S) $(E)

sync:
	git pull origin main
