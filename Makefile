.PHONY: lint test test-live sync run eval deploy

lint:
	ruff check src tests run.py
	mypy src tests run.py

test:
	pytest tests/ -v

test-live:
	pytest tests/integration/test_email_live.py -v --live-email -s --log-cli-level=INFO

sync:
	uv sync

run:
	uv run ai-report

# Run the model evaluation matrix.
# Usage: make eval GENERATORS=gpt-4o,claude-sonnet-4-6 EVALUATORS=gpt-4o-mini SEEDS=3
eval:
	uv run python -m src.eval --generators $(GENERATORS) --evaluators $(EVALUATORS) --seeds $(SEEDS)

# Sync .env to the Raspberry Pi and fix ownership for the aireport service user.
# Code updates flow the other way: push to GitHub, the Pi pulls weekly
# (ai-report-update.service).
PI ?= connorspi
PI_PATH ?= ~/ai-report

deploy:
	scp .env $(PI):$(PI_PATH)/.env
	ssh $(PI) "sudo chown root:aireport $(PI_PATH)/.env && sudo chmod 640 $(PI_PATH)/.env"
	@echo "Synced .env to $(PI):$(PI_PATH)/.env (root:aireport 640)"
