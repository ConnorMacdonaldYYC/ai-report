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

# Deploy code + .env to the Raspberry Pi.
# Bundles the repo into a single file, scps it to /tmp on the Pi, and
# copies .env to ~/ai-report. The cron job on the Pi fetches the bundle
# each run, so re-running `make deploy` before Monday publishes updates.
PI ?= connorspi
PI_PATH ?= ~/ai-report
BUNDLE := /tmp/ai-report.bundle

deploy:
	git bundle create $(BUNDLE) --all
	ssh $(PI) "mkdir -p $(PI_PATH)"
	scp $(BUNDLE) $(PI):$(BUNDLE)
	scp .env $(PI):$(PI_PATH)/.env
	rm $(BUNDLE)
	@echo ""
	@echo "Deployed. On the Pi:"
	@echo "  cd $(PI_PATH) && git clone /tmp/ai-report.bundle . && uv sync"
