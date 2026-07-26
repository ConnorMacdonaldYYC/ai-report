.PHONY: lint test test-live sync run

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
