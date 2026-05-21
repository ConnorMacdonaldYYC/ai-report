.PHONY: lint test sync run

lint:
	ruff check src tests run.py
	mypy src tests run.py

test:
	pytest tests/ -v

sync:
	uv sync

run:
	uv run ai-report
