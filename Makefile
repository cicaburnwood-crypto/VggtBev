.PHONY: check test lint

check: lint test

lint:
	.venv/bin/ruff check .

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest

