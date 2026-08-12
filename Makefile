.PHONY: install test test-all lint format typecheck check demo figures clean

install:
	python -m pip install -e ".[dev,data]"

test:
	pytest -m "not slow and not network"

test-all:
	pytest --cov --cov-report=term-missing

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

typecheck:
	mypy

check: lint typecheck test

demo:
	luckdet demo

# The README's two figures, drawn from the same real prices `luckdet demo` uses
# and committed to docs/figures/ because outputs/ is gitignored.
figures:
	python scripts/make_readme_figures.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist
	rm -f logs_*.zip
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
