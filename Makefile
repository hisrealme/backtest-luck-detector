.PHONY: install test test-all lint format typecheck check demo clean

install:
	python -m pip install -e ".[dev,data]"

test:
	pytest -m "not slow and not network"

test-all:
	pytest --cov --cov-report=term-missing

lint:
	ruff check src tests

format:
	ruff format src tests
	ruff check --fix src tests

typecheck:
	mypy

check: lint typecheck test

demo:
	luckdet demo

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist
	find . -type d -name __pycache__ -exec rm -rf {} +
