.PHONY: install test lint typecheck format notebooks clean

install:
	poetry install

test:
	poetry run pytest -q

lint:
	poetry run ruff check src scripts tests

typecheck:
	poetry run mypy src scripts

format:
	poetry run ruff format src scripts tests

notebooks:
	poetry run jupyter lab notebooks

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
