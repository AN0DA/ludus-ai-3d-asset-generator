.PHONY: lint mypy test

all: lint mypy test

lint:
	uv run ruff format src/ test/ \
	&& uv run ruff check --fix --show-fixes src/ test/ \
	&& uv run bandit -c pyproject.toml -r src/

mypy:
	uv run mypy src/ test/

test:
	uv run pytest --cov --cov-report term-missing:skip-covered