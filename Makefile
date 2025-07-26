.PHONY: help install install-dev test test-fast lint format type-check clean setup-pre-commit run-examples

# Default target
help:
	@echo "Available commands:"
	@echo "  install           Install production dependencies"
	@echo "  install-dev       Install development dependencies"
	@echo "  test              Run all tests"
	@echo "  test-fast         Run fast tests only (skip slow/integration)"
	@echo "  lint              Run linting (ruff)"
	@echo "  format            Format code (black + ruff)"
	@echo "  type-check        Run type checking (mypy)"
	@echo "  clean             Clean up build artifacts"
	@echo "  setup-pre-commit  Install pre-commit hooks"
	@echo "  run-examples      Run all example scripts"
	@echo "  template-copy     Copy template to specified directory"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

# Testing
test:
	pytest tests/ -v --cov=chains --cov=tools --cov=memory --cov=utils

test-fast:
	pytest tests/ -v -m "not slow" --cov=chains --cov=tools --cov=memory --cov=utils

test-integration:
	pytest tests/ -v -m "integration"

test-examples:
	pytest examples/tests/ -v

# Code quality
lint:
	ruff check .

lint-fix:
	ruff check . --fix

format:
	black .
	ruff check . --fix

type-check:
	mypy chains/ tools/ memory/ utils/

# Quality check all
quality: lint type-check test-fast

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/

# Pre-commit
setup-pre-commit:
	pre-commit install

run-pre-commit:
	pre-commit run --all-files

# Examples
run-examples:
	@echo "Running basic chain example..."
	python examples/basic_chain.py
	@echo "Running RAG chain example..."
	python examples/rag_chain.py
	@echo "Running agent chain example..."
	python examples/agent_chain.py
	@echo "Running memory chain example..."
	python examples/memory_chain.py

# Template utilities
template-copy:
	@if [ -z "$(DIR)" ]; then \
		echo "Usage: make template-copy DIR=target-directory"; \
		exit 1; \
	fi
	python copy_template.py $(DIR)

template-copy-force:
	@if [ -z "$(DIR)" ]; then \
		echo "Usage: make template-copy-force DIR=target-directory"; \
		exit 1; \
	fi
	python copy_template.py $(DIR) --force

# Development workflow
dev-setup: install-dev setup-pre-commit
	@echo "Development environment setup complete!"
	@echo "Don't forget to:"
	@echo "  1. Copy .env.example to .env and add your API keys"
	@echo "  2. Run 'make test-fast' to verify setup"

# Full development check
dev-check: format lint type-check test-fast
	@echo "All development checks passed!"

# Documentation
docs-serve:
	@echo "Documentation available in docs/ directory"
	@echo "Consider setting up mkdocs or sphinx for documentation serving"

# LangChain specific commands
langchain-version:
	python -c "import langchain; print(f'LangChain {langchain.__version__} installed')"

validate-env:
	python -c "from utils.settings import validate_environment; print(validate_environment())"

# Docker support (optional)
docker-build:
	docker build -t langchain-context-engineering .

docker-run:
	docker run -it --rm langchain-context-engineering