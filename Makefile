# Makefile for Sentiment Analysis
.PHONY: help install test lint format clean run

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	uv sync --all-extras

test: ## Run tests with coverage
	uv run pytest tests/ --cov=src --cov=app --cov-report=term

lint: ## Check and fix code quality
	uv run ruff check --fix ./src ./app ./tests
	uv run ruff format ./src ./app ./tests

format: ## Format code only
	uv run ruff format ./src ./app ./tests

run: ## Run the Dash application
	uv run python app/voice_to_text_app.py

clean: ## Remove temporary files
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/ 