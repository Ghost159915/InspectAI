# InspectAI — Makefile
# Works on Mac and Linux. On Windows, use 'docker compose' commands directly
# or run the Python commands manually.

.PHONY: help install run test lint format docker-up docker-down clean

help:
	@echo ""
	@echo "InspectAI — available commands:"
	@echo "  make install     Install Python dependencies"
	@echo "  make run         Run the Gradio app locally"
	@echo "  make test        Run the test suite with coverage"
	@echo "  make lint        Check code style with ruff"
	@echo "  make format      Auto-format code with ruff"
	@echo "  make docker-up   Start via Docker Compose"
	@echo "  make docker-down Stop Docker containers"
	@echo "  make clean       Remove cache and temp files"
	@echo ""

install:
	pip install -r requirements.txt

run:
	python main.py

test:
	pytest tests/ -v --cov=app --cov-report=term-missing

lint:
	ruff check app/ tests/

format:
	ruff format app/ tests/
	ruff check --fix app/ tests/

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned."
