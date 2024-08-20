.PHONY: install test run clean generate-data

install:
	poetry install

test:
	poetry run pytest tests/

run:
	poetry run python main.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

generate-data:
	poetry run python scripts/generate_random_weights.py
	poetry run python scripts/generate_cifar10_subset.py

.DEFAULT_GOAL := help
help:
	@echo "Available commands:"
	@echo "  make install      : Install project dependencies"
	@echo "  make test         : Run tests"
	@echo "  make run          : Run the main script"
	@echo "  make clean        : Remove Python cache files"
	@echo "  make generate-data: Generate random weights and CIFAR-10 subset"
