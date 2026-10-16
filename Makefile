.PHONY: help install test lint demo compare report clean

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies in editable mode"
	@echo "  make test       - Run test suite with pytest"
	@echo "  make demo       - Run prompt regression test scenario"
	@echo "  make migration  - Run model migration benchmark scenario"
	@echo "  make passk      - Run Pass@k probabilistic evaluation scenario"
	@echo "  make bias       - Run LLM judge position bias mitigation scenario"
	@echo "  make report     - Generate HTML report from evaluation results"
	@echo "  make clean      - Clean cache and temporary files"

install:
	pip install -e ".[dev]"

test:
	pytest -v tests/

lint:
	ruff check src/ tests/ examples/

demo:
	python examples/01_prompt_regression_test.py

migration:
	python examples/02_model_migration_benchmark.py

passk:
	python examples/03_probabilistic_pass_at_k.py

bias:
	python examples/04_llm_judge_bias_mitigation.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
