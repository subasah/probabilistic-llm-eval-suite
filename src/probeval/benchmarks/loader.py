"""Benchmark dataset loader and suite parser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union
from probeval.config import TestCase


class BenchmarkLoader:
    """Loads benchmark test suites from JSONL or JSON files."""

    @staticmethod
    def load_jsonl(filepath: Union[str, Path]) -> List[TestCase]:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")

        test_cases: List[TestCase] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    data = json.loads(line)
                    test_cases.append(TestCase(**data))
                except Exception as err:
                    raise ValueError(f"Error parsing line {line_no} in {path}: {err}") from err

        return test_cases

    @staticmethod
    def load_suite(category_or_file: str, benchmarks_dir: Union[str, Path] = "benchmarks") -> List[TestCase]:
        """Convenience loader supporting standard category names or explicit file paths."""
        path = Path(category_or_file)
        if path.is_file():
            return BenchmarkLoader.load_jsonl(path)

        b_dir = Path(benchmarks_dir)
        candidates = [
            b_dir / f"{category_or_file}.jsonl",
            b_dir / category_or_file,
        ]
        for c in candidates:
            if c.is_file():
                return BenchmarkLoader.load_jsonl(c)

        raise FileNotFoundError(f"Could not locate benchmark '{category_or_file}' in '{benchmarks_dir}'")
