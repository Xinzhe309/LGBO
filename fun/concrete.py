"""Concrete-strength objective backed by the processed UCI measurements."""

from __future__ import annotations

from pathlib import Path

from fun.tabular import TabularEvaluator, normalizer_from_csv, print_evaluator_demo


DEFAULT_CSV_PATH = Path(__file__).with_name("concrete.csv")
PARAM_NAMES = ["Cement", "Blast", "Fly", "Water", "Superplas", "CoarseAg", "FineAggr"]
METRIC_KEY = "Strength"


class ConcreteEvaluator(TabularEvaluator):
    def __init__(self, csv_path: str | Path = DEFAULT_CSV_PATH, **kwargs):
        kwargs.setdefault("try_regular_grid", False)
        super().__init__(csv_path, PARAM_NAMES, METRIC_KEY, **kwargs)


def make_bo_normalizer_concrete(csv_path=DEFAULT_CSV_PATH, bounds_override=None):
    return normalizer_from_csv(csv_path, PARAM_NAMES, bounds_override)


if __name__ == "__main__":
    print_evaluator_demo(ConcreteEvaluator())
