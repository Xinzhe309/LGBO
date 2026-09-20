"""HPLC objective backed by the downloaded Olympus measurements."""

from __future__ import annotations

from pathlib import Path

from fun.tabular import TabularEvaluator, normalizer_from_csv, print_evaluator_demo


DEFAULT_CSV_PATH = Path(__file__).with_name("hplc.csv")
PARAM_NAMES = [
    "sample_loop", "additional_volume", "tubing_volume", "sample_flow",
    "push_speed", "wait_time",
]
METRIC_KEY = "peak_area"


class HPLCEvaluator(TabularEvaluator):
    def __init__(self, csv_path: str | Path = DEFAULT_CSV_PATH, **kwargs):
        super().__init__(csv_path, PARAM_NAMES, METRIC_KEY, **kwargs)


def make_bo_normalizer(csv_path=DEFAULT_CSV_PATH, bounds_override=None):
    return normalizer_from_csv(csv_path, PARAM_NAMES, bounds_override)


if __name__ == "__main__":
    print_evaluator_demo(HPLCEvaluator())
