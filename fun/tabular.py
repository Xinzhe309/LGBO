"""Shared interpolation utilities for the tabular LGBO benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from fun.toy_fun import BONormalizer


class RegularGridNDInterpolator:
    """Small dependency-free multilinear interpolator with clamped boundaries."""

    def __init__(self, grids: list[np.ndarray], values: np.ndarray):
        self.grids = [np.asarray(grid, dtype=float) for grid in grids]
        self.values = np.asarray(values, dtype=float)
        self.ndim = len(self.grids)
        expected = tuple(len(grid) for grid in self.grids)
        if self.values.shape != expected:
            raise ValueError(f"values shape {self.values.shape} != grid shape {expected}")
        if any(len(grid) < 2 or np.any(np.diff(grid) <= 0) for grid in self.grids):
            raise ValueError("each grid must contain at least two increasing values")

    @staticmethod
    def _locate(grid: np.ndarray, value: float) -> tuple[int, int, float]:
        if value <= grid[0]:
            return 0, 1, 0.0
        if value >= grid[-1]:
            return len(grid) - 2, len(grid) - 1, 1.0
        upper = int(np.searchsorted(grid, value, side="right"))
        lower = upper - 1
        weight = (value - grid[lower]) / (grid[upper] - grid[lower])
        return lower, upper, float(weight)

    def __call__(self, point: Sequence[float]) -> float:
        if len(point) != self.ndim:
            raise ValueError(f"expected {self.ndim} values; got {len(point)}")
        locations = [self._locate(grid, float(value)) for grid, value in zip(self.grids, point)]
        total = 0.0
        for mask in range(1 << self.ndim):
            weight = 1.0
            index = []
            for dimension, (lower, upper, fraction) in enumerate(locations):
                if (mask >> dimension) & 1:
                    weight *= fraction
                    index.append(upper)
                else:
                    weight *= 1.0 - fraction
                    index.append(lower)
            total += weight * self.values[tuple(index)]
        return float(total)


class TabularEvaluator:
    """Regular-grid interpolation with the archived kNN-IDW fallback."""

    def __init__(
        self,
        csv_path: str | Path,
        parameter_names: Sequence[str],
        metric_key: str,
        *,
        try_regular_grid: bool = True,
        max_grid_cells: int = 200_000,
        max_levels_per_dimension: int = 64,
        knn_k: int = 12,
        idw_power: float = 2.0,
        eps: float = 1e-12,
    ):
        self.csv_path = Path(csv_path)
        self.parameter_names = list(parameter_names)
        self.metric_key = metric_key
        self.knn_k = int(knn_k)
        self.idw_power = float(idw_power)
        self.eps = float(eps)

        required = self.parameter_names + [self.metric_key]
        frame = pd.read_csv(self.csv_path)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        for column in required:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=required).reset_index(drop=True)
        if frame.empty:
            raise ValueError("CSV contains no complete numeric rows")

        inputs = frame[self.parameter_names].to_numpy(dtype=float)
        self.y = frame[self.metric_key].to_numpy(dtype=float)
        self.lo = inputs.min(axis=0)
        upper = inputs.max(axis=0)
        self.span = np.maximum(upper - self.lo, self.eps)
        self.bounds = list(zip(self.lo.tolist(), upper.tolist()))
        self.X_scaled = (inputs - self.lo) / self.span
        self.use_grid = False
        self.interpolator: RegularGridNDInterpolator | None = None

        if try_regular_grid:
            levels = [np.sort(frame[column].unique().astype(float)) for column in self.parameter_names]
            shape = tuple(len(level) for level in levels)
            cells = int(np.prod(shape))
            feasible = (
                all(2 <= len(level) <= max_levels_per_dimension for level in levels)
                and cells <= max_grid_cells
            )
            if feasible:
                indexers = [{value: index for index, value in enumerate(level)} for level in levels]
                values = np.empty(shape, dtype=float)
                filled = np.zeros(shape, dtype=bool)
                for _, row in frame.iterrows():
                    index = tuple(
                        indexers[dimension][float(row[column])]
                        for dimension, column in enumerate(self.parameter_names)
                    )
                    values[index] = float(row[self.metric_key])
                    filled[index] = True
                if filled.all():
                    self.use_grid = True
                    self.interpolator = RegularGridNDInterpolator(levels, values)

    def _evaluate_one(self, point: Sequence[float]) -> float:
        if len(point) != len(self.parameter_names):
            raise ValueError(
                f"expected {len(self.parameter_names)} values in order {self.parameter_names}"
            )
        values = np.asarray(point, dtype=float)
        if self.use_grid:
            assert self.interpolator is not None
            return self.interpolator(values)

        scaled = (values - self.lo) / self.span
        distances = np.sqrt(np.sum((self.X_scaled - scaled[None, :]) ** 2, axis=1))
        nearest = int(np.argmin(distances))
        if distances[nearest] <= self.eps:
            return float(self.y[nearest])
        count = min(self.knn_k, len(distances))
        indices = np.argpartition(distances, count - 1)[:count]
        weights = 1.0 / np.power(distances[indices] + self.eps, self.idw_power)
        weights /= weights.sum()
        return float(np.sum(weights * self.y[indices]))

    def evaluate_from_list(self, point: Sequence[Any]) -> dict[str, float]:
        return {self.metric_key: self._evaluate_one([float(value) for value in point])}

    def evaluate_from_dict(self, parameters: dict[str, Any]) -> dict[str, float]:
        return self.evaluate_from_list([parameters[name] for name in self.parameter_names])

    def evaluate(self, points: Sequence[Any]) -> dict[str, float] | list[dict[str, float]]:
        if len(points) > 0 and isinstance(points[0], (list, tuple, np.ndarray)):
            return [self.evaluate_from_list(point) for point in points]
        return self.evaluate_from_list(points)


def normalizer_from_csv(
    csv_path: str | Path,
    parameter_names: Sequence[str],
    bounds_override: list[tuple[float, float]] | None = None,
) -> BONormalizer:
    if bounds_override is not None:
        if len(bounds_override) != len(parameter_names):
            raise ValueError(f"bounds_override must contain {len(parameter_names)} dimensions")
        return BONormalizer(bounds_override)
    frame = pd.read_csv(csv_path)
    bounds = []
    for column in parameter_names:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        bounds.append((float(values.min()), float(values.max())))
    return BONormalizer(bounds)


def print_evaluator_demo(evaluator: TabularEvaluator) -> None:
    frame = pd.read_csv(evaluator.csv_path)
    first = [float(frame.iloc[0][name]) for name in evaluator.parameter_names]
    midpoint = [(lower + upper) / 2.0 for lower, upper in evaluator.bounds]
    print("CSV:", evaluator.csv_path)
    print("bounds:", evaluator.bounds)
    print("interpolation:", "multilinear grid" if evaluator.use_grid else "kNN-IDW")
    print("first row:", evaluator.evaluate_from_list(first))
    print("midpoint:", evaluator.evaluate_from_list(midpoint))
