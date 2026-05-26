"""Classic toy functions used by the dry LGBO runner."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union


Vec = Sequence[float]
Batch = Sequence[Vec]
Result = Dict[str, float]
BatchResult = List[Result]

DEFAULT_BOUNDS = {
    "rastrigin": (-5.12, 5.12),
    "ackley": (-5.0, 5.0),
    "griewank": (-600.0, 600.0),
    "levy": (-10.0, 10.0),
}


def _is_batch(x: Union[Vec, Batch]) -> bool:
    return isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple))


def _clip_to_bounds(x: Vec, bounds: Tuple[float, float]) -> list[float]:
    lo, hi = bounds
    return [min(max(float(v), lo), hi) for v in x]


def rastrigin(x: Vec) -> float:
    d = len(x)
    a = 10.0
    return a * d + sum(xi * xi - a * math.cos(2.0 * math.pi * xi) for xi in x)


def ackley(x: Vec) -> float:
    d = len(x)
    if d == 0:
        return 0.0
    s1 = sum(xi * xi for xi in x)
    s2 = sum(math.cos(2.0 * math.pi * xi) for xi in x)
    return -20.0 * math.exp(-0.2 * math.sqrt(s1 / d)) - math.exp(s2 / d) + 20.0 + math.e


def griewank(x: Vec) -> float:
    sum_term = sum(xi * xi for xi in x) / 4000.0
    prod_term = 1.0
    for i, xi in enumerate(x, start=1):
        prod_term *= math.cos(xi / math.sqrt(float(i)))
    return 1.0 + sum_term - prod_term


def levy(x: Vec) -> float:
    d = len(x)
    if d == 0:
        return 0.0
    w = [1.0 + (xi - 1.0) / 4.0 for xi in x]
    value = math.sin(math.pi * w[0]) ** 2
    for wi in w[:-1]:
        value += (wi - 1.0) ** 2 * (1.0 + 10.0 * math.sin(math.pi * wi + 1.0) ** 2)
    value += (w[-1] - 1.0) ** 2 * (1.0 + math.sin(2.0 * math.pi * w[-1]) ** 2)
    return value


def toy_results(
    func_name: str,
    x: Union[Vec, Batch],
    *,
    noise_std: float = 0.0,
    seed: int | None = None,
    clip_to_domain: bool = True,
    bounds: Tuple[float, float] | None = None,
) -> Union[Result, BatchResult]:
    name = func_name.strip().lower()
    if name not in DEFAULT_BOUNDS:
        raise ValueError("func_name must be one of: rastrigin, ackley, griewank, levy")

    rng = random.Random(seed)
    f_map = {
        "rastrigin": rastrigin,
        "ackley": ackley,
        "griewank": griewank,
        "levy": levy,
    }
    f_eval = f_map[name]
    dom_bounds = bounds if bounds is not None else DEFAULT_BOUNDS[name]

    def eval_one(xi: Vec) -> Result:
        x_eff = _clip_to_bounds(xi, dom_bounds) if clip_to_domain else [float(v) for v in xi]
        value = float(f_eval(x_eff))
        if noise_std > 0.0:
            value += rng.gauss(0.0, noise_std)
        return {"f": value}

    if _is_batch(x):
        return [eval_one(xi) for xi in x]  # type: ignore[arg-type]
    return eval_one(x)  # type: ignore[arg-type]


def get_bo_bounds(func_name: str, d: int) -> list[tuple[float, float]]:
    name = func_name.strip().lower()
    if name not in DEFAULT_BOUNDS:
        raise ValueError("func_name must be one of: rastrigin, ackley, griewank, levy")
    lo, hi = DEFAULT_BOUNDS[name]
    return [(lo, hi)] * int(d)


@dataclass
class BONormalizer:
    bounds: list[tuple[float, float]]

    def __post_init__(self) -> None:
        self.bounds = [(float(lo), float(hi)) for lo, hi in self.bounds]
        if any(lo >= hi for lo, hi in self.bounds):
            raise ValueError("invalid bounds: require lo < hi")
        self.d = len(self.bounds)

    def normalize_point(self, x: Sequence[float]) -> list[float]:
        if len(x) != self.d:
            raise ValueError(f"dimension mismatch: got {len(x)} but d={self.d}")
        out = []
        for (lo, hi), xi in zip(self.bounds, x):
            value = min(max(float(xi), lo), hi)
            out.append((value - lo) / (hi - lo))
        return out

    def denormalize_point(self, z: Sequence[float]) -> list[float]:
        if len(z) != self.d:
            raise ValueError(f"dimension mismatch: got {len(z)} but d={self.d}")
        out = []
        for (lo, hi), zi in zip(self.bounds, z):
            value = min(max(float(zi), 0.0), 1.0)
            out.append(lo + value * (hi - lo))
        return out

    def normalize_region(self, lb: Sequence[float], ub: Sequence[float]) -> tuple[list[float], list[float]]:
        lb_n = self.normalize_point(lb)
        ub_n = self.normalize_point(ub)
        return [min(a, b) for a, b in zip(lb_n, ub_n)], [max(a, b) for a, b in zip(lb_n, ub_n)]

    def denormalize_region(self, lb_n: Sequence[float], ub_n: Sequence[float]) -> tuple[list[float], list[float]]:
        lb = self.denormalize_point(lb_n)
        ub = self.denormalize_point(ub_n)
        return [min(a, b) for a, b in zip(lb, ub)], [max(a, b) for a, b in zip(lb, ub)]


def make_bo_normalizer(
    func_name: str,
    d: int,
    bounds_override: Optional[list[tuple[float, float]]] = None,
) -> BONormalizer:
    return BONormalizer(bounds_override if bounds_override is not None else get_bo_bounds(func_name, d))


if __name__ == "__main__":
    print("ackley:", toy_results("ackley", [[0.0, 0.0], [1.0, 1.0]]))
