"""Paper-faithful LGBO glue code.

This module keeps the public runners on the actual LGBO path:

LLM point/region -> decide.py preference adapter -> boo.py region-lifted BO
proposal -> denormalized experimental batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from boo import propose_points_from_plan
from decide import decide_preference_cola_from_expert, decide_preference_tilt_from_expert
from prior_monte_carlo import WeightedQLogEI


@dataclass
class ContinuousSpace:
    names: list[str]
    bounds: list[tuple[float, float]]

    @classmethod
    def from_parameters(cls, parameters: Sequence[Mapping[str, Any]]) -> "ContinuousSpace":
        names: list[str] = []
        bounds: list[tuple[float, float]] = []
        for param in parameters:
            if str(param.get("type", "continuous")).lower() != "continuous":
                raise ValueError("public LGBO runners currently require continuous parameters")
            name = str(param.get("name", "")).strip()
            if not name:
                raise ValueError("every parameter needs a name")
            raw_bounds = param.get("bounds")
            if not isinstance(raw_bounds, Sequence) or len(raw_bounds) != 2:
                raise ValueError(f"parameter {name!r} requires bounds=[lo, hi]")
            lo, hi = float(raw_bounds[0]), float(raw_bounds[1])
            if lo >= hi:
                raise ValueError(f"parameter {name!r} has invalid bounds")
            names.append(name)
            bounds.append((lo, hi))
        if not names:
            raise ValueError("at least one parameter is required")
        return cls(names=names, bounds=bounds)

    @property
    def d(self) -> int:
        return len(self.names)

    def normalize_vector(self, x: Sequence[Any]) -> list[float]:
        if len(x) != self.d:
            raise ValueError(f"dimension mismatch: got {len(x)} but d={self.d}")
        out = []
        for value, (lo, hi) in zip(x, self.bounds):
            v = min(max(float(value), lo), hi)
            out.append((v - lo) / (hi - lo))
        return out

    def denormalize_vector(self, z: Sequence[Any]) -> list[float]:
        if len(z) != self.d:
            raise ValueError(f"dimension mismatch: got {len(z)} but d={self.d}")
        out = []
        for value, (lo, hi) in zip(z, self.bounds):
            v = min(max(float(value), 0.0), 1.0)
            out.append(lo + v * (hi - lo))
        return out

    def normalize_point(self, point: Mapping[str, Any] | Sequence[Any]) -> list[float]:
        if isinstance(point, Mapping):
            return self.normalize_vector([point[name] for name in self.names])
        return self.normalize_vector(point)

    def denormalize_point(self, z: Sequence[Any]) -> dict[str, float]:
        values = self.denormalize_vector(z)
        return {name: value for name, value in zip(self.names, values)}

    def normalize_region(self, lb: Sequence[Any], ub: Sequence[Any]) -> tuple[list[float], list[float]]:
        lb_n = self.normalize_vector(lb)
        ub_n = self.normalize_vector(ub)
        return [min(a, b) for a, b in zip(lb_n, ub_n)], [max(a, b) for a, b in zip(lb_n, ub_n)]


def safe_confidence(value: Any) -> float:
    try:
        conf = float(value)
    except Exception:
        conf = 0.5
    return min(max(conf, 1e-4), 1.0 - 1e-4)


def assistant_preference_to_expert(parsed: Mapping[str, Any], space: ContinuousSpace) -> list[Any] | None:
    mode = parsed.get("mode")
    conf = safe_confidence(parsed.get("confidence", 0.5))
    if mode == "point" and parsed.get("point") is not None:
        return ["point", space.normalize_vector(parsed["point"]), conf]
    if mode == "region" and parsed.get("lb") is not None and parsed.get("ub") is not None:
        lb_n, ub_n = space.normalize_region(parsed["lb"], parsed["ub"])
        return ["region", [lb_n, ub_n], conf]
    return None


def make_plan_from_preference(
    parsed: Mapping[str, Any],
    space: ContinuousSpace,
    *,
    policy: str = "tilt",
    grid_size: int = 512,
    guidance_scale: float = 3.0,
) -> dict[str, Any]:
    expert = assistant_preference_to_expert(parsed, space)
    if expert is None:
        return {"mode": "none", "confidence": 0.0, "why": "no valid LLM point/region preference"}

    policy_name = policy.strip().lower()
    if policy_name == "cola":
        return decide_preference_cola_from_expert(
            expert,
            d=space.d,
            grid_size=grid_size,
            guidance_scale=guidance_scale,
        )
    if policy_name != "tilt":
        raise ValueError("policy must be 'tilt' or 'cola'")
    return decide_preference_tilt_from_expert(
        expert,
        d=space.d,
        grid_size=grid_size,
        guidance_scale=guidance_scale,
    )


def propose_lgbo_batch(
    *,
    X_norm: torch.Tensor,
    y: torch.Tensor,
    parsed_preference: Mapping[str, Any],
    space: ContinuousSpace,
    goal: str,
    batch_q: int,
    policy: str = "tilt",
    grid_size: int = 512,
    guidance_scale: float = 3.0,
    cand_size: int = 8192,
    num_paths_batch: int = 1024,
    seed: int = 0,
    dtype: torch.dtype = torch.double,
    device: torch.device | None = None,
) -> tuple[list[dict[str, float]], dict[str, Any], torch.Tensor]:
    """Run one LGBO acquisition step and return physical-coordinate points."""
    if X_norm.ndim != 2 or X_norm.shape[1] != space.d:
        raise ValueError("X_norm must have shape [N, d]")
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    if y.ndim != 2 or y.shape[0] != X_norm.shape[0]:
        raise ValueError("y must have shape [N, 1] and align with X_norm")

    torch.manual_seed(seed)
    device = device or torch.device("cpu")
    X_norm = X_norm.to(device=device, dtype=dtype).clamp(1e-6, 1.0 - 1e-6)
    y = y.to(device=device, dtype=dtype)
    if goal.strip().lower() == "min":
        y_for_bo = -y
    elif goal.strip().lower() == "max":
        y_for_bo = y
    else:
        raise ValueError("goal must be 'min' or 'max'")

    plan = make_plan_from_preference(
        parsed_preference,
        space,
        policy=policy,
        grid_size=grid_size,
        guidance_scale=guidance_scale,
    )
    Z_new = propose_points_from_plan(
        sampler_cls=WeightedQLogEI,
        X=X_norm,
        y=y_for_bo,
        plan=plan,
        q=max(1, int(batch_q)),
        cand_size=int(cand_size),
        num_paths_batch=int(num_paths_batch),
        device=device,
        dtype=dtype,
    )
    if Z_new.ndim == 1:
        Z_new = Z_new.unsqueeze(0)
    points = [space.denormalize_point(z.detach().cpu().tolist()) for z in Z_new]
    return points, plan, Z_new.detach().cpu()


def serialize_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Make a plan JSON-friendly without depending on torch repr strings."""
    def convert(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, Mapping):
            return {str(k): convert(v) for k, v in value.items() if k != "why"}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    return convert(plan)


def tensor_from_observations(
    observations: Sequence[Mapping[str, Any]],
    space: ContinuousSpace,
    *,
    y_key: str = "y",
    dtype: torch.dtype = torch.double,
) -> tuple[torch.Tensor, torch.Tensor]:
    X_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    for obs in observations:
        x = obs.get("x", obs.get("point"))
        if x is None:
            continue
        if y_key not in obs and "y" not in obs:
            continue
        y_value = obs.get(y_key, obs.get("y"))
        X_rows.append(space.normalize_point(x))
        y_rows.append([float(y_value)])
    if len(X_rows) < 2:
        raise ValueError("LGBO needs at least two observations with x and numeric y")
    return torch.tensor(X_rows, dtype=dtype), torch.tensor(y_rows, dtype=dtype)
