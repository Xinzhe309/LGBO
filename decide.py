# -*- coding: utf-8 -*-
from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

ExpertInput = Union[List[Any], Tuple[Any, ...], str]
PrefKind = str


def _normalize_expert_str(s: str) -> str:
    """Convert a permissive expert string into JSON-friendly syntax."""
    s = s.strip()
    s = re.sub(r"^\s*\[\s*([A-Za-z]+)\s*,", r'["\1",', s)
    s = re.sub(r"([\{\s,])([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', s)
    return s.replace("'", '"')


def _loads_forgiving(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return ast.literal_eval(s)


def _norm_ppf(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.sqrt(2.0) * float(torch.erfinv(torch.tensor(2.0 * p - 1.0)))


def confidence_to_delta(
    confidence: float,
    scale: float = 1.0,
    two_sided: bool = False,
) -> float:
    p = float(confidence)
    if two_sided:
        p = 0.5 * (1.0 + p)
    return scale * _norm_ppf(p)


def _box_from_center_radius(
    center: torch.Tensor,
    radius: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lb = (center - radius).clamp(1e-6, 1 - 1e-6)
    ub = (center + radius).clamp(1e-6, 1 - 1e-6)
    return lb, ub


def decide_preference(
    *,
    kind: PrefKind,
    confidence: float,
    d: int,
    grid_size: int = 512,
    guidance_scale: float = 3.0,
    region_box: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    region_center: Optional[torch.Tensor] = None,
    region_radius: Optional[float] = None,
    x_star: Optional[torch.Tensor] = None,
    y_star: Optional[float] = None,
    E_soft_low: float = 2.0,
    E_soft_high: float = 6.0,
) -> Dict[str, Any]:
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1).")

    plan: Dict[str, Any] = {"kind_input": kind}
    delta = confidence_to_delta(confidence, scale=float(guidance_scale), two_sided=False)
    plan["delta"] = float(delta)
    plan["guidance_scale"] = float(guidance_scale)

    if kind == "region":
        if region_box is None and (region_center is None or region_radius is None):
            raise ValueError("region mode requires region_box or (region_center, region_radius).")
        if region_box is None:
            lb, ub = _box_from_center_radius(region_center, float(region_radius))
        else:
            lb, ub = region_box
        lb = lb.clone().detach()
        ub = ub.clone().detach()
        if lb.shape != (d,) or ub.shape != (d,):
            raise ValueError(f"region bounds must both have shape ({d},).")

        vol = float(torch.clamp(ub - lb, 0, 1).prod().item())
        expected_hits = grid_size * vol

        if expected_hits < E_soft_low:
            center = 0.5 * (lb + ub)
            plan.update(
                {
                    "mode": "point",
                    "x_star": center,
                    "y_star": y_star,
                    "why": (
                        f"region too small for grid_size={grid_size} "
                        f"(E={expected_hits:.2f} < {E_soft_low}); fallback to point."
                    ),
                }
            )
        elif expected_hits < E_soft_high:
            plan.update(
                {
                    "mode": "region-soft",
                    "region": {"lb": lb, "ub": ub, "grid_size": grid_size, "smooth": 0.08},
                    "x_star": 0.5 * (lb + ub),
                    "y_star": y_star,
                    "why": (
                        f"medium region density E={expected_hits:.2f} in "
                        f"[{E_soft_low}, {E_soft_high}); use soft box."
                    ),
                }
            )
        else:
            plan.update(
                {
                    "mode": "region",
                    "region": {"lb": lb, "ub": ub, "grid_size": grid_size, "smooth": None},
                    "x_star": 0.5 * (lb + ub),
                    "y_star": y_star,
                    "why": f"region well-covered: E={expected_hits:.2f} >= {E_soft_high}.",
                }
            )
        return plan

    if kind == "value":
        if y_star is None:
            raise ValueError("value mode requires y_star.")
        plan.update(
            {
                "mode": "value",
                "y_star": y_star,
                "why": "value prior selected; confidence controls prior strength.",
            }
        )
        return plan

    if kind == "point":
        if x_star is None:
            raise ValueError("point mode requires x_star in the normalized domain.")
        plan.update(
            {
                "mode": "point",
                "x_star": x_star,
                "why": "point prior selected; confidence controls prior strength.",
            }
        )
        return plan

    raise ValueError(f"unknown preference kind: {kind}")


def _to_float(x: Any) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        return float(x.strip())
    raise ValueError(f"cannot convert to float: {x!r}")


def _to_1d_tensor(items: Any, *, dtype=torch.double) -> torch.Tensor:
    if isinstance(items, torch.Tensor):
        tensor = items.detach().to(dtype=dtype)
    elif isinstance(items, (list, tuple)):
        tensor = torch.tensor([_to_float(v) for v in items], dtype=dtype)
    elif isinstance(items, str):
        tensor = torch.tensor([_to_float(v) for v in items.split(",")], dtype=dtype)
    else:
        raise ValueError(f"expected a 1D vector as list, tuple, tensor, or string; got {type(items)}")

    if tensor.ndim != 1:
        raise ValueError("expected a 1D vector.")
    return tensor.clamp(1e-6, 1 - 1e-6)


def parse_expert_input(
    expert: ExpertInput,
    *,
    d: Optional[int] = None,
    grid_size: int = 512,
    guidance_scale: float = 3.0,
    E_soft_low: float = 2.0,
    E_soft_high: float = 6.0,
    dtype: torch.dtype = torch.double,
) -> Dict[str, Any]:
    """Parse [kind, payload, confidence] into keyword arguments for decide_preference."""
    if not isinstance(expert, (list, tuple)) or len(expert) != 3:
        raise ValueError("expert input must be [kind, payload, confidence].")

    kind, payload, confidence = expert
    kind = str(kind).lower().strip()
    conf = _to_float(confidence)
    if not (0.0 < conf < 1.0):
        raise ValueError("confidence must be in (0, 1).")

    kwargs: Dict[str, Any] = {
        "kind": kind,
        "confidence": conf,
        "grid_size": int(grid_size),
        "guidance_scale": float(guidance_scale),
        "E_soft_low": float(E_soft_low),
        "E_soft_high": float(E_soft_high),
    }

    if kind == "point":
        x_star = _to_1d_tensor(payload, dtype=dtype)
        dim = len(x_star) if d is None else d
        if d is not None and len(x_star) != d:
            raise ValueError(f"x_star length {len(x_star)} != d {d}.")
        kwargs.update({"d": int(dim), "x_star": x_star})
        return kwargs

    if kind == "value":
        dim = int(d) if d is not None else 1
        kwargs.update({"d": dim, "y_star": float(_to_float(payload))})
        return kwargs

    if kind == "region":
        lb = ub = center = None
        radius = None

        if isinstance(payload, dict):
            if "lb" in payload and "ub" in payload:
                lb = _to_1d_tensor(payload["lb"], dtype=dtype)
                ub = _to_1d_tensor(payload["ub"], dtype=dtype)
                if len(lb) != len(ub):
                    raise ValueError("lb and ub must have the same length.")
            elif "center" in payload and "radius" in payload:
                center = _to_1d_tensor(payload["center"], dtype=dtype)
                radius = _to_float(payload["radius"])
            else:
                raise ValueError("region dict must contain ('lb', 'ub') or ('center', 'radius').")
        elif isinstance(payload, (list, tuple)) and len(payload) == 2:
            a, b = payload
            if isinstance(a, (list, tuple, str)) and isinstance(b, (list, tuple, str)):
                lb = _to_1d_tensor(a, dtype=dtype)
                ub = _to_1d_tensor(b, dtype=dtype)
                if len(lb) != len(ub):
                    raise ValueError("lb and ub must have the same length.")
            elif isinstance(a, (list, tuple, str)) and isinstance(b, (int, float, str)):
                center = _to_1d_tensor(a, dtype=dtype)
                radius = _to_float(b)
            else:
                raise ValueError("unrecognized region shorthand.")
        else:
            raise ValueError("region payload must be a dict or a two-item sequence.")

        if lb is not None and ub is not None:
            dim = len(lb) if d is None else d
            if d is not None and len(lb) != d:
                raise ValueError(f"lb/ub length {len(lb)} != d {d}.")
            kwargs.update({"d": int(dim), "region_box": (lb, ub)})
            return kwargs

        if center is not None and radius is not None:
            if radius <= 0:
                raise ValueError("radius must be positive.")
            dim = len(center) if d is None else d
            if d is not None and len(center) != d:
                raise ValueError(f"center length {len(center)} != d {d}.")
            kwargs.update({"d": int(dim), "region_center": center, "region_radius": float(radius)})
            return kwargs

        raise RuntimeError("internal region parsing failure.")

    raise ValueError(f"unknown preference kind: {kind}")


def parse_expert_input_auto(expert: ExpertInput, **kwargs_for_parse) -> Dict[str, Any]:
    """Accept either a structured expert object or a permissive string."""
    if isinstance(expert, str):
        expert_obj = _loads_forgiving(_normalize_expert_str(expert))
    else:
        expert_obj = expert
    return parse_expert_input(expert_obj, **kwargs_for_parse)


def _box_width_with_clamp(center_i: float, r: float, eps: float = 1e-6) -> float:
    lb = max(center_i - r, eps)
    ub = min(center_i + r, 1.0 - eps)
    return max(ub - lb, 0.0)


def _effective_volume(center: torch.Tensor, r: float, eps: float = 1e-6) -> float:
    width = 1.0
    for value in center.detach().double().cpu().tolist():
        width *= _box_width_with_clamp(float(value), r, eps)
    return max(width, 0.0)


def choose_soft_radius_edge(
    d: int,
    grid_size: int,
    *,
    center: torch.Tensor,
    E_soft_low: float = 2.0,
    low_margin: float = 0.05,
    eps: float = 1e-6,
) -> float:
    """Choose a radius whose clipped box is just large enough for soft-region mode."""
    _ = d
    target_hits = E_soft_low + float(low_margin)
    target_volume = max(target_hits / float(grid_size), 1e-12)

    lo, hi = 1e-8, 0.5 - eps
    if _effective_volume(center, hi, eps) < target_volume:
        return float(hi)

    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if _effective_volume(center, mid, eps) >= target_volume:
            hi = mid
        else:
            lo = mid

    r = 0.5 * (lo + hi)
    effective_hits = grid_size * _effective_volume(center, r, eps)
    if effective_hits < E_soft_low:
        for _ in range(10):
            r = min(r * 1.2, 0.5 - eps)
            effective_hits = grid_size * _effective_volume(center, r, eps)
            if effective_hits >= E_soft_low:
                break
    return float(r)


def decide_preference_tilt_from_expert(
    expert: ExpertInput,
    *,
    d: Optional[int] = None,
    grid_size: int = 512,
    guidance_scale: float = 3.0,
    E_soft_low: float = 2.0,
    E_soft_high: float = 6.0,
    dtype: torch.dtype = torch.double,
    low_margin: float = 0.05,
    r_cap: Optional[float] = None,
) -> Dict[str, Any]:
    kw = parse_expert_input_auto(
        expert,
        d=d,
        grid_size=grid_size,
        guidance_scale=guidance_scale,
        E_soft_low=E_soft_low,
        E_soft_high=E_soft_high,
        dtype=dtype,
    )
    if kw["kind"] == "point":
        x = kw["x_star"]
        dim = kw["d"]
        radius = choose_soft_radius_edge(
            dim,
            grid_size,
            center=x,
            E_soft_low=E_soft_low,
            low_margin=low_margin,
        )
        if r_cap is not None:
            radius = min(radius, float(r_cap))
        return decide_preference(
            kind="region",
            confidence=kw["confidence"],
            d=dim,
            grid_size=grid_size,
            guidance_scale=guidance_scale,
            region_center=x,
            region_radius=radius,
            E_soft_low=E_soft_low,
            E_soft_high=E_soft_high,
        )
    return decide_preference(**kw)


def decide_preference_cola_from_expert(
    expert: ExpertInput,
    *,
    d: Optional[int] = None,
    grid_size: int = 512,
    guidance_scale: float = 3.0,
    E_soft_low: float = 2.0,
    E_soft_high: float = 6.0,
    dtype: torch.dtype = torch.double,
) -> Dict[str, Any]:
    """COLA adapter: reduce regions to their center point, keep points and values."""
    kw = parse_expert_input_auto(
        expert,
        d=d,
        grid_size=grid_size,
        guidance_scale=guidance_scale,
        E_soft_low=E_soft_low,
        E_soft_high=E_soft_high,
        dtype=dtype,
    )
    if kw["kind"] == "region":
        dim = kw["d"]
        if "region_box" in kw:
            lb, ub = kw["region_box"]
            center = 0.5 * (lb + ub)
        else:
            center = kw["region_center"]
        return decide_preference(
            kind="point",
            confidence=kw["confidence"],
            d=dim,
            x_star=center,
            grid_size=grid_size,
            guidance_scale=guidance_scale,
            E_soft_low=E_soft_low,
            E_soft_high=E_soft_high,
        )
    return decide_preference(**kw)


def decide_preference__from_expert(
    expert: ExpertInput,
    *,
    d: Optional[int] = None,
    grid_size: int = 512,
    guidance_scale: float = 3.0,
    E_soft_low: float = 2.0,
    E_soft_high: float = 6.0,
    dtype: torch.dtype = torch.double,
    low_margin: float = 0.05,
    r_cap: Optional[float] = None,
) -> Dict[str, Any]:
    """Pass parsed expert preferences directly to decide_preference."""
    _ = (low_margin, r_cap)
    kw = parse_expert_input_auto(
        expert,
        d=d,
        grid_size=grid_size,
        guidance_scale=guidance_scale,
        E_soft_low=E_soft_low,
        E_soft_high=E_soft_high,
        dtype=dtype,
    )
    return decide_preference(**kw)
