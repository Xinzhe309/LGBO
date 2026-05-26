from __future__ import annotations

from typing import Optional, Type

import torch
from torch import Tensor

from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from prior import (
    LinearExponentialRegionalMeanTiltPlugAndPlay,
    TiltedModel,
    _make_sobol_grid_norm,
)
from prior_monte_carlo import (
    PointBumpPrior,
    ValuePeakPrior,
    WeightedQEI,
    WeightedQLogEI,
    WeightedTS,
)


@torch.no_grad()
def _greedy_select_with_min_dist(
    acq_fn,
    C: Tensor,
    n: int,
    r_min: Optional[float],
    *,
    chunk_size: int = 256,
) -> Tensor:
    """Greedily select candidates while optionally enforcing a minimum distance."""
    num_candidates = C.size(0)
    device = C.device
    mask = torch.ones(num_candidates, dtype=torch.bool, device=device)
    selected_idx = []

    for _ in range(n):
        active = torch.arange(num_candidates, device=device)[mask]
        if active.numel() == 0:
            break

        best_val = None
        best_j_rel = None
        for start in range(0, active.numel(), chunk_size):
            end = min(start + chunk_size, active.numel())
            idx_chunk = active[start:end]
            vals = acq_fn(C[idx_chunk].unsqueeze(1)).view(-1)
            j_rel_chunk = int(vals.argmax().item())
            val_chunk = vals[j_rel_chunk]
            if best_val is None or val_chunk > best_val:
                best_val = val_chunk
                best_j_rel = start + j_rel_chunk

        j = int(active[best_j_rel].item())
        selected_idx.append(j)

        if r_min is None:
            mask[j] = False
        else:
            dists = torch.cdist(C[active], C[j : j + 1]).squeeze(-1)
            mask[active[dists < float(r_min)]] = False

    return C[torch.tensor(selected_idx, device=device)]


def propose_points_from_plan(
    sampler_cls: Type,
    X: Tensor,
    y: Tensor,
    plan: dict,
    q: Optional[int] = None,
    *,
    num_paths_single: int = 512,
    num_paths_batch: int = 1024,
    cand_size: int = 8192,
    temperature: float = 1.0,
    min_dist: Optional[float] = None,
    distinct_paths: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.double,
) -> Tensor:
    """
    Propose normalized BO points from an external preference plan.

    The function returns proposals only; it does not evaluate the objective or
    update the sampler state. If q is None, the returned shape is [1, d].
    Otherwise, the returned shape is [q, d].
    """
    if X.ndim != 2:
        raise ValueError("X must have shape [N, d].")
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    if y.ndim != 2 or y.shape[0] != X.shape[0]:
        raise ValueError("y must have shape [N, 1] and align with X.")

    _, d = X.shape
    device = device or (X.device if X.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    X = X.to(device=device, dtype=dtype).contiguous()
    y = y.to(device=device, dtype=dtype).contiguous()
    n_out = 1 if q is None else int(q)

    bounds = torch.stack(
        [
            torch.zeros(d, dtype=dtype, device=device),
            torch.ones(d, dtype=dtype, device=device),
        ],
        dim=0,
    )
    candidates = _make_sobol_grid_norm(d, cand_size, dtype, device)
    sampler = sampler_cls(bounds=bounds, X_init=X, Y_init=y, dtype=dtype, device=device)

    mode: str = plan.get("mode", "none")
    region_info = plan.get("region", None)
    x_star: Optional[Tensor] = plan.get("x_star", None)
    y_star = plan.get("y_star", None)
    delta = plan.get("delta", 0.6)
    smooth_override = (region_info or {}).get("smooth", None)

    if mode.startswith("region"):
        if region_info is None or not {"lb", "ub"} <= set(region_info.keys()):
            raise ValueError("region plans must contain plan['region'] with lb and ub.")

        lb: Tensor = region_info["lb"].to(device=device, dtype=dtype).clone().detach()
        ub: Tensor = region_info["ub"].to(device=device, dtype=dtype).clone().detach()
        grid_size = int(region_info.get("grid_size", 512))
        smooth = 0.06 if smooth_override is None else float(smooth_override)

        tilt = LinearExponentialRegionalMeanTiltPlugAndPlay(
            bounds=bounds,
            grid_size=grid_size,
            smooth=smooth,
            dtype=dtype,
            device=device,
        )
        tilt.set_box_region(lb, ub)
        tilt.fit_lambda_by_delta(
            base_model=sampler.model,
            delta=float(delta),
            observation_noise=False,
        )
        tilt.prepare_cache(base_model=sampler.model)
        effective_model = TiltedModel(sampler.model, tilt).eval()

        y_best_std = sampler._current_best_std().item()
        mc_paths = num_paths_batch if n_out > 1 else num_paths_single
        acq = qLogExpectedImprovement(
            model=effective_model,
            best_f=y_best_std,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([mc_paths])),
        )
        return _greedy_select_with_min_dist(
            acq,
            candidates,
            n=n_out,
            r_min=min_dist,
            chunk_size=128 if n_out > 1 else 256,
        )

    if mode == "value":
        if y_star is None:
            raise ValueError("value plans must contain plan['y_star'].")
        y_star_std = (
            torch.as_tensor(y_star, dtype=dtype, device=device) - sampler._y_mean
        ) / sampler._y_std.clamp_min(1e-12)
        sampler.set_location_prior(
            ValuePeakPrior(
                y_star_std=float(y_star_std.item()),
                d=d,
                dtype=dtype,
                device=device,
                beta=float(delta),
                raw_samples=2**12,
            )
        )
        return sampler.ask(
            n=n_out,
            num_paths=num_paths_batch if n_out > 1 else num_paths_single,
            cand_size=cand_size,
            temperature=temperature,
            min_dist=min_dist,
            distinct_paths=distinct_paths,
        )

    if mode == "point":
        if x_star is None:
            raise ValueError("point plans must contain plan['x_star'].")
        sampler.set_location_prior(
            PointBumpPrior(x_star_norm=x_star.to(device=device, dtype=dtype), sigma=float(delta))
        )
        return sampler.ask(
            n=n_out,
            num_paths=num_paths_batch if n_out > 1 else num_paths_single,
            cand_size=cand_size,
            temperature=temperature,
            min_dist=min_dist,
            distinct_paths=distinct_paths,
        )

    return sampler.ask(
        n=n_out,
        num_paths=num_paths_batch if n_out > 1 else num_paths_single,
        cand_size=cand_size,
        temperature=temperature,
        min_dist=min_dist,
        distinct_paths=distinct_paths,
    )
