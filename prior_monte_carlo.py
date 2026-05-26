from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.sampling.pathwise.posterior_samplers import MatheronPath, draw_matheron_paths
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


class BaseBOSampler(ABC):
    """
    Base class for pathwise Bayesian optimization samplers.

    X is expected to live in the normalized [0, 1]^d space. Y is kept on its
    original scale and standardized internally before fitting the GP model.
    """

    def __init__(
        self,
        bounds: Tensor,
        X_init: Tensor,
        Y_init: Tensor,
        dtype: torch.dtype = torch.double,
        device: torch.device | None = None,
    ):
        if bounds.shape != (2, X_init.shape[-1]):
            raise ValueError("bounds must have shape [2, d] and align with X_init.")
        self.bounds = bounds
        self.d = bounds.shape[1]
        self.dtype = dtype
        self.device = device or X_init.device
        self.train_X = X_init.to(dtype=dtype, device=self.device)
        self.train_Y = Y_init.to(dtype=dtype, device=self.device)
        self._y_mean: Tensor | None = None
        self._y_std: Tensor | None = None
        self.location_prior = None
        self.value_prior = None
        self.model: SingleTaskGP | None = None
        self.fit_model()

    def set_location_prior(self, prior_loc) -> None:
        self.location_prior = prior_loc
        if self.value_prior is not None and hasattr(self.location_prior, "register_maxval"):
            self.location_prior.register_maxval(self.value_prior)

    def set_value_prior(self, prior_val) -> None:
        self.value_prior = prior_val
        if self.location_prior is not None and hasattr(self.location_prior, "register_maxval"):
            self.location_prior.register_maxval(self.value_prior)

    def ask(self, n: int = 1, num_paths: int = 256, **kwargs) -> Tensor:
        """Return n proposed points in normalized coordinates."""
        if self.model is None:
            raise RuntimeError("fit_model() must run before ask().")
        paths = self._make_paths(num_paths=num_paths)
        weights = self._compute_prior_weights(paths, **kwargs)
        X_next = self.propose(paths=paths, weights=weights, n=n, **kwargs)
        eps = torch.tensor(1e-6, dtype=self.dtype, device=self.device)
        return X_next.to(dtype=self.dtype, device=self.device).clamp(eps, 1 - eps)

    def tell(self, X_new: Tensor, Y_new: Tensor, refit: bool = True) -> None:
        self.train_X = torch.cat(
            [self.train_X, X_new.to(dtype=self.dtype, device=self.device)],
            dim=0,
        )
        self.train_Y = torch.cat(
            [self.train_Y, Y_new.to(dtype=self.dtype, device=self.device)],
            dim=0,
        )
        if refit:
            self.fit_model()

    @abstractmethod
    def propose(
        self,
        paths: MatheronPath,
        weights: Tensor,
        n: int = 1,
        **kwargs,
    ) -> Tensor:
        """Implement the acquisition or sampling rule and return [n, d]."""
        raise NotImplementedError

    def fit_model(self) -> None:
        Ystd, mean, std = self._standardize_with_stats(self.train_Y)
        self._y_mean, self._y_std = mean, std
        model = SingleTaskGP(train_X=self.train_X, train_Y=Ystd)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()
        self.model = model

    @torch.no_grad()
    def _make_paths(self, num_paths: int = 256, observation_noise: bool = False) -> MatheronPath:
        _ = observation_noise
        if self.model is None:
            raise RuntimeError("fit_model() must run before sampling paths.")
        return draw_matheron_paths(
            self.model,
            sample_shape=torch.Size([num_paths]),
        )

    def _compute_prior_weights(
        self,
        paths: MatheronPath,
        raw_samples: int = 2**10,
        decay_factor: float = 1.0,
        prior_floor: float = 0.0,
        **kwargs,
    ) -> Tensor:
        if self.location_prior is None:
            P = paths.sample_shape[0] if hasattr(paths, "sample_shape") else None
            if P is None:
                Xprobe = torch.rand(3, self.d, dtype=self.dtype, device=self.device)
                Yprobe = paths(Xprobe.unsqueeze(-3))
                P = Yprobe.shape[0]
            weights = torch.ones(1, P, 1, dtype=self.dtype, device=self.device)
            return (P * weights) / weights.sum()

        if self.value_prior is not None and hasattr(self.location_prior, "register_maxval"):
            self.location_prior.register_maxval(self.value_prior)

        weights = self.location_prior.compute_norm_probs(
            paths,
            decay_factor=decay_factor,
            prior_floor=prior_floor,
            raw_samples=raw_samples,
            **kwargs,
        )
        P = weights.shape[-2]
        return (P * weights) / (weights.sum() + 1e-12)

    def _current_best_std(self) -> torch.Tensor:
        if self._y_mean is None or self._y_std is None:
            raise RuntimeError("fit_model() must run before reading the standardized best value.")
        Ystd = (self.train_Y - self._y_mean) / self._y_std.clamp_min(
            torch.tensor(1e-12, dtype=self._y_std.dtype, device=self._y_std.device)
        )
        return Ystd.max()

    @staticmethod
    def _standardize_with_stats(Y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean = Y.mean(dim=0, keepdim=True)
        std = Y.std(dim=0, unbiased=False, keepdim=True).clamp_min(
            torch.tensor(1e-12, dtype=Y.dtype, device=Y.device)
        )
        return (Y - mean) / std, mean, std

    def sobol_candidates(self, n: int) -> Tensor:
        eng = torch.quasirandom.SobolEngine(self.d, scramble=True)
        X = eng.draw(n).to(dtype=self.dtype, device=self.device)
        eps = torch.tensor(1e-6, dtype=self.dtype, device=self.device)
        return X.clamp(eps, 1 - eps)

    def to_raw(self, X_norm: Tensor) -> Tensor:
        return unnormalize(X_norm, self.bounds)

    def to_norm(self, X_raw: Tensor) -> Tensor:
        return normalize(X_raw, self.bounds)


class WeightedQEI(BaseBOSampler):
    """Monte Carlo qEI with optional prior weights over posterior paths."""

    def propose(
        self,
        paths,
        weights: Tensor,
        n: int = 1,
        cand_size: int = 8192,
        temperature: float = 1.0,
        **kwargs,
    ) -> Tensor:
        _ = kwargs
        P = weights.shape[-2]
        w = weights.squeeze(0).squeeze(-1)
        w = (w**temperature)
        w = w / (w.sum() + 1e-12)

        candidates = self.sobol_candidates(cand_size)
        Y = paths(candidates.unsqueeze(-3)).squeeze(-1).squeeze(1)
        y_best = self._current_best_std()
        relu = torch.nn.ReLU()
        improvement = relu(Y - y_best)

        if n == 1:
            score = (w.view(-1, 1) * improvement).sum(dim=0)
            return candidates[int(score.argmax().item()) : int(score.argmax().item()) + 1]

        selected = []
        current = torch.full((P,), -1e30, dtype=Y.dtype, device=Y.device)
        base_gain = relu(current - y_best)
        mask = torch.ones(candidates.shape[0], dtype=torch.bool, device=candidates.device)

        for _ in range(n):
            Y_active = Y[:, mask]
            new_max = torch.maximum(current.view(-1, 1), Y_active)
            delta = relu(new_max - y_best) - base_gain.view(-1, 1)
            score = (w.view(-1, 1) * delta).sum(dim=0)
            rel_idx = int(score.argmax().item())
            j = torch.arange(candidates.shape[0], device=mask.device)[mask][rel_idx].item()
            selected.append(j)
            current = torch.maximum(current, Y[:, j])
            base_gain = relu(current - y_best)
            mask[j] = False

        return candidates[torch.tensor(selected, device=candidates.device)]


class WeightedQLogEI(BaseBOSampler):
    """Log-smoothed Monte Carlo qEI with optional prior weights over paths."""

    def propose(
        self,
        paths,
        weights: Tensor,
        n: int = 1,
        cand_size: int = 2048,
        temperature: float = 1.0,
        **kwargs,
    ) -> Tensor:
        _ = kwargs
        P = weights.shape[-2]
        w = weights.squeeze(0).squeeze(-1)
        w = (w**temperature)
        w = w / (w.sum() + 1e-12)

        candidates = self.sobol_candidates(cand_size)
        Y = paths(candidates.unsqueeze(-3)).squeeze(-1).squeeze(1)
        y_best = self._current_best_std()
        relu = torch.nn.ReLU()

        if n == 1:
            improvement = torch.log1p(relu(Y - y_best))
            score = (w.view(-1, 1) * improvement).sum(dim=0)
            return candidates[int(score.argmax().item()) : int(score.argmax().item()) + 1]

        selected = []
        current = torch.full((P,), -1e30, dtype=Y.dtype, device=Y.device)
        base_gain = torch.log1p(relu(current - y_best))
        mask = torch.ones(candidates.shape[0], dtype=torch.bool, device=candidates.device)

        for _ in range(n):
            Y_active = Y[:, mask]
            new_max = torch.maximum(current.view(-1, 1), Y_active)
            delta = torch.log1p(relu(new_max - y_best)) - base_gain.view(-1, 1)
            score = (w.view(-1, 1) * delta).sum(dim=0)
            rel_idx = int(score.argmax().item())
            j = torch.arange(candidates.shape[0], device=mask.device)[mask][rel_idx].item()
            selected.append(j)
            current = torch.maximum(current, Y[:, j])
            base_gain = torch.log1p(relu(current - y_best))
            mask[j] = False

        return candidates[torch.tensor(selected, device=candidates.device)]


class WeightedTS(BaseBOSampler):
    """Weighted Thompson sampling over posterior paths."""

    def propose(
        self,
        paths: MatheronPath,
        weights: Tensor,
        n: int = 1,
        cand_size: int = 2048,
        temperature: float = 1.0,
        distinct_paths: bool = True,
        **kwargs,
    ) -> Tensor:
        _ = kwargs
        P = weights.shape[-2]
        w = weights.squeeze(0).squeeze(-1)
        w = (w**temperature)
        w = w / (w.sum() + 1e-12)

        candidates = self.sobol_candidates(cand_size)
        with torch.no_grad():
            Y = paths(candidates.unsqueeze(-3)).squeeze(-1).squeeze(1)

        if n == 1:
            i = torch.multinomial(w, 1, replacement=True).item()
            j = int(Y[i].argmax().item())
            return candidates[j : j + 1]

        used_candidates = torch.zeros(candidates.shape[0], dtype=torch.bool, device=candidates.device)
        w_work = w.clone()
        selected = []

        for step in range(n):
            replacement = True
            if distinct_paths:
                replacement = step >= P
            if w_work.sum() <= 1e-12:
                w_work = w.clone()

            i = torch.multinomial(w_work, 1, replacement=replacement).item()
            y_i = Y[i].clone()
            if (~used_candidates).any():
                y_i[used_candidates] = -1e30
                j = int(y_i.argmax().item())
            else:
                j = int(Y[i].argmax().item())

            selected.append(candidates[j])
            used_candidates[j] = True

            if distinct_paths and not replacement:
                w_work[i] = 0.0
                total = w_work.sum()
                w_work = w_work / total if total > 1e-12 else w.clone()

        return torch.stack(selected, dim=0)


class PointBumpPrior:
    """Path prior that favors solutions near a known normalized point."""

    def __init__(self, x_star_norm: Tensor, sigma: float = 0.06, prior_floor: float = 1e-6):
        self.x_star = x_star_norm.detach()
        self.sigma2 = max(float(sigma) ** 2, 1e-12)
        self.floor = float(prior_floor)

    @torch.no_grad()
    def compute_norm_probs(self, paths, raw_samples: int = 2**12, **kwargs) -> Tensor:
        _ = kwargs
        dim = self.x_star.numel()
        eng = torch.quasirandom.SobolEngine(dim, scramble=True, seed=777)
        candidates = eng.draw(raw_samples).to(dtype=self.x_star.dtype, device=self.x_star.device)
        Y = paths(candidates.unsqueeze(-3)).squeeze(-1).squeeze(1)
        diff2 = ((candidates - self.x_star) ** 2).sum(dim=-1)
        position_score = torch.exp(-diff2 / (2 * self.sigma2))
        raw = position_score.sum().expand(Y.shape[0]).clamp_min(self.floor)
        weights = raw / (raw.sum() + 1e-12)
        P = weights.numel()
        return (P * weights).view(1, P, 1)


class ValuePeakPrior:
    """Path prior that favors paths whose maximum is close to a target value."""

    def __init__(
        self,
        y_star_std: float,
        *,
        d: int,
        dtype: torch.dtype,
        device: torch.device,
        beta: float = 0.3,
        prior_floor: float = 1e-6,
        raw_samples: int = 2**12,
        sobol_seed: int = 778,
    ):
        self.y_star_std = float(y_star_std)
        self.beta2 = max(float(beta) ** 2, 1e-12)
        self.floor = float(prior_floor)
        self.raw_samples = int(raw_samples)
        self.d = int(d)
        self.dtype = dtype
        self.device = device
        self.sobol_seed = int(sobol_seed)

    @torch.no_grad()
    def compute_norm_probs(self, paths, **kwargs) -> Tensor:
        _ = kwargs
        probe = torch.rand(3, self.d, dtype=self.dtype, device=self.device)
        Y_probe = paths(probe.unsqueeze(-3)).squeeze(-1).squeeze(1)
        P = Y_probe.shape[0]

        eng = torch.quasirandom.SobolEngine(self.d, scramble=True, seed=self.sobol_seed)
        candidates = eng.draw(self.raw_samples).to(dtype=self.dtype, device=self.device)
        Y = paths(candidates.unsqueeze(-3)).squeeze(-1).squeeze(1)
        y_max = Y.max(dim=-1).values

        raw = torch.exp(-((y_max - self.y_star_std) ** 2) / (2 * self.beta2))
        raw = raw.clamp_min(self.floor)
        weights = raw / (raw.sum() + 1e-12)
        return (P * weights).view(1, P, 1)
