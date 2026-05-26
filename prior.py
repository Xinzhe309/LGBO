from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.distributions as dist
from torch import Tensor

from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from botorch.sampling.pathwise import MatheronPath
from botorch.sampling.pathwise.posterior_samplers import draw_matheron_paths
from botorch.utils.sampling import optimize_posterior_samples
from botorch.utils.transforms import normalize, t_batch_mode_transform, unnormalize
from gpytorch.distributions import MultivariateNormal


def make_matheron_paths(model: Model, num_paths: int = 256, observation_noise: bool = False):
    """Build Matheron posterior sample paths across supported BoTorch versions."""
    if hasattr(MatheronPath, "from_gp"):
        return MatheronPath.from_gp(
            model,
            num_paths=num_paths,
            observation_noise=observation_noise,
        )
    return draw_matheron_paths(
        model,
        sample_shape=torch.Size([num_paths]),
    )


class UserPrior:
    pass


class UserPriorLocation(ABC):
    """
    Base class for priors over the optimizer location in normalized input space.

    Subclasses define log probability scores on X in [0, 1]^d.
    """

    def __init__(
        self,
        bounds: Tensor,
        prior_floor: float = 1e-12,
        dtype: torch.dtype = torch.double,
        seed: int = 42,
    ):
        self.bounds = bounds
        self.norm_bounds = torch.stack(
            [
                torch.zeros(bounds.shape[1], dtype=dtype, device=bounds.device),
                torch.ones(bounds.shape[1], dtype=dtype, device=bounds.device),
            ],
            dim=0,
        )
        self.dim = bounds.shape[1]
        self.prior_floor = prior_floor
        self.dtype = dtype
        self.seed = seed
        self.optval_prior: Optional[UserPriorValue] = None

    def register_maxval(self, optval_prior: "UserPriorValue") -> None:
        self.optval_prior = optval_prior

    def compute_logprobs(
        self,
        matheron_paths: MatheronPath,
        raw_samples: int = 2**11,
        **kwargs,
    ) -> Tensor:
        self.optimal_inputs, self.optimal_outputs = optimize_posterior_samples(
            matheron_paths,
            bounds=self.norm_bounds,
            raw_samples=raw_samples,
            num_restarts=20,
        )
        logprobs = self.forward(self.optimal_inputs)
        if self.optval_prior is not None:
            logprobs = logprobs + self.optval_prior.evaluate(self.optimal_outputs)
        return logprobs

    def get_optima(self):
        return self.optimal_inputs, self.optimal_outputs

    def compute_norm_probs(
        self,
        matheron_paths: MatheronPath,
        decay_factor: Optional[Union[Tensor, int, float]] = 1.0,
        prior_floor: Optional[Union[Tensor, int, float]] = 0.0,
        **kwargs,
    ) -> Tensor:
        logprobs = self.compute_logprobs(matheron_paths, **kwargs)
        logprobs_norm = logprobs - logprobs.max()
        probs = torch.exp(logprobs_norm)
        decay_probs = torch.pow(probs, decay_factor).clamp_min(prior_floor)
        num_paths = probs.shape[-2]
        return (num_paths * decay_probs) / decay_probs.sum()

    @abstractmethod
    def evaluate(self, X: Tensor) -> Tensor:
        """Return log probability scores with shape (..., n, 1)."""
        raise NotImplementedError

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        return self.evaluate(X).squeeze(1)

    @abstractmethod
    def sample(self, num_samples: int = 1) -> Tensor:
        """Sample in the original, unnormalized input space."""
        raise NotImplementedError

    def _sample(self, num_samples: int = 1) -> Tensor:
        return normalize(self.sample(num_samples=num_samples), self.bounds)


class DefaultPrior(UserPriorLocation):
    """Independent Gaussian prior around default parameter values."""

    def __init__(
        self,
        bounds: Tensor,
        parameter_defaults: Tensor,
        confidence: Optional[float] = 0.25,
        spread_dim: bool = True,
    ):
        _ = spread_dim
        super().__init__(bounds)
        self.dim = bounds.shape[1]
        self.parameter_defaults = normalize(parameter_defaults, bounds)
        if self.parameter_defaults.dim() != 1 or self.parameter_defaults.shape[0] != self.dim:
            raise ValueError(f"parameter_defaults must have shape ({self.dim},).")

        if isinstance(confidence, (float, int)):
            confidence = torch.full(
                (self.dim,),
                float(confidence),
                dtype=self.parameter_defaults.dtype,
                device=self.parameter_defaults.device,
            )
        else:
            confidence = torch.as_tensor(
                confidence,
                dtype=self.parameter_defaults.dtype,
                device=self.parameter_defaults.device,
            )
            if confidence.shape != (self.dim,):
                raise ValueError("confidence must be scalar or a vector of length d.")

        self.priors_list = []
        self.norm_factors = []
        for mu, std in zip(self.parameter_defaults, confidence):
            std = torch.clamp(std, min=1e-6)
            distr = dist.Normal(mu, std)
            upper = torch.tensor(1.0, dtype=mu.dtype, device=mu.device)
            lower = torch.tensor(0.0, dtype=mu.dtype, device=mu.device)
            self.priors_list.append(distr)
            self.norm_factors.append(distr.cdf(upper) - distr.cdf(lower))

    @property
    def default(self) -> Tensor:
        return unnormalize(self.parameter_defaults, self.bounds)

    @property
    def _default(self) -> Tensor:
        return self.parameter_defaults

    def sample(self, num_samples: int) -> Tensor:
        out_norm = torch.empty(
            num_samples,
            self.dim,
            dtype=self.parameter_defaults.dtype,
            device=self.parameter_defaults.device,
        )
        for dim in range(self.dim):
            out_norm[:, dim] = self.priors_list[dim].rsample(torch.Size([num_samples]))
        return unnormalize(out_norm, self.bounds)

    def evaluate(self, X: Tensor) -> Tensor:
        log_prob = torch.zeros(X.shape[:-1], dtype=X.dtype, device=X.device)
        for dim in range(self.dim):
            lp = self.priors_list[dim].log_prob(X[..., dim])
            factor = torch.clamp(
                self.norm_factors[dim],
                min=torch.tensor(1e-12, dtype=lp.dtype, device=lp.device),
            )
            log_prob = log_prob + lp - torch.log(factor)
        return log_prob.unsqueeze(-1)


class PreferencePrior(UserPriorLocation):
    """Preference prior from known better and worse normalized configurations."""

    def __init__(self, bounds: Tensor, better_configs: Tensor, worse_configs: Tensor):
        super().__init__(bounds)
        self.dim = bounds.shape[1]
        self.better_configs = better_configs.clone()
        self.worse_configs = worse_configs.clone()
        if self.better_configs.dim() != 2 or self.better_configs.shape[1] != self.dim:
            raise ValueError("better_configs must have shape [n, d].")
        if self.worse_configs.dim() != 2 or self.worse_configs.shape[1] != self.dim:
            raise ValueError("worse_configs must have shape [m, d].")

    @property
    def _default(self) -> Tensor:
        return self.better_configs[0]

    def sample(self, num_samples: int) -> Tensor:
        if len(self.better_configs) < num_samples:
            raise ValueError(
                "PreferencePrior cannot sample without replacement: "
                f"requested {num_samples}, available {len(self.better_configs)}."
            )
        idx = np.random.choice(len(self.better_configs), size=num_samples, replace=False)
        return unnormalize(self.better_configs[idx], self.bounds)

    def evaluate(self, X: Tensor) -> Tensor:
        eps = 1e-12
        alpha = 50.0
        beta = 50.0

        orig_shape = X.shape[:-1]
        X2 = X.reshape(-1, self.dim)

        def min_sq_dist(A: Tensor, B: Tensor) -> Tensor:
            d2 = (A[:, None, :] - B[None, :, :]).pow(2).sum(-1)
            return d2.min(dim=1).values

        d2_better = min_sq_dist(X2, self.better_configs)
        d2_worse = min_sq_dist(X2, self.worse_configs)
        score = torch.log(torch.exp(-alpha * d2_better) + eps) - torch.log(
            torch.exp(-beta * d2_worse) + eps
        )
        return score.reshape(*orig_shape, 1)


class UserPriorValue(UserPrior, ABC):
    def __init__(
        self,
        prior_floor: float = 1e-12,
        dtype: torch.dtype = torch.double,
        seed: int = 42,
    ):
        self.prior_floor = prior_floor
        self.dtype = dtype
        self.seed = seed
        self.mean = None
        self.std = None

    def setup(self, Y_normalized: Tensor, mean: Tensor, std: Tensor) -> None:
        self.Y_unnormalized = Y_normalized * std + mean
        self.mean = mean
        self.std = std

    def _unnormalize(self, Y: Tensor) -> Tensor:
        if self.mean is None or self.std is None:
            return Y
        return Y * self.std + self.mean

    @abstractmethod
    def evaluate(self, Y_opt: Tensor) -> Tensor:
        raise NotImplementedError

    @t_batch_mode_transform()
    def forward(self, Y: Tensor) -> Tensor:
        return self.evaluate(Y)


class UserPriorHardMaxValue(UserPriorValue):
    def __init__(
        self,
        maxopt_value: Optional[float] = None,
        minopt_value: Optional[float] = None,
        prior_floor: float = 1e-12,
        dtype: torch.dtype = torch.double,
        seed: int = 42,
    ):
        super().__init__(prior_floor=prior_floor, dtype=dtype, seed=seed)
        self.minopt_value = minopt_value
        self.maxopt_value = maxopt_value

    def evaluate(self, Y_opt: Tensor) -> Tensor:
        Y_unnormalized = self._unnormalize(Y_opt)
        mask = torch.ones_like(Y_unnormalized, dtype=torch.bool)
        if self.minopt_value is not None:
            mask = mask & (Y_unnormalized > self.minopt_value)
        if self.maxopt_value is not None:
            mask = mask & (Y_unnormalized < self.maxopt_value)
        return torch.log(mask.to(Y_opt.dtype) + self.prior_floor)


class UserPriorMaxValue(UserPriorValue):
    def __init__(
        self,
        parameter_default: float,
        confidence: float,
        prior_floor: float = 1e-12,
        dtype: torch.dtype = torch.double,
        seed: int = 42,
    ):
        super().__init__(prior_floor=prior_floor, dtype=dtype, seed=seed)
        self.prior_dist = dist.Normal(
            torch.tensor([parameter_default], dtype=dtype),
            torch.tensor([max(confidence, 1e-6)], dtype=dtype),
        )

    def evaluate(self, Y_opt: Tensor) -> Tensor:
        Y_unnormalized = self._unnormalize(Y_opt)
        logp = self.prior_dist.log_prob(Y_unnormalized)
        return torch.clamp(logp, min=math.log(self.prior_floor))


def _make_sobol_grid_norm(
    dim: int,
    n_points: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    eng = torch.quasirandom.SobolEngine(dim)
    X = eng.draw(n_points).to(dtype=dtype, device=device)
    eps = torch.tensor(1e-6, dtype=dtype, device=device)
    return X.clamp(eps, 1 - eps)


def _box_mask_norm(X: Tensor, lb: Tensor, ub: Tensor) -> Tensor:
    return ((X >= lb) & (X <= ub)).all(dim=-1)


class LinearExponentialRegionalMeanTiltPlugAndPlay:
    r"""
    Linear exponential regional prior implemented as an analytic mean tilt.

    For a weighted Sobol grid vector a, the tilted posterior mean is
    mean(X) + lambda * Cov(X, G) a. The covariance is left unchanged.
    """

    def __init__(
        self,
        bounds: Tensor,
        grid_size: int = 512,
        smooth: Optional[float] = None,
        dtype: torch.dtype = torch.double,
        device: Optional[torch.device] = None,
    ):
        if bounds.dim() != 2 or bounds.shape[0] != 2:
            raise ValueError("bounds must have shape [2, d].")
        self.bounds = bounds
        self.d = bounds.shape[1]
        self.dtype = dtype
        self.device = device or bounds.device
        self.Xg = _make_sobol_grid_norm(self.d, grid_size, dtype, self.device)
        self.smooth = smooth
        self._a: Optional[Tensor] = None
        self._lam: Optional[float] = None
        self._region_lb: Optional[Tensor] = None
        self._region_ub: Optional[Tensor] = None

    def set_box_region(
        self,
        lb_norm: Union[Sequence[float], Tensor],
        ub_norm: Union[Sequence[float], Tensor],
    ) -> None:
        lb = torch.as_tensor(lb_norm, dtype=self.dtype, device=self.device)
        ub = torch.as_tensor(ub_norm, dtype=self.dtype, device=self.device)
        if lb.shape != (self.d,) or ub.shape != (self.d,):
            raise ValueError(f"region bounds must have shape ({self.d},).")
        self._region_lb = lb.clamp(0, 1)
        self._region_ub = ub.clamp(0, 1)
        self._build_a()

    def set_dim_interval(self, dim: int, lo: float, hi: float) -> None:
        if not (0 <= dim < self.d):
            raise ValueError(f"dim must be in [0, {self.d}).")
        lb = torch.zeros(self.d, dtype=self.dtype, device=self.device)
        ub = torch.ones(self.d, dtype=self.dtype, device=self.device)
        lo, hi = sorted((float(lo), float(hi)))
        lb[dim], ub[dim] = max(0.0, lo), min(1.0, hi)
        self._region_lb, self._region_ub = lb, ub
        self._build_a()

    def _build_a(self) -> None:
        if self._region_lb is None or self._region_ub is None:
            raise RuntimeError("set a region before building grid weights.")
        if self.smooth is None:
            a_raw = _box_mask_norm(self.Xg, self._region_lb, self._region_ub).to(self.dtype)
        else:
            smooth = torch.tensor(self.smooth, dtype=self.dtype, device=self.device)
            z1 = (self.Xg - self._region_lb) / (smooth + 1e-12)
            z2 = (self._region_ub - self.Xg) / (smooth + 1e-12)
            a_raw = torch.sigmoid(z1).prod(dim=-1) * torch.sigmoid(z2).prod(dim=-1)
        self._a = (a_raw / torch.clamp(a_raw.sum(), min=1e-12)).to(self.dtype)

    def set_lambda(self, lam: float) -> None:
        self._lam = float(lam)

    @torch.no_grad()
    def fit_lambda_by_delta(
        self,
        base_model: Model,
        delta: float,
        observation_noise: bool = False,
        **posterior_kwargs,
    ) -> None:
        """Calibrate lambda so the regional mean lift has approximate size delta."""
        if self._a is None:
            raise RuntimeError("set a region before fitting lambda.")
        post = base_model.posterior(
            self.Xg,
            observation_noise=observation_noise,
            **posterior_kwargs,
        )
        mvn = post.mvn
        a = self._a

        try:
            aSa = mvn.lazy_covariance_matrix.quadratic_form(a)
        except (AttributeError, NotImplementedError):
            try:
                Sigma_F = mvn.covariance_matrix
            except AttributeError:
                Sigma_F = mvn.lazy_covariance_matrix.evaluate()
            aSa = (a * (Sigma_F @ a)).sum()

        denom = max(float(aSa.item() if torch.is_tensor(aSa) else aSa), 1e-6)
        self._lam = float(delta) / math.sqrt(denom)

    def _ensure_region(self) -> None:
        if self._a is None:
            raise RuntimeError("set a region before using the regional prior.")

    @torch.no_grad()
    def prepare_cache(self, base_model: Model, observation_noise: bool = False) -> None:
        """Cache training-set terms needed for repeated tilted posterior calls."""
        _ = observation_noise
        self._ensure_region()
        Xtr = base_model.train_inputs[0]

        K_tt = base_model.covar_module(Xtr, Xtr).evaluate()
        noise = (
            base_model.likelihood.noise
            if hasattr(base_model.likelihood, "noise")
            else torch.tensor(0.0, dtype=K_tt.dtype, device=K_tt.device)
        )
        A = K_tt + noise * torch.eye(K_tt.size(-1), dtype=K_tt.dtype, device=K_tt.device)

        K_tg = base_model.covar_module(Xtr, self.Xg).evaluate()
        v = K_tg @ self._a
        L = torch.linalg.cholesky(A)
        u = torch.cholesky_solve(v.unsqueeze(-1), L).squeeze(-1)
        self._cache = {"Xtr": Xtr, "L": L, "u": u}

    @torch.no_grad()
    def posterior(
        self,
        base_model: Model,
        X: Tensor,
        observation_noise: bool = False,
        **posterior_kwargs,
    ) -> GPyTorchPosterior:
        self._ensure_region()
        if self._lam is None:
            raise RuntimeError("set lambda before calling posterior.")

        post_X = base_model.posterior(
            X,
            observation_noise=observation_noise,
            **posterior_kwargs,
        )
        mvn_X = post_X.mvn
        mean_X = mvn_X.mean
        cov_lazy = mvn_X.lazy_covariance_matrix.add_jitter(1e-6)

        cache = getattr(self, "_cache", None)
        if cache is not None:
            Xtr, u = cache["Xtr"], cache["u"]
            K_xg_a = base_model.covar_module(X, self.Xg).evaluate() @ self._a
            K_xt_u = base_model.covar_module(X, Xtr).evaluate() @ u
            shift = K_xg_a - K_xt_u
        else:
            Xg = self.Xg.expand(*X.shape[:-2], *self.Xg.shape)
            X_cat = torch.cat([X, Xg], dim=-2)
            post_joint = base_model.posterior(
                X_cat,
                observation_noise=observation_noise,
                **posterior_kwargs,
            )
            mvn = post_joint.mvn
            try:
                cov_full = mvn.covariance_matrix
            except AttributeError:
                cov_full = mvn.lazy_covariance_matrix.evaluate()
            n = X.shape[-2]
            G = self.Xg.shape[-2]
            cov_XG = cov_full[..., :n, n : n + G]
            shift = torch.matmul(cov_XG, self._a.view(-1, 1)).squeeze(-1)

        mean_tilt = mean_X + self._lam * shift
        return GPyTorchPosterior(MultivariateNormal(mean_tilt, cov_lazy))


class TiltedModel(Model):
    """Thin BoTorch Model wrapper that applies a regional mean tilt."""

    def __init__(
        self,
        base_model: Model,
        mean_tilt_pp: LinearExponentialRegionalMeanTiltPlugAndPlay,
    ):
        super().__init__()
        object.__setattr__(self, "base_model", base_model)
        object.__setattr__(self, "mean_tilt_pp", mean_tilt_pp)

    @property
    def basemodel(self) -> Model:
        return object.__getattribute__(self, "base_model")

    @property
    def num_outputs(self) -> int:
        base_model = object.__getattribute__(self, "base_model")
        return getattr(base_model, "num_outputs", 1)

    def subset_output(self, idcs: Tensor) -> "TiltedModel":
        base_model = object.__getattribute__(self, "base_model")
        if hasattr(base_model, "subset_output"):
            sub = base_model.subset_output(idcs)
            return TiltedModel(sub, object.__getattribute__(self, "mean_tilt_pp"))
        return self

    def posterior(
        self,
        X: Tensor,
        observation_noise: bool = False,
        **kwargs,
    ) -> GPyTorchPosterior:
        base_model = object.__getattribute__(self, "base_model")
        mean_tilt_pp = object.__getattribute__(self, "mean_tilt_pp")
        return mean_tilt_pp.posterior(
            base_model,
            X,
            observation_noise=observation_noise,
            **kwargs,
        )

    def train(self, mode: bool = True):
        base_model = object.__getattribute__(self, "base_model")
        base_model.train(mode)
        return self

    def eval(self):
        base_model = object.__getattribute__(self, "base_model")
        base_model.eval()
        return self

    def to(self, *args, **kwargs):
        base_model = object.__getattribute__(self, "base_model").to(*args, **kwargs)
        object.__setattr__(self, "base_model", base_model)
        return self

    def __getattr__(self, name):
        if name in ("base_model", "mean_tilt_pp"):
            raise AttributeError(name)
        try:
            base_model = object.__getattribute__(self, "base_model")
        except AttributeError:
            raise AttributeError(name)
        return getattr(base_model, name)
