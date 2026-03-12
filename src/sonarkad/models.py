"""Models: B-spline edges, SonarKAD variants, and baselines."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .bspline import make_open_uniform_knots, bspline_basis_matrix_np, bspline_basis_matrix_torch
from .surrogate import thorp_absorption_db_per_km


def thorp_absorption_db_per_km_torch(f_hz: torch.Tensor) -> torch.Tensor:
    """Torch version of :func:`thorp_absorption_db_per_km`.

    Parameters
    ----------
    f_hz:
        Frequency in Hz. Any shape.

    Returns
    -------
    alpha_db_per_km:
        Thorp absorption in dB/km.

    Notes
    -----
    Uses the common Thorp approximation:
      α = 0.11 f^2/(1+f^2) + 44 f^2/(4100+f^2) + 2.75e-4 f^2 + 0.003,
    with f in kHz.
    """
    f_khz = f_hz.to(dtype=torch.float32) / 1000.0
    f2 = f_khz * f_khz
    return 0.11 * f2 / (1.0 + f2) + 44.0 * f2 / (4100.0 + f2) + 2.75e-4 * f2 + 0.003


# ---------------------------------------------------------------------
# B-spline edge layer
# ---------------------------------------------------------------------


@dataclass
class BSplineLayerConfig:
    n_basis: int = 23
    degree: int = 3
    xmin: float = 0.0
    xmax: float = 1.0
    # Ridge parameter for physics-init least squares
    ridge_lambda: float = 1e-6


def bspline_layer_config_from_dict(d: Optional[Dict[str, object]] = None) -> BSplineLayerConfig:
    """Parse a :class:`BSplineLayerConfig` from a (possibly legacy) dict.

    Several earlier iterations of this repo used different key names
    (e.g., ``num_knots`` / ``x_min`` / ``x_max``). Python 3.12 raises a
    ``TypeError`` if unexpected keys are passed to a dataclass constructor.

    This helper provides backward compatibility by mapping common synonyms
    onto the dataclass field names and silently dropping unknown keys.

    Notes
    -----
    In this codebase, ``n_basis`` refers to the number of B-spline basis
    functions (not the length of the knot vector).
    """
    d0: Dict[str, object] = dict(d or {})

    # Synonyms for n_basis
    if "n_basis" not in d0:
        if "num_knots" in d0:
            d0["n_basis"] = d0.get("num_knots")
        elif "n_knots" in d0:
            d0["n_basis"] = d0.get("n_knots")
        elif "num_basis" in d0:
            d0["n_basis"] = d0.get("num_basis")

    # Synonyms for domain
    if "xmin" not in d0 and "x_min" in d0:
        d0["xmin"] = d0.get("x_min")
    if "xmax" not in d0 and "x_max" in d0:
        d0["xmax"] = d0.get("x_max")

    # Filter unexpected keys.
    allowed = {f.name for f in fields(BSplineLayerConfig)}
    d1 = {k: v for k, v in d0.items() if k in allowed}
    return BSplineLayerConfig(**d1)


class BSplineLayer(nn.Module):
    """A learnable spline edge layer.

    For an input vector x in R^{n_in}, the output is:
        y = Linear(x) + sum_i sum_m c_{i,o,m} B_m(x_i)

    Notes
    -----
    - This implementation is intentionally explicit (basis evaluation + einsum) so the
      learned *univariate* functions can be inspected and plotted directly.
    - For SonarKAD we use ``num_inputs=num_outputs=1`` for each branch.
    """

    def __init__(self, num_inputs: int, num_outputs: int, cfg: BSplineLayerConfig):
        super().__init__()
        self.num_inputs = int(num_inputs)
        self.num_outputs = int(num_outputs)
        self.cfg = cfg

        knots_np = make_open_uniform_knots(cfg.n_basis, cfg.degree, cfg.xmin, cfg.xmax)
        self.register_buffer("knots", torch.tensor(knots_np, dtype=torch.float32))

        # Spline coefficients: (n_in, n_out, n_basis)
        self.coefficients = nn.Parameter(torch.randn(self.num_inputs, self.num_outputs, cfg.n_basis) * 0.05)

        # Optional linear residual term
        self.base_linear = nn.Linear(self.num_inputs, self.num_outputs, bias=True)
        nn.init.zeros_(self.base_linear.weight)
        nn.init.zeros_(self.base_linear.bias)

    def add_scaled_from_(self, other: "BSplineLayer", scale: float) -> None:
        """In-place: self += scale * other for all learnable parameters.

        This is used by the interaction gauge-fix step to move separable
        components from the low-rank interaction into the additive terms,
        while preserving the overall network output.
        """
        if scale == 0.0:
            return
        with torch.no_grad():
            self.coefficients.add_(other.coefficients, alpha=scale)
            self.base_linear.weight.add_(other.base_linear.weight, alpha=scale)
            self.base_linear.bias.add_(other.base_linear.bias, alpha=scale)

    def scale_output_(self, scale: float) -> None:
        """In-place: multiply the layer output by a constant scale.

        After calling this method, for any input x:
            f_new(x) = scale * f_old(x).

        This is useful for gauge-fixing low-rank interaction factors (u_k, v_k)
        where their product is identifiable but their individual scales are not.
        """
        s = float(scale)
        if s == 1.0:
            return
        with torch.no_grad():
            self.coefficients.mul_(s)
            self.base_linear.weight.mul_(s)
            self.base_linear.bias.mul_(s)


    def roughness_penalty(self, order: int = 2) -> torch.Tensor:
        """Return a spline-coefficient roughness penalty.

        The penalty is computed on the coefficient axis (finite differences). While
        this is not exactly the integrated squared second-derivative of the spline,
        it is a simple and effective smoothness prior in practice.

        Parameters
        ----------
        order:
            1 (first difference) or 2 (second difference).
        """
        c = self.coefficients
        if c.size(-1) < (int(order) + 1):
            return c.new_tensor(0.0)
        if int(order) == 1:
            d = c[..., 1:] - c[..., :-1]
        elif int(order) == 2:
            d = c[..., 2:] - 2.0 * c[..., 1:-1] + c[..., :-2]
        else:
            raise ValueError(f"Unsupported roughness order: {order}")
        return torch.mean(d ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_in)
        if x.dim() != 2 or x.size(1) != self.num_inputs:
            raise ValueError(f"Expected x shape (B,{self.num_inputs}); got {tuple(x.shape)}")

        # basis per input dimension: (B, n_in, n_basis)
        basis_list = []
        for i in range(self.num_inputs):
            Bi = bspline_basis_matrix_torch(x[:, i], self.knots, self.cfg.degree)  # (B, n_basis)
            basis_list.append(Bi.unsqueeze(1))
        basis = torch.cat(basis_list, dim=1)  # (B, n_in, n_basis)

        spline_out = torch.einsum("bin,ion->bo", basis, self.coefficients)
        return self.base_linear(x) + spline_out

    @torch.no_grad()
    def set_parameters_from_least_squares(
        self,
        x_grid: np.ndarray,
        y_target: np.ndarray,
        ridge_lambda: Optional[float] = None,
    ) -> Dict[str, float]:
        """Fit (linear + spline) parameters to a 1D target function using ridge LS.

        Used for physics-informed initialization of ``phi_r``.
        """
        if self.num_inputs != 1 or self.num_outputs != 1:
            raise NotImplementedError("LS init currently implemented for 1->1 layers only.")
        lam = self.cfg.ridge_lambda if ridge_lambda is None else float(ridge_lambda)

        x_grid = np.asarray(x_grid, dtype=np.float64).reshape(-1)
        y_target = np.asarray(y_target, dtype=np.float64).reshape(-1)
        if x_grid.shape[0] != y_target.shape[0]:
            raise ValueError("x_grid and y_target must have same length.")

        B = bspline_basis_matrix_np(x_grid, self.knots.cpu().numpy(), self.cfg.degree)  # (N, n_basis)

        # Design matrix: [x, 1, B]
        A = np.concatenate([x_grid.reshape(-1, 1), np.ones((x_grid.size, 1)), B], axis=1)

        # Ridge solve: (A^T A + lam I) theta = A^T y
        AtA = A.T @ A
        reg = lam * np.eye(AtA.shape[0])
        theta = np.linalg.solve(AtA + reg, A.T @ y_target)  # (2+n_basis,)

        w = float(theta[0])
        b = float(theta[1])
        c = theta[2:].astype(np.float32)

        self.base_linear.weight[:] = torch.tensor([[w]], dtype=self.base_linear.weight.dtype, device=self.base_linear.weight.device)
        self.base_linear.bias[:] = torch.tensor([b], dtype=self.base_linear.bias.dtype, device=self.base_linear.bias.device)
        self.coefficients[:] = torch.tensor(c.reshape(1, 1, -1), dtype=self.coefficients.dtype, device=self.coefficients.device)

        y_fit = A @ theta
        rmse = float(np.sqrt(np.mean((y_fit - y_target) ** 2)))
        return {"rmse": rmse, "ridge_lambda": float(lam)}


# ---------------------------------------------------------------------
# SonarKAD: additive + optional low-rank interaction ψ(r,f)
# ---------------------------------------------------------------------


@dataclass
class AbsorptionTermConfig:
    """Configuration for the explicit nonseparable absorption term.

    Motivation
    ----------
    In many practical bands (e.g., 1--5 kHz), transmission loss contains a
    frequency-dependent absorption contribution that is *not separable*:

        TL_abs(r,f) ≈ α(f) r,

    where α(f) varies over the band. In log-intensity form, this contributes a
    term proportional to ``-α(f) r``.

    This module parameterizes the contribution as

        φ_abs(r,f) = - r_km [α(f) - α(fc)],

    so that only the *frequency-dependent* part of absorption is modeled
    explicitly, improving identifiability with the large-scale range trend.
    """

    enabled: bool = False

    # Modes:
    # - 'thorp_fixed':  α(f) from Thorp formula
    # - 'thorp_scale':  α(f) = exp(log_scale) * α_thorp(f)
    # - 'spline':       α(f) learned as a positive spline (dB/km)
    mode: str = "thorp_scale"

    # Anchor to α(fc) to avoid mixing with φ_r(r).
    reference_fc: bool = True

    # Only for 'thorp_scale'
    init_log_scale: float = 0.0

    # Only for 'spline'
    spline: BSplineLayerConfig = field(default_factory=BSplineLayerConfig)
    alpha_floor_db_per_km: float = 0.0


@dataclass
class SonarKADConfig:
    """Configuration for SonarKAD.

    The original (separable) model is strictly additive:
        y = SL + mean_tl + φ_r(r) + φ_f(f) + b

    To align with real shallow-water data (striations / modal interference), we add
    an optional low-rank interaction term:
        y = ... + ψ(r,f),    ψ(r,f)=∑_{k=1}^K u_k(r) v_k(f)

    Setting ``interaction_rank=0`` recovers the original additive SonarKAD.
    """

    spline: BSplineLayerConfig = field(default_factory=BSplineLayerConfig)

    # Physics-informed init for φ_r
    physics_init_grid_n: int = 256
    fc_hz: float = 3000.0
    use_absorption: bool = True

    # Physical frequency range for mapping f_norm -> f_hz when needed
    f_min_hz: float = 0.0
    f_max_hz: float = 0.0

    # Optional explicit absorption term  -r_km[α(f)-α(fc)]
    absorption: AbsorptionTermConfig = field(default_factory=AbsorptionTermConfig)

    # Additive gauge fixing (identifiability)
    gauge_fix_each_epoch: bool = True
    gauge_fix_grid_n: int = 200
    gauge_fix_interaction: bool = True  # center ψ on a grid via an internal ψ-bias

    # Optional: remove the u_k / v_k scale ambiguity by normalizing factors on the gauge grid.
    gauge_fix_normalize_factors: bool = True
    gauge_fix_factor_mode: str = "std"  # 'std' or 'l2'
    gauge_fix_factor_eps: float = 1e-6
    gauge_fix_fix_sign: bool = True

    # Convenience constant in the surrogate (can be 0 for real data)
    SL_db: float = 180.0

    # Low-rank interaction
    interaction_rank: int = 0


class LowRankInteraction(nn.Module):
    """ψ(r,f)=∑ u_k(r) v_k(f) + b_int, with learnable spline factors."""

    def __init__(self, rank: int, spline_cfg: BSplineLayerConfig):
        super().__init__()
        self.rank = int(rank)
        if self.rank <= 0:
            raise ValueError("rank must be >= 1")

        self.u = nn.ModuleList([BSplineLayer(1, 1, spline_cfg) for _ in range(self.rank)])
        self.v = nn.ModuleList([BSplineLayer(1, 1, spline_cfg) for _ in range(self.rank)])

        # A dedicated interaction bias makes it easy to enforce mean-zero ψ via gauge fixing.
        self.int_bias = nn.Parameter(torch.zeros(1))

    def forward(self, r: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        if r.shape != f.shape:
            raise ValueError(f"r and f must have the same shape. Got r={tuple(r.shape)} f={tuple(f.shape)}")
        out = 0.0
        for uk, vk in zip(self.u, self.v):
            out = out + uk(r) * vk(f)
        return out + self.int_bias


    @property
    def u_layers(self) -> nn.ModuleList:
        """Alias for backwards compatibility."""
        return self.u

    @property
    def v_layers(self) -> nn.ModuleList:
        """Alias for backwards compatibility."""
        return self.v


class AbsorptionTerm(nn.Module):
    """Explicit nonseparable absorption term:  φ_abs(r,f) = -r_km[α(f)-α(fc)]."""

    def __init__(
        self,
        *,
        r_min_m: float,
        r_max_m: float,
        f_min_hz: float,
        f_max_hz: float,
        fc_hz: float,
        cfg: AbsorptionTermConfig,
    ):
        super().__init__()
        self.r_min_m = float(r_min_m)
        self.r_max_m = float(r_max_m)
        self.f_min_hz = float(f_min_hz)
        self.f_max_hz = float(f_max_hz)
        self.fc_hz = float(fc_hz)
        self.cfg = cfg

        if self.f_max_hz <= self.f_min_hz:
            raise ValueError(
                "AbsorptionTerm requires a valid physical frequency range (f_min_hz < f_max_hz). "
                f"Got f_min_hz={self.f_min_hz} f_max_hz={self.f_max_hz}."
            )

        mode = str(cfg.mode).strip().lower()
        if mode not in {"thorp_fixed", "thorp_scale", "spline"}:
            raise ValueError(f"Unsupported absorption mode: {cfg.mode}")
        self.mode = mode

        if self.mode == "thorp_scale":
            self.log_scale = nn.Parameter(torch.tensor([float(cfg.init_log_scale)], dtype=torch.float32))
        else:
            self.log_scale = None

        if self.mode == "spline":
            # Positive spline in dB/km (softplus).
            self.alpha_spline = BSplineLayer(1, 1, cfg.spline)
        else:
            self.alpha_spline = None

        # Precompute alpha(fc) for fixed Thorp mode.
        if self.mode in {"thorp_fixed", "thorp_scale"}:
            with torch.no_grad():
                a_fc = float(thorp_absorption_db_per_km(np.array([self.fc_hz], dtype=np.float64))[0])
            self.register_buffer("alpha_fc_db_per_km", torch.tensor([a_fc], dtype=torch.float32))
        else:
            self.register_buffer("alpha_fc_db_per_km", torch.tensor([0.0], dtype=torch.float32))

    def _map_f(self, f_norm: torch.Tensor) -> torch.Tensor:
        # f_norm in [0,1] -> Hz
        return self.f_min_hz + f_norm * (self.f_max_hz - self.f_min_hz)

    def _map_r_km(self, r_norm: torch.Tensor) -> torch.Tensor:
        r_m = self.r_min_m + r_norm * (self.r_max_m - self.r_min_m)
        return r_m / 1000.0

    def _alpha_db_per_km(self, f_hz: torch.Tensor) -> torch.Tensor:
        if self.mode in {"thorp_fixed", "thorp_scale"}:
            a = thorp_absorption_db_per_km_torch(f_hz)
            if self.mode == "thorp_scale" and self.log_scale is not None:
                a = torch.exp(self.log_scale) * a
            return a

        assert self.alpha_spline is not None
        # Map Hz back to normalized domain for spline evaluation.
        f_norm = (f_hz - self.f_min_hz) / max(self.f_max_hz - self.f_min_hz, 1e-12)
        raw = self.alpha_spline(f_norm.reshape(-1, 1)).reshape(f_hz.shape)
        a = torch.nn.functional.softplus(raw) + float(self.cfg.alpha_floor_db_per_km)
        return a

    def forward(self, r_norm: torch.Tensor, f_norm: torch.Tensor) -> torch.Tensor:
        if r_norm.shape != f_norm.shape:
            raise ValueError(f"r_norm and f_norm must have same shape. Got {tuple(r_norm.shape)} vs {tuple(f_norm.shape)}")

        r_km = self._map_r_km(r_norm)
        f_hz = self._map_f(f_norm)
        alpha = self._alpha_db_per_km(f_hz)

        if bool(self.cfg.reference_fc):
            if self.mode in {"thorp_fixed", "thorp_scale"}:
                alpha_ref = self.alpha_fc_db_per_km
            else:
                alpha_ref = self._alpha_db_per_km(torch.tensor(self.fc_hz, device=f_hz.device, dtype=f_hz.dtype))
            alpha = alpha - alpha_ref

        # Log-intensity contribution (dB)
        return -r_km * alpha


class SonarKAD(nn.Module):
    """2D SonarKAD specialized for inputs x=[r_norm, f_norm] in [0,1]^2."""

    def __init__(self, r_min_m: float, r_max_m: float, cfg: SonarKADConfig):
        super().__init__()
        self.r_min = float(r_min_m)
        self.r_max = float(r_max_m)
        self.cfg = cfg

        self.phi_r = BSplineLayer(1, 1, cfg.spline)
        self.phi_f = BSplineLayer(1, 1, cfg.spline)
        self.bias = nn.Parameter(torch.zeros(1))

        # Optional explicit absorption term  -r_km[α(f)-α(fc)]
        self.absorption_term: Optional[AbsorptionTerm] = None
        if bool(getattr(cfg, "absorption", AbsorptionTermConfig()).enabled):
            self.absorption_term = AbsorptionTerm(
                r_min_m=self.r_min,
                r_max_m=self.r_max,
                f_min_hz=float(getattr(cfg, "f_min_hz", 0.0)),
                f_max_hz=float(getattr(cfg, "f_max_hz", 0.0)),
                fc_hz=float(getattr(cfg, "fc_hz", 0.0)),
                cfg=getattr(cfg, "absorption"),
            )

        self.interaction: Optional[LowRankInteraction] = None
        if int(cfg.interaction_rank) > 0:
            self.interaction = LowRankInteraction(int(cfg.interaction_rank), cfg.spline)

        # Stored after physics_init
        self.register_buffer("mean_tl", torch.tensor(0.0, dtype=torch.float32))

    @torch.no_grad()
    def physics_init(self) -> Dict[str, float]:
        """Initialize φ_r to match spherical spreading + absorption at fc (ridge LS)."""
        x_grid = np.linspace(0.0, 1.0, int(self.cfg.physics_init_grid_n), dtype=np.float64)
        r_phys = x_grid * (self.r_max - self.r_min) + self.r_min

        target = -20.0 * np.log10(r_phys)
        if self.cfg.use_absorption:
            alpha_fc = float(thorp_absorption_db_per_km(np.array([self.cfg.fc_hz]))[0])  # dB/km
            target = target - alpha_fc * (r_phys / 1000.0)

        mean_tl = float(np.mean(target))
        self.mean_tl.copy_(torch.tensor(mean_tl, dtype=self.mean_tl.dtype, device=self.mean_tl.device))
        target_centered = target - mean_tl

        stats = self.phi_r.set_parameters_from_least_squares(x_grid=x_grid, y_target=target_centered)
        stats.update({"mean_tl": mean_tl})
        return stats

    @torch.no_grad()
    def gauge_fix(self, grid_n: Optional[int] = None) -> Dict[str, float]:
        r"""Fix gauge/identifiability on a reference grid.

        The additive decomposition

            y(r,f) = b + \phi_r(r) + \phi_f(f) + \psi(r,f)

        is not unique unless constraints are imposed. This routine enforces a
        convenient *gauge* on a uniform reference grid:

        1) mean(\phi_r)=0 and mean(\phi_f)=0 (their means are absorbed into b)
        2) if a low-rank interaction \psi(r,f)=\sum_k u_k(r)v_k(f) is enabled and
           ``gauge_fix_interaction`` is True, we additionally enforce
           mean(u_k)=0 and mean(v_k)=0 for each factor on the grid. This implies
           that \psi has (approximately) zero range and frequency marginals on
           that grid, improving interpretability of \psi as an *interaction*.

        Notes
        -----
        - The factor-mean constraints are enforced by shifting the corresponding
          B-spline biases and compensating by adding the induced separable terms
          into \phi_r and \phi_f. The overall network output is preserved.
        """

        n = int(self.cfg.gauge_fix_grid_n if grid_n is None else grid_n)
        device = self.bias.device
        dtype = self.bias.dtype

        r_grid = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype).reshape(-1, 1)
        f_grid = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype).reshape(-1, 1)

        mu_rf = torch.tensor(0.0, device=device, dtype=dtype)
        mu_u_max = 0.0
        mu_v_max = 0.0

        with torch.no_grad():
            # --- Interaction factor centering (optional) ---
            if self.interaction is not None and bool(self.cfg.gauge_fix_interaction):
                # 1) Center v_k(f) means; move u_k(r)*mean(v_k) into phi_r.
                for u_k, v_k in zip(self.interaction.u_layers, self.interaction.v_layers):
                    mu_v = v_k(f_grid).mean()
                    mu_v_max = max(mu_v_max, float(mu_v.abs().item()))
                    if float(mu_v.abs().item()) > 0.0:
                        v_k.base_linear.bias.add_(-mu_v)
                        self.phi_r.add_scaled_from_(u_k, float(mu_v.item()))

                # 2) Center u_k(r) means; move mean(u_k)*v_k(f) into phi_f.
                #    (v_k has already been centered above.)
                for u_k, v_k in zip(self.interaction.u_layers, self.interaction.v_layers):
                    mu_u = u_k(r_grid).mean()
                    mu_u_max = max(mu_u_max, float(mu_u.abs().item()))
                    if float(mu_u.abs().item()) > 0.0:
                        u_k.base_linear.bias.add_(-mu_u)
                        self.phi_f.add_scaled_from_(v_k, float(mu_u.item()))

                # 3) Remove any residual mean of the interaction over the product grid.
                rr = r_grid.repeat_interleave(n, dim=0)  # (n^2, 1)
                ff = f_grid.repeat(n, 1)  # (n^2, 1)
                mu_rf = self.interaction(rr, ff).mean()
                self.interaction.int_bias.add_(-mu_rf)
                self.bias.add_(mu_rf)


                # 4) Optional factor normalization to improve identifiability and comparability.
                if bool(getattr(self.cfg, 'gauge_fix_normalize_factors', False)):
                    mode = str(getattr(self.cfg, 'gauge_fix_factor_mode', 'std')).strip().lower()
                    eps_n = float(getattr(self.cfg, 'gauge_fix_factor_eps', 1e-6))
                    fix_sign = bool(getattr(self.cfg, 'gauge_fix_fix_sign', True))

                    f_mid = torch.tensor([[0.5]], device=device, dtype=dtype)

                    for u_k, v_k in zip(self.interaction.u_layers, self.interaction.v_layers):
                        u_vals = u_k(r_grid).view(-1)
                        if mode == 'l2':
                            s_u = float(torch.sqrt(torch.mean(u_vals ** 2)).item())
                        else:
                            s_u = float(torch.std(u_vals).item())

                        if np.isfinite(s_u) and s_u > eps_n:
                            # Normalize u_k to unit scale; compensate in v_k to keep u_k*v_k unchanged.
                            u_k.scale_output_(1.0 / s_u)
                            v_k.scale_output_(s_u)

                        if fix_sign:
                            try:
                                v_mid = float(v_k(f_mid).view(-1)[0].item())
                            except Exception:
                                v_mid = 0.0
                            if np.isfinite(v_mid) and v_mid < 0.0:
                                u_k.scale_output_(-1.0)
                                v_k.scale_output_(-1.0)

            # --- Additive branch centering (always) ---
            mu_r = self.phi_r(r_grid).mean()
            mu_f = self.phi_f(f_grid).mean()

            self.phi_r.base_linear.bias.add_(-mu_r)
            self.phi_f.base_linear.bias.add_(-mu_f)
            self.bias.add_(mu_r + mu_f)

        return {
            "mu_r": float(mu_r.item()),
            "mu_f": float(mu_f.item()),
            "mu_rf": float(mu_rf.item()),
            "mu_u_max": float(mu_u_max),
            "mu_v_max": float(mu_v_max),
        }

    def forward_components(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        r = x[:, 0:1]
        f = x[:, 1:2]
        out_r = self.phi_r(r)
        out_f = self.phi_f(f)

        out_abs = torch.zeros_like(out_r)
        if self.absorption_term is not None:
            out_abs = self.absorption_term(r, f)

        out_rf = torch.zeros_like(out_r)
        if self.interaction is not None:
            out_rf = self.interaction(r, f)
        return {"phi_r": out_r, "phi_f": out_f, "phi_abs": out_abs, "psi_rf": out_rf}


    def spline_roughness_penalty(
        self,
        *,
        order: int = 2,
        include_interaction: bool = True,
        include_absorption: bool = True,
    ) -> torch.Tensor:
        """Sum roughness penalties over all spline edge functions.

        Intended for training-time regularization.
        """
        pen = self.phi_r.roughness_penalty(order=order) + self.phi_f.roughness_penalty(order=order)

        if include_absorption and (self.absorption_term is not None) and getattr(self.absorption_term, "alpha_spline", None) is not None:
            # Only the spline mode has learnable univariate edge functions.
            pen = pen + self.absorption_term.alpha_spline.roughness_penalty(order=order)  # type: ignore[union-attr]

        if include_interaction and (self.interaction is not None):
            for u_k in self.interaction.u_layers:
                pen = pen + u_k.roughness_penalty(order=order)
            for v_k in self.interaction.v_layers:
                pen = pen + v_k.roughness_penalty(order=order)
        return pen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        comps = self.forward_components(x)
        y_hat = (
            comps["phi_r"]
            + comps["phi_f"]
            + comps.get("phi_abs", 0.0)
            + comps["psi_rf"]
            + self.bias
            + self.cfg.SL_db
            + self.mean_tl
        )
        return y_hat




# ---------------------------------------------------------------------
# SonarKAD: Source–Propagation Aligned Rank-K KAN (new acronym)
# ---------------------------------------------------------------------

class SonarKAD(SonarKAD):
    """SonarKAD: Source–Propagation Aligned Rank-K Kolmogorov–Arnold Network.

    In this codebase, SonarKAD is implemented as SonarKAD with a *nonseparable* low-rank
    interaction term ψ(r,f)=∑ u_k(r)v_k(f) enabled via ``interaction_rank>0``.

    - ``interaction_rank=0``  => original separable SonarKAD (additive only)
    - ``interaction_rank>=1`` => SonarKAD (additive + low-rank coupling)

    This alias is provided mainly for paper clarity; it does not change the
    underlying implementation.
    """

    pass

class SmallMLP(nn.Module):
    """A small baseline MLP (parameter-matched in experiments)."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
