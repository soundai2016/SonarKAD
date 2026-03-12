"""Baselines for (r,f)->RL regression in the log-power domain.

These baselines are included to address common reviewer expectations in the
JASA/JOE passive-sonar literature:

1) Parametric spreading/absorption surrogate (sonar-equation style)
2) Spline-based generalized additive model (GAM): g_r(r)+g_f(f)
3) A simple waveguide-invariant striation fit for range--frequency coupling

The goal is NOT to compete with full coherent-field models (modes/PE/MFP), but to
provide transparent, traditional benchmarks for intensity-domain regression.

All implementations are NumPy-only (no external stats packages), so the code can
run in minimal environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .bspline import make_open_uniform_knots, bspline_basis_matrix_np


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Explained variance (EV) in the regression sense.

    EV = 1 - Var(residual) / Var(y_true)

    This is robust to constant offsets.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    res = y_true - y_pred
    var_y = float(np.var(y_true))
    if var_y <= 1e-30:
        return 0.0
    return float(1.0 - np.var(res) / var_y)


# -----------------------------------------------------------------------------
# Parametric TL baseline
# -----------------------------------------------------------------------------


def thorp_alpha_db_per_km(f_hz: np.ndarray) -> np.ndarray:
    """Thorp absorption coefficient α(f) in dB/km.

    Classic Thorp (1967) empirical formula. This is widely used as a first-order
    absorption model.

    Parameters
    ----------
    f_hz : array
        Frequency in Hz.

    Returns
    -------
    alpha_db_per_km : array
        Absorption in dB/km.
    """
    f_khz = np.asarray(f_hz, dtype=np.float64) / 1000.0
    f2 = f_khz ** 2
    # Thorp (1967): valid over broad ranges; low-frequency terms become small.
    alpha = 0.11 * f2 / (1.0 + f2) + 44.0 * f2 / (4100.0 + f2) + 2.75e-4 * f2 + 0.003
    return alpha


@dataclass
class ParametricTLConfig:
    """Configuration for a simple parametric TL surrogate."""

    include_log_spreading: bool = True
    include_linear_range: bool = False
    include_thorp_absorption: bool = True
    ridge_lambda: float = 1e-6


def fit_parametric_tl(
    r_m: np.ndarray,
    f_hz: np.ndarray,
    y_db: np.ndarray,
    train_mask: np.ndarray,
    cfg: Optional[ParametricTLConfig] = None,
) -> Tuple[Dict[str, object], Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """Fit a simple parametric spreading/absorption baseline.

    Model (example)
    --------------
    y = b + c1 * log10(r) + c2 * r + c3 * alpha(f) * r

    Notes
    -----
    - In the 50--400 Hz band, absorption is small but nonzero.
    - The goal is a transparent baseline, not a full propagation model.
    """
    if cfg is None:
        cfg = ParametricTLConfig()

    r = np.asarray(r_m, dtype=np.float64).reshape(-1)
    f = np.asarray(f_hz, dtype=np.float64).reshape(-1)
    y = np.asarray(y_db, dtype=np.float64).reshape(-1)
    m = np.asarray(train_mask, dtype=bool).reshape(-1)

    cols = [np.ones_like(r)]
    names = ["bias"]

    if cfg.include_log_spreading:
        cols.append(np.log10(np.maximum(r, 1e-3)))
        names.append("log10_r")

    if cfg.include_linear_range:
        cols.append(r)
        names.append("r")

    if cfg.include_thorp_absorption:
        alpha_db_per_m = thorp_alpha_db_per_km(f) / 1000.0
        cols.append(alpha_db_per_m * r)
        names.append("alpha(f)*r")

    A = np.stack(cols, axis=1)  # (N,p)
    A_tr = A[m]
    y_tr = y[m]

    # Ridge: (A^T A + λI) w = A^T y
    lam = float(cfg.ridge_lambda)
    ATA = A_tr.T @ A_tr
    ATy = A_tr.T @ y_tr
    ATA_reg = ATA + lam * np.eye(ATA.shape[0])

    w = np.linalg.solve(ATA_reg, ATy)

    def predict(r_m_q: np.ndarray, f_hz_q: np.ndarray) -> np.ndarray:
        rq = np.asarray(r_m_q, dtype=np.float64).reshape(-1)
        fq = np.asarray(f_hz_q, dtype=np.float64).reshape(-1)
        cols_q = [np.ones_like(rq)]
        if cfg.include_log_spreading:
            cols_q.append(np.log10(np.maximum(rq, 1e-3)))
        if cfg.include_linear_range:
            cols_q.append(rq)
        if cfg.include_thorp_absorption:
            alpha_db_per_m_q = thorp_alpha_db_per_km(fq) / 1000.0
            cols_q.append(alpha_db_per_m_q * rq)
        Aq = np.stack(cols_q, axis=1)
        return (Aq @ w).reshape(-1)

    info = {"type": "parametric_tl", "feature_names": names, "weights": w.tolist(), "config": cfg.__dict__}
    return info, predict


# -----------------------------------------------------------------------------
# Spline-based GAM baseline
# -----------------------------------------------------------------------------


def _second_difference_penalty(n: int) -> np.ndarray:
    """Return D2^T D2 for a length-n coefficient vector."""
    if n <= 2:
        return np.zeros((n, n), dtype=np.float64)
    D = np.zeros((n - 2, n), dtype=np.float64)
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D.T @ D


@dataclass
class GAMConfig:
    n_basis_r: int = 23
    n_basis_f: int = 23
    degree: int = 3
    lambda_r: float = 1e-2
    lambda_f: float = 1e-2
    ridge_lambda: float = 1e-8


def fit_gam_spline(
    r_norm: np.ndarray,
    f_norm: np.ndarray,
    y_db: np.ndarray,
    train_mask: np.ndarray,
    cfg: Optional[GAMConfig] = None,
) -> Tuple[Dict[str, object], Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """Fit a spline GAM: y ≈ b + g_r(r) + g_f(f).

    Parameters
    ----------
    r_norm, f_norm:
        Normalized to roughly [0,1].
    """
    if cfg is None:
        cfg = GAMConfig()

    r = np.asarray(r_norm, dtype=np.float64).reshape(-1)
    f = np.asarray(f_norm, dtype=np.float64).reshape(-1)
    y = np.asarray(y_db, dtype=np.float64).reshape(-1)
    m = np.asarray(train_mask, dtype=bool).reshape(-1)

    # Basis matrices
    knots_r = make_open_uniform_knots(cfg.n_basis_r, cfg.degree, 0.0, 1.0)
    knots_f = make_open_uniform_knots(cfg.n_basis_f, cfg.degree, 0.0, 1.0)
    Br = bspline_basis_matrix_np(r, knots_r, cfg.degree)  # (N, nr)
    Bf = bspline_basis_matrix_np(f, knots_f, cfg.degree)  # (N, nf)

    # Drop the first basis function from each to reduce collinearity with intercept
    Br = Br[:, 1:]
    Bf = Bf[:, 1:]

    A = np.concatenate([np.ones((r.size, 1)), Br, Bf], axis=1)  # (N, 1+nr-1+nf-1)
    A_tr = A[m]
    y_tr = y[m]

    # Penalty: smoothness on Br/Bf coefficients
    nr = Br.shape[1]
    nf = Bf.shape[1]
    P = np.zeros((1 + nr + nf, 1 + nr + nf), dtype=np.float64)
    P[1 : 1 + nr, 1 : 1 + nr] = float(cfg.lambda_r) * _second_difference_penalty(nr)
    P[1 + nr :, 1 + nr :] = float(cfg.lambda_f) * _second_difference_penalty(nf)
    # Intercept is not penalized; ridge is added later with data-dependent scaling.
    P[0, 0] = 0.0
    ATA = A_tr.T @ A_tr
    ATy = A_tr.T @ y_tr

    # Scale ridge regularization to the data term magnitude for numerical stability
    # across events (N varies by event/split).
    scale = float(np.mean(np.diag(ATA))) if ATA.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    # Add a ridge term proportional to the ATA scale (intercept unpenalized).
    if float(cfg.ridge_lambda) > 0.0:
        P = P + float(cfg.ridge_lambda) * scale * np.eye(P.shape[0])
        P[0, 0] = 0.0

    M = ATA + P
    try:
        w = np.linalg.solve(M, ATy)
    except np.linalg.LinAlgError:
        # Fallback to least-squares solve if the normal equations are ill-conditioned
        w = np.linalg.lstsq(M, ATy, rcond=None)[0]

    def predict(rn: np.ndarray, fn: np.ndarray) -> np.ndarray:
        rn = np.asarray(rn, dtype=np.float64).reshape(-1)
        fn = np.asarray(fn, dtype=np.float64).reshape(-1)
        Brq = bspline_basis_matrix_np(rn, knots_r, cfg.degree)[:, 1:]
        Bfq = bspline_basis_matrix_np(fn, knots_f, cfg.degree)[:, 1:]
        Aq = np.concatenate([np.ones((rn.size, 1)), Brq, Bfq], axis=1)
        return (Aq @ w).reshape(-1)

    info = {
        "type": "gam_spline",
        "config": cfg.__dict__,
        "knots_r": knots_r.tolist(),
        "knots_f": knots_f.tolist(),
        "degree": int(cfg.degree),
        "weights": w.tolist(),
        "dropped_first_basis": True,
    }
    return info, predict


# -----------------------------------------------------------------------------
# Waveguide-invariant striation baseline
# -----------------------------------------------------------------------------


@dataclass
class WaveguideInvariantConfig:
    beta_grid: Tuple[float, ...] = tuple(np.round(np.linspace(0.5, 2.0, 16), 3).tolist())
    n_basis: int = 25
    degree: int = 3
    lambda_smooth: float = 1e-2
    ridge_lambda: float = 1e-8


def fit_waveguide_invariant_striation(
    r_m: np.ndarray,
    f_hz: np.ndarray,
    residual_db: np.ndarray,
    train_mask: np.ndarray,
    cfg: Optional[WaveguideInvariantConfig] = None,
) -> Tuple[Dict[str, object], Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """Fit a simple 1D striation model based on the waveguide invariant.

    Theory sketch
    ------------
    For many shallow-water waveguides, prominent interference ridges satisfy an
    approximate invariant relation:

        f * r^β ≈ const  <=>  log f + β log r ≈ const

    If a residual field is dominated by striations, then it can often be modeled
    as a (possibly oscillatory) function of the scalar variable:

        ξ = log f + β log r.

    This baseline chooses β by grid search and fits a smooth spline g(ξ) to the
    residual.

    Important limitation
    --------------------
    This is a *diagnostic* baseline, not a full coherent modal/ray model. It is
    intentionally simple to serve as a transparent comparator.
    """
    if cfg is None:
        cfg = WaveguideInvariantConfig()

    r = np.asarray(r_m, dtype=np.float64).reshape(-1)
    f = np.asarray(f_hz, dtype=np.float64).reshape(-1)
    e = np.asarray(residual_db, dtype=np.float64).reshape(-1)
    m = np.asarray(train_mask, dtype=bool).reshape(-1)

    eps = 1e-9
    log_r = np.log(np.maximum(r, eps))
    log_f = np.log(np.maximum(f, eps))

    best_beta = None
    best_rmse = np.inf
    best_model = None

    for beta in cfg.beta_grid:
        xi = log_f + float(beta) * log_r
        # Normalize xi to [0,1] for spline stability
        xi_min = float(np.min(xi[m]))
        xi_max = float(np.max(xi[m]))
        xi_n = (xi - xi_min) / (xi_max - xi_min + 1e-12)

        knots = make_open_uniform_knots(cfg.n_basis, cfg.degree, 0.0, 1.0)
        B = bspline_basis_matrix_np(xi_n, knots, cfg.degree)
        # Drop first basis to avoid intercept-like collinearity; residual is zero-mean-ish
        B = B[:, 1:]

        B_tr = B[m]
        e_tr = e[m]
        ATA = B_tr.T @ B_tr
        scale = float(np.mean(np.diag(ATA))) if ATA.size else 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0

        P = float(cfg.lambda_smooth) * _second_difference_penalty(B.shape[1])
        if float(cfg.ridge_lambda) > 0.0:
            P = P + float(cfg.ridge_lambda) * scale * np.eye(B.shape[1])

        M = ATA + P
        try:
            w = np.linalg.solve(M, B_tr.T @ e_tr)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(M, B_tr.T @ e_tr, rcond=None)[0]
        e_hat = (B @ w).reshape(-1)
        cur_rmse = rmse(e_tr, e_hat[m])

        if cur_rmse < best_rmse:
            best_rmse = cur_rmse
            best_beta = float(beta)
            best_model = {
                "beta": float(beta),
                "xi_min": xi_min,
                "xi_max": xi_max,
                "knots": knots.tolist(),
                "degree": int(cfg.degree),
                "weights": w.tolist(),
                "dropped_first_basis": True,
            }

    assert best_beta is not None and best_model is not None

    def predict(r_m_q: np.ndarray, f_hz_q: np.ndarray) -> np.ndarray:
        rq = np.asarray(r_m_q, dtype=np.float64).reshape(-1)
        fq = np.asarray(f_hz_q, dtype=np.float64).reshape(-1)
        log_rq = np.log(np.maximum(rq, eps))
        log_fq = np.log(np.maximum(fq, eps))
        xi = log_fq + best_beta * log_rq
        xi_n = (xi - best_model["xi_min"]) / (best_model["xi_max"] - best_model["xi_min"] + 1e-12)
        knots = np.asarray(best_model["knots"], dtype=np.float64)
        B = bspline_basis_matrix_np(xi_n, knots, int(best_model["degree"]))[:, 1:]
        w = np.asarray(best_model["weights"], dtype=np.float64)
        return (B @ w).reshape(-1)

    info = {"type": "waveguide_invariant", "config": cfg.__dict__, "best_beta": best_beta, "train_rmse": float(best_rmse), "model": best_model}
    return info, predict


# -----------------------------------------------------------------------------
# Simple modal striation baseline (Pekeris-style constant-c approximation)
# -----------------------------------------------------------------------------


@dataclass
class PekerisModalStriationConfig:
    """Configuration for a lightweight modal-interference striation baseline.

    The intent is to provide a *traditional-acoustics-inspired* comparator that
    goes one step beyond the waveguide-invariant diagnostic: we explicitly model
    the residual striation field as interference between a small number of
    propagating normal modes in a constant-depth, constant-sound-speed waveguide
    (Pekeris-style approximation).

    This is still not a full normal-mode solver (e.g., KRAKEN) and is not meant
    to compete with coherent-field models. It is a compact baseline that is easy
    to implement and interpretable.
    """

    water_depth_m: float = 217.0
    c0_m_per_s: float = 1500.0
    mode_m_max: int = 20
    n_pairs: int = 3
    ridge_lambda: float = 1e-6
    pair_strategy: str = "adjacent"  # 'adjacent' or 'all'


def _pekeris_km(f_hz: np.ndarray, m: int, H: float, c0: float) -> np.ndarray:
    """Approximate horizontal wavenumber for mode m in a constant-c waveguide."""
    f = np.asarray(f_hz, dtype=np.float64)
    k = 2.0 * np.pi * f / float(c0)
    kz = float(m) * np.pi / float(H)
    # For evanescent modes, sqrt becomes imaginary; clip at zero to avoid NaNs.
    arg = np.maximum(k * k - kz * kz, 0.0)
    return np.sqrt(arg)


def _candidate_mode_pairs(cfg: PekerisModalStriationConfig) -> List[Tuple[int, int]]:
    M = int(cfg.mode_m_max)
    if M < 2:
        raise ValueError("mode_m_max must be >= 2")
    strat = cfg.pair_strategy.strip().lower()
    pairs: List[Tuple[int, int]] = []
    if strat == "adjacent":
        for m in range(1, M):
            pairs.append((m, m + 1))
    elif strat == "all":
        for m in range(1, M):
            for n in range(m + 1, M + 1):
                pairs.append((m, n))
    else:
        raise KeyError(f"Unknown pair_strategy: {cfg.pair_strategy!r}")
    return pairs


def _fit_modal_pairs_ls(
    r_m: np.ndarray,
    f_hz: np.ndarray,
    residual_db: np.ndarray,
    train_mask: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
    cfg: PekerisModalStriationConfig,
) -> Tuple[np.ndarray, float]:
    """Fit linear coefficients for a fixed set of mode pairs."""
    r = np.asarray(r_m, dtype=np.float64).reshape(-1)
    f = np.asarray(f_hz, dtype=np.float64).reshape(-1)
    e = np.asarray(residual_db, dtype=np.float64).reshape(-1)
    msk = np.asarray(train_mask, dtype=bool).reshape(-1)

    H = float(cfg.water_depth_m)
    c0 = float(cfg.c0_m_per_s)

    cols: List[np.ndarray] = []
    for (m1, m2) in pairs:
        k1 = _pekeris_km(f, int(m1), H, c0)
        k2 = _pekeris_km(f, int(m2), H, c0)
        dk = (k1 - k2).astype(np.float64)
        phase = dk * r
        cols.append(np.cos(phase))
        cols.append(np.sin(phase))

    if not cols:
        raise ValueError("No pairs provided")

    A = np.stack(cols, axis=1)  # (N, 2P)
    A_tr = A[msk]
    e_tr = e[msk]

    lam = float(cfg.ridge_lambda)
    ATA = A_tr.T @ A_tr
    ATy = A_tr.T @ e_tr
    w = np.linalg.solve(ATA + lam * np.eye(ATA.shape[0]), ATy)
    e_hat_tr = (A_tr @ w).reshape(-1)
    cur_rmse = rmse(e_tr, e_hat_tr)
    return w, float(cur_rmse)


def fit_pekeris_modal_striation(
    r_m: np.ndarray,
    f_hz: np.ndarray,
    residual_db: np.ndarray,
    train_mask: np.ndarray,
    cfg: Optional[PekerisModalStriationConfig] = None,
) -> Tuple[Dict[str, object], Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    r"""Fit a compact modal-interference striation baseline.

    The baseline models the residual as a sum of a small number of two-mode
    interference terms:

        e(r,f) \approx \sum_p a_p cos(Δk_p(f) r) + b_p sin(Δk_p(f) r)

    where Δk_p(f) is computed using a constant-c normal-mode approximation.

    Selection strategy
    ------------------
    We use greedy forward selection over candidate mode pairs to choose `n_pairs`
    terms that minimize training RMSE.
    """
    if cfg is None:
        cfg = PekerisModalStriationConfig()

    pairs_all = _candidate_mode_pairs(cfg)
    K = int(cfg.n_pairs)
    if K < 1:
        raise ValueError("n_pairs must be >= 1")

    selected: List[Tuple[int, int]] = []
    remaining = pairs_all.copy()

    best_w: Optional[np.ndarray] = None
    best_rmse = np.inf

    for _ in range(K):
        best_pair = None
        best_pair_w = None
        best_pair_rmse = np.inf

        for cand in remaining:
            trial_pairs = selected + [cand]
            w, cur_rmse = _fit_modal_pairs_ls(r_m, f_hz, residual_db, train_mask, trial_pairs, cfg)
            if cur_rmse < best_pair_rmse:
                best_pair_rmse = cur_rmse
                best_pair = cand
                best_pair_w = w

        if best_pair is None or best_pair_w is None:
            break

        selected.append(best_pair)
        remaining = [p for p in remaining if p != best_pair]
        best_w = best_pair_w
        best_rmse = best_pair_rmse

    if not selected or best_w is None:
        raise RuntimeError("Modal striation baseline failed to select any mode pairs")

    # Cache for prediction
    H = float(cfg.water_depth_m)
    c0 = float(cfg.c0_m_per_s)
    w_final = np.asarray(best_w, dtype=np.float64).reshape(-1)

    # Theory note: For a constant-c Pekeris waveguide in the high-frequency limit,
    # Δk(f) ~ 1/f, implying f*r ≈ const along interference ridges (β≈1).
    beta_theory = 1.0

    def predict(r_m_q: np.ndarray, f_hz_q: np.ndarray) -> np.ndarray:
        rq = np.asarray(r_m_q, dtype=np.float64).reshape(-1)
        fq = np.asarray(f_hz_q, dtype=np.float64).reshape(-1)

        cols_q: List[np.ndarray] = []
        for (m1, m2) in selected:
            k1 = _pekeris_km(fq, int(m1), H, c0)
            k2 = _pekeris_km(fq, int(m2), H, c0)
            dk = (k1 - k2)
            phase = dk * rq
            cols_q.append(np.cos(phase))
            cols_q.append(np.sin(phase))
        A = np.stack(cols_q, axis=1)
        return (A @ w_final).reshape(-1)

    info: Dict[str, object] = {
        "type": "pekeris_modal_striation",
        "config": cfg.__dict__,
        "selected_pairs": [tuple(map(int, p)) for p in selected],
        "weights": w_final.tolist(),
        "train_rmse": float(best_rmse),
        "beta_theory": float(beta_theory),
    }
    return info, predict


# -----------------------------------------------------------------------------
# CTD-profile modal-interference striation baseline (finite-difference modes)
# -----------------------------------------------------------------------------

@dataclass
class ProfileModalStriationConfig:
    """Modal-interference striation baseline using a depth-dependent c(z) profile.

    This is intentionally a *lightweight* (NumPy-only) normal-mode approximation:
    we solve a 1-D finite-difference eigenproblem in the water column with simple
    (pressure-release) Dirichlet boundary conditions at z=0 and z=H. This is not a
    full geoacoustic model, but it is substantially closer to classical waveguide
    thinking than purely statistical coupling fits.

    The baseline fits the residual (after an additive model) as a sparse sum of
    sinusoidal range oscillations with frequency-dependent horizontal wavenumber
    differences Δk_mn(f), analogous to modal-interference ``striations``.

    Notes
    -----
    - Boundary conditions and bottom properties matter in real shallow-water
      propagation. We keep this baseline deliberately simple; its purpose is an
      interpretable reference, not a best-possible physics solver.
    - For speed and robustness we select a small number of mode pairs by greedy
      forward selection on the training set.
    """

    water_depth_m: float = 217.0
    dz_m: float = 1.0

    mode_m_max: int = 20
    n_pairs: int = 3

    ridge_lambda: float = 1e-6
    candidate_pairs: str = "adjacent"  # 'adjacent' or 'all'


def _solve_modes_fd_dirichlet(
    f_hz: float,
    z_grid_m: np.ndarray,
    c_grid_mps: np.ndarray,
    *,
    n_modes: int,
) -> np.ndarray:
    """Solve for horizontal wavenumbers k_r,m(f) using a simple FD eigenproblem.

    Eigenproblem (Dirichlet at both ends):
        φ''(z) + k(z)^2 φ(z) = k_r^2 φ(z)

    Returns the largest positive eigenvalues (k_r^2) converted to k_r.
    """
    f = float(f_hz)
    if f <= 0:
        return np.zeros((0,), dtype=np.float64)

    z = np.asarray(z_grid_m, dtype=np.float64).reshape(-1)
    c = np.asarray(c_grid_mps, dtype=np.float64).reshape(-1)
    if z.size != c.size or z.size < 5:
        raise ValueError("z_grid_m and c_grid_mps must be the same length and reasonably sized.")
    H = float(z[-1] - z[0])
    if H <= 0:
        raise ValueError("Invalid depth grid: non-positive span.")

    dz = float(z[1] - z[0])
    if dz <= 0:
        raise ValueError("Depth grid must be increasing.")

    # Interior points exclude Dirichlet boundaries at 0 and H.
    c_i = c[1:-1]
    if c_i.size < 3:
        return np.zeros((0,), dtype=np.float64)

    omega = 2.0 * np.pi * f
    k2_i = (omega / np.maximum(c_i, 1e-6)) ** 2  # (N,)

    N = int(c_i.size)

    # Second-derivative matrix D2 (central differences) on interior points:
    # φ'' ≈ (φ_{i-1} - 2φ_i + φ_{i+1}) / dz^2
    main = (-2.0 / (dz * dz)) * np.ones(N, dtype=np.float64) + k2_i
    off = (1.0 / (dz * dz)) * np.ones(N - 1, dtype=np.float64)

    A = np.diag(main) + np.diag(off, k=1) + np.diag(off, k=-1)

    eigvals = np.linalg.eigh(A)[0]  # ascending
    eigvals = eigvals[::-1]  # descending: largest first

    # Keep physically meaningful (propagating) modes: k_r^2 > 0
    eigvals = eigvals[eigvals > 0.0]
    if eigvals.size == 0:
        return np.zeros((0,), dtype=np.float64)

    k_r = np.sqrt(eigvals[: int(n_modes)]).astype(np.float64)
    return k_r


def fit_profile_modal_striation(
    r_m: np.ndarray,
    f_hz: np.ndarray,
    residual_db: np.ndarray,
    train_mask: np.ndarray,
    *,
    z_profile_m: np.ndarray,
    c_profile_mps: np.ndarray,
    cfg: Optional[ProfileModalStriationConfig] = None,
) -> Tuple[Dict[str, object], Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """Fit a CTD-profile-based modal-interference striation baseline.

    Parameters
    ----------
    r_m, f_hz:
        Flattened sample coordinates.
    residual_db:
        Residual to fit (y - additive_base).
    train_mask:
        Boolean mask selecting training samples.
    z_profile_m, c_profile_mps:
        Sound-speed profile samples (from CTD). ``z_profile_m`` should be in meters
        and increasing with depth. Values are interpolated to a uniform grid.
    """
    if cfg is None:
        cfg = ProfileModalStriationConfig()

    r = np.asarray(r_m, dtype=np.float64).reshape(-1)
    f = np.asarray(f_hz, dtype=np.float64).reshape(-1)
    y = np.asarray(residual_db, dtype=np.float64).reshape(-1)
    m = np.asarray(train_mask, dtype=bool).reshape(-1)

    if r.size != f.size or r.size != y.size:
        raise ValueError("r_m, f_hz, residual_db must have the same length.")
    if r.size == 0:
        raise ValueError("Empty dataset.")

    H = float(cfg.water_depth_m)
    dz = float(cfg.dz_m)
    if dz <= 0:
        raise ValueError("dz_m must be positive.")

    # Build a uniform depth grid and interpolate the CTD profile.
    z0 = float(np.min(z_profile_m))
    z1 = float(np.max(z_profile_m))
    if z1 < H:
        # If CTD does not reach full depth, we extrapolate the last value downwards.
        z1 = H
    z_grid = np.arange(0.0, H + 0.5 * dz, dz, dtype=np.float64)

    z_prof = np.asarray(z_profile_m, dtype=np.float64).reshape(-1)
    c_prof = np.asarray(c_profile_mps, dtype=np.float64).reshape(-1)
    # Shift profile so that 0 corresponds to surface if profile is given as depth already.
    # We assume z_profile_m is already depth below surface; clamp/interp to [0,H].
    c_grid = np.interp(z_grid, z_prof, c_prof, left=c_prof[0], right=c_prof[-1]).astype(np.float64)

    # Unique frequency grid for mode computations (tones or band grid).
    fu = np.unique(f.astype(np.float64))
    fu.sort()

    # Solve for k_r,m at each unique frequency.
    mode_m_max = int(cfg.mode_m_max)
    if mode_m_max < 2:
        raise ValueError("mode_m_max must be >= 2.")

    kr_list: List[np.ndarray] = []
    min_modes = 10**9
    for ff in fu:
        kr = _solve_modes_fd_dirichlet(float(ff), z_grid, c_grid, n_modes=mode_m_max)
        kr_list.append(kr)
        min_modes = min(min_modes, int(kr.size))

    M = int(min(mode_m_max, min_modes))
    if M < 2:
        raise ValueError(
            f"Not enough propagating modes across frequencies to build striation baseline: "
            f"min_modes={min_modes}. Try increasing frequency band or adjusting H."
        )

    # Stack into (n_f, M)
    K = np.zeros((fu.size, M), dtype=np.float64)
    for i, kr in enumerate(kr_list):
        K[i, :] = kr[:M]

    # Candidate mode pairs
    pairs: List[Tuple[int, int]] = []
    if str(cfg.candidate_pairs).strip().lower() == "all":
        for m1 in range(1, M + 1):
            for m2 in range(m1 + 1, M + 1):
                pairs.append((m1, m2))
    else:
        # Default: adjacent pairs (dominant striations)
        for m1 in range(1, M):
            pairs.append((m1, m1 + 1))

    # Precompute candidate columns
    cols: List[Tuple[Tuple[int, int], np.ndarray, np.ndarray]] = []
    for (m1, m2) in pairs:
        dk_u = (K[:, m1 - 1] - K[:, m2 - 1])  # (n_f,)
        dk = np.interp(f, fu, dk_u).astype(np.float64)
        phase = dk * r
        cols.append(((m1, m2), np.cos(phase), np.sin(phase)))

    def _ridge_fit(selected: List[Tuple[int, int]]) -> np.ndarray:
        # Build design matrix for selected pairs: [cos, sin] per pair
        if not selected:
            return np.zeros((0,), dtype=np.float64)
        ccols: List[np.ndarray] = []
        for (m1, m2) in selected:
            for (pp, cc, ss) in cols:
                if pp == (m1, m2):
                    ccols.append(cc)
                    ccols.append(ss)
                    break
        A = np.stack(ccols, axis=1)  # (N, 2*len(selected))
        A_tr = A[m]
        y_tr = y[m]
        lam = float(cfg.ridge_lambda)
        ATA = A_tr.T @ A_tr
        ATy = A_tr.T @ y_tr
        w = np.linalg.solve(ATA + lam * np.eye(ATA.shape[0]), ATy)
        return w

    def _predict_with(selected: List[Tuple[int, int]], w: np.ndarray, rq: np.ndarray, fq: np.ndarray) -> np.ndarray:
        rq = np.asarray(rq, dtype=np.float64).reshape(-1)
        fq = np.asarray(fq, dtype=np.float64).reshape(-1)
        if not selected:
            return np.zeros_like(rq, dtype=np.float64)
        ccols: List[np.ndarray] = []
        for (m1, m2) in selected:
            dk_u = (K[:, m1 - 1] - K[:, m2 - 1])
            dk = np.interp(fq, fu, dk_u).astype(np.float64)
            phase = dk * rq
            ccols.append(np.cos(phase))
            ccols.append(np.sin(phase))
        A = np.stack(ccols, axis=1)
        return (A @ w).reshape(-1)

    # Greedy forward selection of pairs
    selected: List[Tuple[int, int]] = []
    best_rmse = np.inf
    best_w = np.zeros((0,), dtype=np.float64)

    n_pairs = int(cfg.n_pairs)
    n_pairs = max(0, min(n_pairs, len(pairs)))

    for _ in range(n_pairs):
        best_cand = None
        best_cand_w = None
        best_cand_rmse = best_rmse
        for (m1, m2) in pairs:
            if (m1, m2) in selected:
                continue
            cand_sel = selected + [(m1, m2)]
            w = _ridge_fit(cand_sel)
            yhat_tr = _predict_with(cand_sel, w, r[m], f[m])
            e = rmse(y[m], yhat_tr)
            if e < best_cand_rmse:
                best_cand_rmse = e
                best_cand = (m1, m2)
                best_cand_w = w
        if best_cand is None:
            break
        selected.append(best_cand)
        best_rmse = float(best_cand_rmse)
        best_w = best_cand_w if best_cand_w is not None else best_w

    def predict(r_m_q: np.ndarray, f_hz_q: np.ndarray) -> np.ndarray:
        return _predict_with(selected, best_w, r_m_q, f_hz_q)

    info: Dict[str, object] = {
        "type": "profile_modal_striation",
        "config": cfg.__dict__,
        "selected_pairs": [tuple(map(int, p)) for p in selected],
        "weights": best_w.tolist(),
        "train_rmse": float(best_rmse),
        "mode_m_max_effective": int(M),
        "dz_m": float(dz),
        "water_depth_m": float(H),
        "f_unique_hz": fu.tolist(),
    }
    return info, predict


# -----------------------------------------------------------------------------
# Striation-orientation diagnostic (structure tensor) in (log r, log f)
# -----------------------------------------------------------------------------

def estimate_beta_structure_tensor(
    Z_rf: np.ndarray,
    r_m: np.ndarray,
    f_hz: np.ndarray,
    *,
    use_log_coordinates: bool = True,
    eps: float = 1e-12,
) -> float:
    """Estimate the dominant waveguide-invariant slope parameter β from a 2-D map.

    The waveguide-invariant approximation suggests striations are approximately
    straight lines in (log r, log f):
        log f ≈ -β log r + const.

    We estimate the dominant line orientation using a global structure tensor of
    the map gradients. This is a lightweight alternative to full Radon/Hough
    transforms (no extra dependencies).

    Parameters
    ----------
    Z_rf:
        2-D map with shape (Nr, Nf). Rows correspond to r, columns to f.
    r_m, f_hz:
        Coordinate vectors of length Nr and Nf.
    use_log_coordinates:
        If True (recommended), compute gradients in (log r, log f).

    Returns
    -------
    beta_hat:
        Estimated β. Sign is chosen so that increasing r corresponds to decreasing f
        (β>0 for typical shallow-water striations).
    """
    Z = np.asarray(Z_rf, dtype=np.float64)
    r = np.asarray(r_m, dtype=np.float64).reshape(-1)
    f = np.asarray(f_hz, dtype=np.float64).reshape(-1)
    if Z.ndim != 2 or Z.shape != (r.size, f.size):
        raise ValueError("Z_rf must have shape (len(r_m), len(f_hz)).")

    if use_log_coordinates:
        x = np.log(np.maximum(r, eps))
        y = np.log(np.maximum(f, eps))
    else:
        x = r
        y = f

    # Gradients with respect to axis 0 (x) and axis 1 (y)
    gx, gy = np.gradient(Z, x, y, edge_order=1)

    Jxx = float(np.mean(gx * gx))
    Jyy = float(np.mean(gy * gy))
    Jxy = float(np.mean(gx * gy))

    # Orientation of *gradient* principal direction:
    # tan(2θ) = 2Jxy / (Jxx - Jyy)
    theta_grad = 0.5 * np.arctan2(2.0 * Jxy, (Jxx - Jyy))
    # Striation lines are approximately perpendicular to gradient direction.
    theta_line = theta_grad + 0.5 * np.pi

    slope = float(np.tan(theta_line))  # dy/dx
    beta = -slope  # log f = -β log r + const

    # Make β positive by convention (flip if needed).
    if beta < 0:
        beta = -beta
    return float(beta)
