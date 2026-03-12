"""B-spline utilities (NumPy + PyTorch).

This module provides:
- Open-uniform (clamped) knot construction on [xmin, xmax]
- Cox–de Boor recursion to evaluate B-spline basis functions

Notation
--------
- degree p >= 0 (cubic spline => p=3)
- number of basis functions n_basis >= p+1
- knot vector t has length n_basis + p + 1
"""

from __future__ import annotations

import numpy as np
import torch


def make_open_uniform_knots(
    n_basis: int,
    degree: int,
    xmin: float = 0.0,
    xmax: float = 1.0,
) -> np.ndarray:
    """Construct an open-uniform (clamped) knot vector on [xmin, xmax]."""
    if n_basis < degree + 1:
        raise ValueError(f"n_basis must be >= degree+1. Got n_basis={n_basis}, degree={degree}.")
    if xmax <= xmin:
        raise ValueError("xmax must be > xmin.")
    n_knots = n_basis + degree + 1
    # number of interior knots (excluding the repeated endpoints)
    n_inner = n_knots - 2 * (degree + 1)
    if n_inner < 0:
        raise RuntimeError("Invalid knot count derived from n_basis and degree.")
    if n_inner == 0:
        inner = np.array([], dtype=np.float64)
    else:
        inner = np.linspace(xmin, xmax, n_inner + 2, dtype=np.float64)[1:-1]
    knots = np.concatenate(
        [np.full(degree + 1, xmin, dtype=np.float64), inner, np.full(degree + 1, xmax, dtype=np.float64)]
    )
    return knots


def bspline_basis_matrix_np(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate all B-spline basis functions at x using Cox–de Boor recursion.

    This NumPy implementation mirrors :func:`bspline_basis_matrix_torch` and is
    vectorized over the basis index (loops only over spline degree).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    knots = np.asarray(knots, dtype=np.float64).reshape(-1)
    n_basis = len(knots) - degree - 1
    N = x.shape[0]
    # degree-0 initialization (vectorized)
    left = knots[:n_basis]
    right = knots[1 : n_basis + 1]
    x_col = x[:, None]
    mask = (x_col >= left[None, :]) & (x_col < right[None, :])
    mask[:, -1] = (x >= left[-1]) & (x <= right[-1])
    B = mask.astype(np.float64)

    # recursion (loop only over degree)
    for d in range(1, degree + 1):
        denom1 = knots[d : d + n_basis] - knots[:n_basis]
        denom2 = knots[d + 1 : d + 1 + n_basis] - knots[1 : 1 + n_basis]

        coef1 = np.zeros((N, n_basis), dtype=np.float64)
        m1 = denom1 != 0
        if np.any(m1):
            coef1[:, m1] = (x_col - knots[:n_basis][None, :])[:, m1] / denom1[m1][None, :]

        coef2 = np.zeros((N, n_basis), dtype=np.float64)
        m2 = denom2 != 0
        if np.any(m2):
            coef2[:, m2] = (knots[d + 1 : d + 1 + n_basis][None, :] - x_col)[:, m2] / denom2[m2][None, :]

        B_right = np.concatenate([B[:, 1:], np.zeros((N, 1), dtype=np.float64)], axis=1)
        B = coef1 * B + coef2 * B_right

    return B


def bspline_basis_matrix_torch(x: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """Torch version of :func:`bspline_basis_matrix_np`.

    Notes
    -----
    This implementation is **vectorized over basis index** (loops only over the
    spline degree). It is substantially faster than the original
    basis-by-basis Python loop for the typical settings used in this repo
    (cubic splines with tens of basis functions).

    The returned basis matrix is identical (up to floating point arithmetic)
    to the Cox--de Boor recursion.
    """
    if x.dim() == 2 and x.size(1) == 1:
        x = x[:, 0]
    elif x.dim() != 1:
        raise ValueError("x must have shape (N,) or (N,1).")

    knots = knots.reshape(-1)
    n_basis = knots.numel() - degree - 1
    N = x.numel()
    device = x.device
    dtype = x.dtype

    # ------------------------------------------------------------------
    # degree-0 initialization (vectorized)
    # ------------------------------------------------------------------
    left = knots[:n_basis]
    right = knots[1 : n_basis + 1]
    x_col = x[:, None]

    # Closed-open intervals except last basis which includes the right endpoint.
    mask = (x_col >= left[None, :]) & (x_col < right[None, :])
    mask_last = (x >= left[-1]) & (x <= right[-1])
    mask[:, -1] = mask_last

    B = mask.to(dtype)

    # ------------------------------------------------------------------
    # Cox--de Boor recursion (loop only over degree)
    # ------------------------------------------------------------------
    for d in range(1, degree + 1):
        denom1 = knots[d : d + n_basis] - knots[:n_basis]
        denom2 = knots[d + 1 : d + 1 + n_basis] - knots[1 : 1 + n_basis]

        # Coefficients (avoid divide-by-zero on repeated knots)
        coef1 = torch.zeros((N, n_basis), device=device, dtype=dtype)
        m1 = denom1 != 0
        if bool(m1.any()):
            coef1[:, m1] = (x_col - knots[:n_basis][None, :])[:, m1] / denom1[m1][None, :]

        coef2 = torch.zeros((N, n_basis), device=device, dtype=dtype)
        m2 = denom2 != 0
        if bool(m2.any()):
            coef2[:, m2] = (knots[d + 1 : d + 1 + n_basis][None, :] - x_col)[:, m2] / denom2[m2][None, :]

        # B_{i+1}^{d-1} term (shifted basis; last column is zero)
        B_right = torch.cat([B[:, 1:], torch.zeros((N, 1), device=device, dtype=dtype)], dim=1)
        B = coef1 * B + coef2 * B_right

    return B
