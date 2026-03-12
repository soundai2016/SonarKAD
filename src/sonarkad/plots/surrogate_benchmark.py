"""Synthetic surrogate benchmark plots.

This module generates publication-ready plots for the synthetic surrogate
experiment used to validate the method under controlled conditions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from ..utils.plotting import PALETTE, clean_axes, set_plot_style


def _mean_ci(data: np.ndarray):
    data = np.asarray(data, dtype=np.float64)
    mean = data.mean(axis=0)
    if data.shape[0] > 1:
        std = data.std(axis=0, ddof=1)
        ci = 1.96 * std / np.sqrt(data.shape[0])
    else:
        ci = np.zeros_like(mean)
    return mean, ci


def plot_surrogate_benchmark(results_npz: str | Path, out_path: str | Path, title_prefix: str = "") -> Path:
    """Create the 2x2 panel figure.

    Parameters
    ----------
    results_npz:
        Path to `results_aggregate.npz`.
    out_path:
        Output figure path (.png, .pdf, ...).
    """
    set_plot_style()
    results_npz = Path(results_npz)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(results_npz, allow_pickle=True)

    r_phys = data["r_phys"]
    f_phys = data["f_phys"]

    # Component recovery
    theory_r_base = data["theory_r_base"]
    theory_r_full = data["theory_r_full"]
    theory_f = data["theory_f"]

    learned_r_phys = data["learned_r_phys"]
    learned_f_phys = data["learned_f_phys"]

    # Training loss
    loss_phys = data["loss_sonarkad_phys"]
    loss_rand = data["loss_sonarkad_rand"]
    loss_mlp = data["loss_mlp"]

    # Robustness
    snr_levels = data["snr_levels"]
    rmse_phys = data["rmse_phys"]
    rmse_rand = data["rmse_rand"]
    rmse_mlp = data["rmse_mlp_snr"]

    mean_loss_phys, ci_loss_phys = _mean_ci(loss_phys)
    mean_loss_rand, ci_loss_rand = _mean_ci(loss_rand)
    mean_loss_mlp, ci_loss_mlp = _mean_ci(loss_mlp)

    mean_rmse_phys, ci_rmse_phys = _mean_ci(rmse_phys)
    mean_rmse_rand, ci_rmse_rand = _mean_ci(rmse_rand)
    mean_rmse_mlp, ci_rmse_mlp = _mean_ci(rmse_mlp)

    # Panel layout
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    ax_a, ax_b = axs[0, 0], axs[0, 1]
    ax_c, ax_d = axs[1, 0], axs[1, 1]

    # (a) φ_r(r): disentangling TL term
    ax_a.plot(r_phys, theory_r_base, "--", color=PALETTE.theory, alpha=0.8, label="Theory base: -20log10 r - α(fc)r/1000")
    ax_a.plot(r_phys, theory_r_full, ":", color=PALETTE.theory, alpha=0.8, label="Truth: full range term")
    learned_r_mean = learned_r_phys.mean(axis=0)
    ax_a.plot(r_phys, learned_r_mean, "-", color=PALETTE.sonarkad_phys, label="SonarKAD (K=0; physics init) learned φ_r(r)")
    ax_a.set_title(f"{title_prefix}a  Disentangling Transmission Loss Component".strip())
    ax_a.set_xlabel("Range (m)")
    ax_a.set_ylabel("Component (dB, mean-centered)")
    ax_a.legend(loc="best")
    clean_axes(ax_a)

    # (b) φ_f(f): recovering source spectral feature
    learned_f_mean = learned_f_phys.mean(axis=0)
    ax_b.plot(f_phys, theory_f, ":", color=PALETTE.accent, label="Truth: spectral signature term")
    ax_b.plot(f_phys, learned_f_mean, "-", color=PALETTE.sonarkad_phys, label="SonarKAD (K=0) learned φ_f(f)")
    ax_b.set_title(f"{title_prefix}b  Recovering Source Spectral Features".strip())
    ax_b.set_xlabel("Frequency (Hz)")
    ax_b.set_ylabel("Component (dB, mean-centered)")
    ax_b.legend(loc="best")
    clean_axes(ax_b)

    # (c) convergence
    x = np.arange(len(mean_loss_phys))
    ax_c.plot(x, mean_loss_phys, color=PALETTE.sonarkad_phys, label="SonarKAD (K=0; physics init)")
    ax_c.fill_between(x, mean_loss_phys - ci_loss_phys, mean_loss_phys + ci_loss_phys, color=PALETTE.sonarkad_phys, alpha=0.2)
    ax_c.plot(x, mean_loss_rand, "--", color=PALETTE.sonarkad_rand, label="SonarKAD (K=0; random init)")
    ax_c.fill_between(x, mean_loss_rand - ci_loss_rand, mean_loss_rand + ci_loss_rand, color=PALETTE.sonarkad_rand, alpha=0.2)
    ax_c.plot(x, mean_loss_mlp, "-.", color=PALETTE.mlp, label="MLP baseline")
    ax_c.fill_between(x, mean_loss_mlp - ci_loss_mlp, mean_loss_mlp + ci_loss_mlp, color=PALETTE.mlp, alpha=0.2)
    ax_c.set_yscale("log")
    ax_c.set_title(f"{title_prefix}c  Training Convergence Analysis".strip())
    ax_c.set_xlabel("Training epochs")
    ax_c.set_ylabel("MSE loss (log scale)")
    ax_c.legend(loc="best")
    clean_axes(ax_c)

    # (d) robustness vs label noise
    ax_d.errorbar(snr_levels, mean_rmse_phys, yerr=ci_rmse_phys, marker="o", color=PALETTE.sonarkad_phys, label="SonarKAD (K=0; physics init)")
    ax_d.errorbar(snr_levels, mean_rmse_rand, yerr=ci_rmse_rand, marker="^", linestyle="--", color=PALETTE.sonarkad_rand, label="SonarKAD (K=0; random init)")
    ax_d.errorbar(snr_levels, mean_rmse_mlp, yerr=ci_rmse_mlp, marker="s", linestyle="-.", color=PALETTE.mlp, label="MLP baseline")
    ax_d.set_title(f"{title_prefix}d  Robustness Evaluation vs. Label Noise".strip())
    ax_d.set_xlabel("Label SNR (dB)")
    ax_d.set_ylabel("Test RMSE (dB)")
    ax_d.legend(loc="best")
    clean_axes(ax_d)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
