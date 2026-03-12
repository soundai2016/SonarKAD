"""Plotting helpers (JASA-friendly default style)."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Palette:
    # Color-blind friendly palette (Okabe-Ito-esque)
    sonarkad_phys: str = "#0072B2"  # blue
    sonarkad_rand: str = "#009E73"  # green
    mlp: str = "#4D4D4D"  # dark gray
    theory: str = "#000000"  # black
    accent: str = "#D55E00"  # vermilion


PALETTE = Palette()


def set_plot_style() -> None:
    """Set Matplotlib params for consistent figures."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 11
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["lines.linewidth"] = 2.2


def clean_axes(ax) -> None:
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, color="gray", alpha=0.5)
    ax.tick_params(direction="in", length=4)
