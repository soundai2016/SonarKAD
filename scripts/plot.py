#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# Plot style
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Palette:
    sonarkad_phys: str = "#0072B2"  # blue
    sonarkad_rand: str = "#009E73"  # green
    mlp: str = "#4D4D4D"  # dark gray
    theory: str = "#000000"  # black
    accent: str = "#D55E00"  # vermilion
    soft_gray: str = "#8A8A8A"


PALETTE = Palette()
SUBCAPTION_SIZE = 11.5
SMALL_TEXT_SIZE = 10.5
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 11.5
LEGEND_SIZE = 11


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11.5,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": 13.5,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "figure.dpi": 300,
            "lines.linewidth": 2.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _try_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        out = float(text)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _format_cell(value: object, ndigits: int = 3) -> str:
    num = _try_float(value)
    if num is None:
        return "" if str(value).strip() == "" else str(value)
    return f"{num:.{ndigits}f}"


def _copy_table(src: Path, dst: Path, *, ndigits: int = 3) -> Path:
    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV table: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rows[0])
        for row in rows[1:]:
            writer.writerow([_format_cell(cell, ndigits=ndigits) for cell in row])
    return dst


def _load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing figure asset: {path}")
    return Image.open(path).convert("RGB")


def _crop_rel(img: Image.Image, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    w, h = img.size
    box = (
        int(max(0, round(x0 * w))),
        int(max(0, round(y0 * h))),
        int(min(w, round(x1 * w))),
        int(min(h, round(y1 * h))),
    )
    return np.asarray(img.crop(box))


def _show_image(ax, image: np.ndarray) -> None:
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _clean_axes(ax, *, grid: bool = True) -> None:
    if grid:
        ax.grid(True, which="major", linestyle=":", linewidth=0.6, color="gray", alpha=0.45)
    ax.tick_params(direction="in", length=4)


def _add_subcaption(ax, label: str, text: str, *, y: float = -0.15, fontsize: float = SUBCAPTION_SIZE) -> None:
    ax.text(
        0.5,
        y,
        f"{label} {text}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
    )


def _clean_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _short_method_name(name: str) -> str:
    mapping = {
        "Parametric TL": "Parametric TL",
        "Additive-only": "Additive only",
        "Spline GAM": "Spline GAM",
        "Waveguide-invariant": "WGI",
        "Modal striation (c0)": "Modal c0",
        "Modal striation (CTD)": "Modal CTD",
        "Unconstrained MLP": "Unconstrained MLP",
        "Proposed (SonarKAD)": "SonarKAD",
    }
    return mapping.get(name, name)


# -----------------------------------------------------------------------------
# B-spline helpers
# -----------------------------------------------------------------------------


def make_open_uniform_knots(n_basis: int, degree: int, xmin: float, xmax: float) -> np.ndarray:
    if n_basis <= degree:
        raise ValueError("n_basis must be greater than degree")
    n_internal = n_basis - degree - 1
    if n_internal > 0:
        internal = np.linspace(xmin, xmax, n_internal + 2)[1:-1]
    else:
        internal = np.array([], dtype=float)
    knots = np.concatenate(
        [
            np.full(degree + 1, xmin, dtype=float),
            internal.astype(float),
            np.full(degree + 1, xmax, dtype=float),
        ]
    )
    return knots


def bspline_basis_matrix_np(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    n_basis = len(knots) - degree - 1
    if n_basis <= 0:
        raise ValueError("Invalid knot vector / degree combination")

    B = np.zeros((x.size, n_basis), dtype=float)

    for i in range(n_basis):
        left = knots[i]
        right = knots[i + 1]
        mask = (x >= left) & (x < right)
        if i == n_basis - 1:
            mask = (x >= left) & (x <= right)
        B[:, i] = mask.astype(float)

    for p in range(1, degree + 1):
        B_new = np.zeros_like(B)
        for i in range(n_basis):
            left_den = knots[i + p] - knots[i]
            right_den = knots[i + p + 1] - knots[i + 1]

            if left_den > 0:
                B_new[:, i] += ((x - knots[i]) / left_den) * B[:, i]
            if i + 1 < n_basis and right_den > 0:
                B_new[:, i] += ((knots[i + p + 1] - x) / right_den) * B[:, i + 1]
        B = B_new
    return B


# -----------------------------------------------------------------------------
# Figure 1: method overview 
# -----------------------------------------------------------------------------


def _draw_topological_contrast(ax) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(0.20, 0.95, "Standard MLP\n(black box)", ha="center", va="top", fontsize=12.5, fontweight="bold")
    inputs = [(0.09, 0.74), (0.09, 0.48), (0.09, 0.22)]
    hiddens = [(0.39, 0.81), (0.39, 0.65), (0.39, 0.49), (0.39, 0.33), (0.39, 0.16)]
    for inp in inputs:
        for hid in hiddens:
            ax.plot([inp[0], hid[0]], [inp[1], hid[1]], color="0.75", alpha=0.45, lw=1.2)
    for pos in inputs:
        ax.add_patch(patches.Circle(pos, 0.038, facecolor="white", edgecolor="gray", lw=2.0))
    for pos in hiddens:
        ax.add_patch(patches.Circle(pos, 0.036, facecolor="#E2E2E2", edgecolor="gray", lw=2.0))
    ax.text(
        0.21,
        0.06,
        "Global matrix multiplication\nfeatures entangled",
        ha="center",
        va="bottom",
        fontsize=11.0,
        style="italic",
        color=PALETTE.accent,
    )

    ax.plot([0.50, 0.50], [0.08, 0.88], color="0.20", linestyle="--", lw=1.4)

    ax.text(0.75, 0.95, "SonarKAD\n(additive + $\\psi$)", ha="center", va="top", fontsize=12.5, fontweight="bold")
    s_inputs = [(0.60, 0.74), (0.60, 0.48), (0.60, 0.22)]
    s_sum = (0.90, 0.48)
    labels = [r"$r$", r"$f$", r"$\theta$"]
    funcs = [r"$\phi_r$", r"$\phi_f$", r"$\phi_\theta$"]

    for i, inp in enumerate(s_inputs):
        x = np.linspace(inp[0], s_sum[0], 80)
        y = np.linspace(inp[1], s_sum[1], 80)
        if i == 1:
            curve = np.zeros_like(x)
        else:
            direction = 1.0 if i == 0 else -1.0
            curve = 0.085 * np.sin(np.linspace(0, np.pi, x.size)) * direction
        ax.plot(x, y + curve, color=PALETTE.sonarkad_phys, lw=3.0, zorder=1)
        mid = x.size // 2
        dy = 0.065 if i == 0 else (0.055 if i == 1 else -0.090)
        ax.text(
            x[mid],
            y[mid] + curve[mid] + dy,
            funcs[i],
            color=PALETTE.sonarkad_phys,
            fontsize=13.0,
            ha="center",
            fontweight="bold",
        )

    for i, pos in enumerate(s_inputs):
        ax.add_patch(patches.Circle(pos, 0.038, facecolor="white", edgecolor=PALETTE.sonarkad_phys, lw=2.8, zorder=2))
        ax.text(pos[0] - 0.055, pos[1], labels[i], ha="right", va="center", fontsize=14)

    ax.add_patch(patches.Circle(s_sum, 0.048, facecolor="white", edgecolor="black", lw=2.8, zorder=2))
    ax.text(s_sum[0], s_sum[1], r"$\Sigma$", ha="center", va="center", fontsize=15)
    ax.text(
        0.77,
        0.06,
        "Univariate additivity\ninterpretable pathways",
        ha="center",
        va="bottom",
        fontsize=11.0,
        style="italic",
        color=PALETTE.sonarkad_rand,
    )


def _draw_bspline_microview(ax) -> None:
    x = np.linspace(0.0, 1.0, 500)
    n_basis = 11
    degree = 3
    knots = make_open_uniform_knots(n_basis=n_basis, degree=degree, xmin=0.0, xmax=1.0)
    B = bspline_basis_matrix_np(x, knots, degree)

    coeffs = np.array([0.60, 0.51, 0.42, 0.33, 0.24, 0.15, 0.06, -0.04, -0.12, -0.22, -0.40], dtype=float)
    phi = B @ coeffs

    for i in range(n_basis):
        bi = B[:, i]
        ax.plot(x, bi, color="gray", linestyle=":", alpha=0.45, lw=1.2)

        wi = coeffs[i]
        fill_color = PALETTE.accent if wi >= 0 else PALETTE.sonarkad_rand
        ax.fill_between(x, 0.0, wi * bi, color=fill_color, alpha=0.16)

        left = knots[i]
        right = knots[i + degree + 1]
        center = 0.5 * (left + right)
        ax.plot([center, center], [0.0, wi], color=fill_color, linestyle="--", alpha=0.6, lw=1.8)
        ax.plot(center, wi, marker="o", color=fill_color, markersize=7, zorder=4)

    ax.plot(x, phi, color=PALETTE.sonarkad_phys, lw=3.0, zorder=5)
    ax.text(
        0.05,
        0.89,
        r"$\phi(x)=a x + c + \sum_i w_i\,B_i(x)$",
        transform=ax.transAxes,
        fontsize=12.5,
        bbox=dict(facecolor="white", alpha=0.92, edgecolor="0.6", boxstyle="round,pad=0.45"),
    )

    custom_lines = [
        Line2D([0], [0], color=PALETTE.sonarkad_phys, lw=3.0),
        Line2D([0], [0], marker="o", color=PALETTE.accent, linestyle="None", markersize=6.5),
        Line2D([0], [0], color="gray", linestyle=":", lw=1.2),
    ]
    ax.legend(
        custom_lines,
        ["Learned function", r"Spline weights $w_i$", r"B-spline bases $B_i(x)$"],
        loc="upper right",
        frameon=True,
        fontsize=LEGEND_SIZE,
    )
    ax.set_xlabel("Input (normalized)")
    ax.set_ylabel("Activation output")
    ax.set_xlim(-0.05, 1.02)
    ax.set_ylim(-0.48, 1.05)
    _clean_axes(ax)


def _draw_macro_alignment(ax) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    bbox_props = dict(boxstyle="round,pad=0.42", fc="#FAFAFA", ec="black", lw=1.8)
    ax.text(
        0.5,
        0.83,
        r"$\mathrm{SE}=\mathrm{SL}(f)-\mathrm{TL}(r,f)-\mathrm{NL}(f)+\mathrm{DI}(\theta)-\mathrm{DT}$",
        ha="center",
        va="center",
        fontsize=15, 
        bbox=bbox_props,
    )
    ax.text(
        0.5,
        0.66,
        r"(In our 2D simulation, $\mathbf{DI}$ is fixed and $\mathbf{DT}$ is absorbed into the bias.)",
        ha="center",
        va="center",
        fontsize=10.8,
        color="0.45",
    )

    terms = [
        (0.18, PALETTE.accent, r"$\phi_f(f)$", r"$\approx \mathrm{SL}(f)-\mathrm{NL}(f)$"),
        (0.50, PALETTE.sonarkad_phys, r"$\phi_r(r)$", r"$\approx -\mathrm{TL}(r)$"),
        (0.82, PALETTE.sonarkad_rand, r"$\phi_\theta(\theta)$", r"$\approx \mathrm{DI}(\theta)$"),
    ]

    box_w = 0.27
    box_h = 0.26
    box_y = 0.11
    for x_pos, col, term, desc in terms:
        rect = patches.FancyBboxPatch(
            (x_pos - box_w / 2, box_y),
            box_w,
            box_h,
            boxstyle="round,pad=0.03,rounding_size=0.05",
            fc="white",
            ec=col,
            lw=2.8,
        )
        ax.add_patch(rect)
        ax.text(x_pos, box_y + 0.15, term, ha="center", va="center", fontsize=15, color=col, fontweight="bold")
        ax.text(x_pos, box_y + 0.05, desc, ha="center", va="center", fontsize=12, color="#333333")
        ax.annotate(
            "",
            xy=(x_pos, 0.56),
            xytext=(x_pos, box_y + box_h + 0.03),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=2.4, mutation_scale=26),
        )


def render_method_overview(out_dir: Path) -> Path:
    set_plot_style()
    fig = plt.figure(figsize=(14.0, 8.8), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.18], height_ratios=[1.0, 0.76], hspace=0.36, wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    _draw_topological_contrast(ax1)
    _add_subcaption(ax1, "(a)", "Topological contrast: entangled versus disentangled parameterization", y=-0.12)

    ax2 = fig.add_subplot(gs[0, 1])
    _draw_bspline_microview(ax2)
    _add_subcaption(ax2, "(b)", "Learnable edge activation via B-spline basis functions", y=-0.15)

    ax3 = fig.add_subplot(gs[1, :])
    _draw_macro_alignment(ax3)
    _add_subcaption(ax3, "(c)", "Alignment between additive components and the passive sonar equation", y=-0.10)

    out_path = out_dir / "figure_method_overview.png"
    fig.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.08)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# -----------------------------------------------------------------------------
# Figure 2 & 3 Combined: SWellEx-96 Data Overview & Decomposition
# -----------------------------------------------------------------------------


def render_combined_data_and_decomposition(results_dir: Path, out_dir: Path) -> Path:
    # 1. Load Data Overview Image
    src_data = results_dir / "figure_swellex96_data_overview.png"
    img_data = _load_image(src_data)

    data_panels = [
        _crop_rel(img_data, 0.000, 0.080, 0.500, 0.500),  # S5 tonal RL
        _crop_rel(img_data, 0.500, 0.080, 1.000, 0.500),  # S5 range
        _crop_rel(img_data, 0.000, 0.520, 0.500, 1.000),  # S59 tonal RL
        _crop_rel(img_data, 0.500, 0.520, 1.000, 1.000),  # S59 range
    ]
    data_captions = [
        "Event S5 tonal received-level field",
        "Event S5 source–receiver range evolution",
        "Event S59 tonal received-level field",
        "Event S59 source–receiver range evolution",
    ]

    # 2. Load Decomposition Image
    src_decomp = results_dir / "figure_swellex96_decomposition.png"
    img_decomp = _load_image(src_decomp)

    # Note: 第一行（频域和 S5）的 y0 改为 0.080
    decomp_panels = [
        _crop_rel(img_decomp, 0.000, 0.080, 0.500, 0.500),  # frequency component
        _crop_rel(img_decomp, 0.500, 0.080, 1.000, 0.500),  # S5 psi map + colorbar
        _crop_rel(img_decomp, 0.000, 0.520, 0.500, 1.000),  # range component
        _crop_rel(img_decomp, 0.500, 0.520, 1.000, 1.000),  # S59 psi map + colorbar
    ]
    decomp_captions = [
        "Frequency-only additive component across events",
        "Event S5 residual interaction map and striation geometry",
        "Range-only additive component across events",
        "Event S59 residual interaction map and striation geometry",
    ]

    panels = data_panels + decomp_panels
    captions = data_captions + decomp_captions
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

    set_plot_style()
    fig = plt.figure(figsize=(14.0, 17.5), facecolor="white")
    gs = fig.add_gridspec(4, 2, width_ratios=[1.0, 1.0], hspace=0.15, wspace=0.05)
    axes = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1]),
    ]

    for ax, panel, label, caption in zip(axes, panels, labels, captions):
        _show_image(ax, panel)
        _add_subcaption(ax, label, caption, y=-0.10)

    out_path = out_dir / "figure_swellex96_combined_data_and_decomposition.png"
    fig.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.06)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# -----------------------------------------------------------------------------
# Figure 4: two-event metrics, diagnostics, and rank ablation 
# -----------------------------------------------------------------------------


def _bar_positions(n: int) -> np.ndarray:
    return np.arange(n, dtype=float)


def _set_method_xticks(ax, methods: list[str]) -> None:
    ax.set_xticks(_bar_positions(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=10.5)


def render_two_event_metrics_and_rank_ablation(results_dir: Path, out_dir: Path) -> Path:
    metrics_rows = _read_csv_rows(results_dir / "table_metrics_two_events.csv")
    diag_rows = _read_csv_rows(results_dir / "table_diagnostics_two_events.csv")
    rank_rows = _read_csv_rows(results_dir / "rank_ablation" / "rank_ablation.csv")

    selected_rank = None
    summary_path = results_dir / "rank_ablation" / "rank_ablation_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            selected_rank = _try_float(json.load(f).get("selected_rank"))

    methods = [_short_method_name(r["method"]) for r in metrics_rows]
    x = _bar_positions(len(methods))
    w = 0.36

    s5_rmse = np.array([float(r["rmse_S5_mean"]) for r in metrics_rows], dtype=float)
    s5_rmse_std = np.array([float(r["rmse_S5_std"]) for r in metrics_rows], dtype=float)
    s59_rmse = np.array([float(r["rmse_S59_mean"]) for r in metrics_rows], dtype=float)
    s59_rmse_std = np.array([float(r["rmse_S59_std"]) for r in metrics_rows], dtype=float)

    s5_ev = np.array([float(r["ev_S5_mean"]) for r in metrics_rows], dtype=float)
    s5_ev_std = np.array([float(r["ev_S5_std"]) for r in metrics_rows], dtype=float)
    s59_ev = np.array([float(r["ev_S59_mean"]) for r in metrics_rows], dtype=float)
    s59_ev_std = np.array([float(r["ev_S59_std"]) for r in metrics_rows], dtype=float)

    coupling_name_map = {
        "Coupling fraction (proposed interaction)": "SonarKAD",
        "Coupling fraction (waveguide-invariant baseline)": "WGI",
        "Coupling fraction (modal striation baseline, const. c)": "Modal c0",
        "Coupling fraction (modal striation baseline, CTD profile)": "Modal CTD",
    }
    beta_name_map = {
        "beta (waveguide-invariant grid search)": "WGI grid",
        "beta (interaction map, structure tensor)": "Interaction",
        "beta (modal baseline, structure tensor)": "Modal c0",
        "beta (profile modal baseline, structure tensor)": "Modal CTD",
    }

    coupling_rows = [r for r in diag_rows if r["diagnostic"] in coupling_name_map]
    coupling_labels = [coupling_name_map[r["diagnostic"]] for r in coupling_rows]
    coupling_x = np.arange(len(coupling_labels), dtype=float)
    coupling_s5 = np.array([float(r["S5_mean"]) for r in coupling_rows], dtype=float)
    coupling_s5_std = np.array([float(r["S5_std"]) for r in coupling_rows], dtype=float)
    coupling_s59 = np.array([float(r["S59_mean"]) for r in coupling_rows], dtype=float)
    coupling_s59_std = np.array([float(r["S59_std"]) for r in coupling_rows], dtype=float)

    beta_rows = []
    for row in diag_rows:
        label = row["diagnostic"]
        if label not in beta_name_map:
            continue
        s5 = _try_float(row.get("S5_mean"))
        s59 = _try_float(row.get("S59_mean"))
        if s5 is None or s59 is None:
            continue
        if not (0.05 <= s5 <= 5.0 and 0.05 <= s59 <= 5.0):
            continue
        beta_rows.append(row)
    beta_labels = [beta_name_map[r["diagnostic"]] for r in beta_rows]
    beta_x = np.arange(len(beta_labels), dtype=float)
    beta_s5 = np.array([float(r["S5_mean"]) for r in beta_rows], dtype=float)
    beta_s5_std = np.array([float(r["S5_std"]) for r in beta_rows], dtype=float)
    beta_s59 = np.array([float(r["S59_mean"]) for r in beta_rows], dtype=float)
    beta_s59_std = np.array([float(r["S59_std"]) for r in beta_rows], dtype=float)

    ks = np.array([int(float(r["K"])) for r in rank_rows], dtype=float)
    rank_rmse = np.array([float(r["rmse_mean"]) for r in rank_rows], dtype=float)
    rank_rmse_std = np.array([float(r["rmse_std"]) for r in rank_rows], dtype=float)
    rank_ev = np.array([float(r["ev_mean"]) for r in rank_rows], dtype=float)
    rank_ev_std = np.array([float(r["ev_std"]) for r in rank_rows], dtype=float)
    rank_coupling = np.array(
        [float(r["coupling_energy_frac_mean"]) if str(r["coupling_energy_frac_mean"]).lower() != "nan" else np.nan for r in rank_rows],
        dtype=float,
    )
    rank_coupling_std = np.array(
        [float(r["coupling_energy_frac_std"]) if str(r["coupling_energy_frac_std"]).lower() != "nan" else np.nan for r in rank_rows],
        dtype=float,
    )

    set_plot_style()
    fig, axs = plt.subplots(2, 3, figsize=(15.2, 8.8), facecolor="white")

    # (a) RMSE
    ax = axs[0, 0]
    ax.bar(x - w / 2, s5_rmse, w, yerr=s5_rmse_std, capsize=2.8, color=PALETTE.sonarkad_phys, label="S5")
    ax.bar(x + w / 2, s59_rmse, w, yerr=s59_rmse_std, capsize=2.8, color=PALETTE.accent, label="S59")
    _set_method_xticks(ax, methods)
    ax.set_ylabel("RMSE (dB)")
    ax.legend(frameon=False, ncol=2, fontsize=10.5, loc="upper right")
    _clean_axes(ax)
    _add_subcaption(ax, "(a)", "Two-event blocked-CV RMSE across baseline methods", y=-0.42)

    # (b) EV
    ax = axs[0, 1]
    ax.bar(x - w / 2, s5_ev, w, yerr=s5_ev_std, capsize=2.8, color=PALETTE.sonarkad_phys, label="S5")
    ax.bar(x + w / 2, s59_ev, w, yerr=s59_ev_std, capsize=2.8, color=PALETTE.accent, label="S59")
    _set_method_xticks(ax, methods)
    ax.set_ylabel("Explained variance")
    ax.set_ylim(0.0, max(1.0, float(np.nanmax([s5_ev.max(), s59_ev.max()]) * 1.28)))
    _clean_axes(ax)
    _add_subcaption(ax, "(b)", "Two-event blocked-CV explained variance across baseline methods", y=-0.42)

    # (c) interaction energy
    ax = axs[0, 2]
    ax.bar(coupling_x - w / 2, coupling_s5, w, yerr=coupling_s5_std, capsize=2.8, color=PALETTE.sonarkad_phys, label="S5")
    ax.bar(coupling_x + w / 2, coupling_s59, w, yerr=coupling_s59_std, capsize=2.8, color=PALETTE.accent, label="S59")
    ax.set_xticks(coupling_x)
    ax.set_xticklabels(coupling_labels, rotation=30, ha="right", fontsize=10.5)
    ax.set_ylabel("Energy fraction")
    _clean_axes(ax)
    _add_subcaption(ax, "(c)", "Estimated interaction energy across interpretable baselines", y=-0.42)

    # (d) beta diagnostics
    ax = axs[1, 0]
    ax.bar(beta_x - w / 2, beta_s5, w, yerr=beta_s5_std, capsize=2.8, color=PALETTE.sonarkad_phys, label="S5")
    ax.bar(beta_x + w / 2, beta_s59, w, yerr=beta_s59_std, capsize=2.8, color=PALETTE.accent, label="S59")
    ax.set_xticks(beta_x)
    ax.set_xticklabels(beta_labels, rotation=30, ha="right", fontsize=10.5)
    ax.set_ylabel(r"$\beta$")
    _clean_axes(ax)
    _add_subcaption(ax, "(d)", "Waveguide-invariant diagnostic parameter estimates", y=-0.38)

    # (e) rank RMSE + EV (twin axis)
    ax = axs[1, 1]
    ax.errorbar(ks, rank_rmse, yerr=rank_rmse_std, marker="o", color=PALETTE.sonarkad_phys, capsize=3, label="RMSE")
    ax.set_xlabel("Interaction rank $K$")
    ax.set_ylabel("RMSE (dB)", color=PALETTE.sonarkad_phys)
    ax.tick_params(axis="y", labelcolor=PALETTE.sonarkad_phys)
    _clean_axes(ax)
    ax2 = ax.twinx()
    ax2.errorbar(ks, rank_ev, yerr=rank_ev_std, marker="s", color=PALETTE.accent, capsize=3, label="Explained variance")
    ax2.set_ylabel("Explained variance", color=PALETTE.accent)
    ax2.tick_params(axis="y", labelcolor=PALETTE.accent)
    ax2.grid(False)
    if selected_rank is not None:
        ax.axvline(float(selected_rank), linestyle="--", linewidth=1.3, color="0.25")
        ax.text(
            0.05,
            0.05,
            f"selected $K$ = {int(selected_rank)}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.2,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75", alpha=0.95),
        )
    lines = [Line2D([0], [0], color=PALETTE.sonarkad_phys, marker="o"), Line2D([0], [0], color=PALETTE.accent, marker="s")]
    ax.legend(lines, ["RMSE", "Explained variance"], loc="upper right", frameon=False, fontsize=10.2)
    _add_subcaption(ax, "(e)", "Rank-ablation trends for RMSE and explained variance", y=-0.38)

    # (f) rank interaction energy
    ax = axs[1, 2]
    mask = np.isfinite(rank_coupling)
    ax.errorbar(
        ks[mask],
        rank_coupling[mask],
        yerr=rank_coupling_std[mask],
        marker="o",
        color=PALETTE.sonarkad_rand,
        capsize=3,
    )
    if selected_rank is not None:
        ax.axvline(float(selected_rank), linestyle="--", linewidth=1.3, color="0.25")
    ax.set_xlabel("Interaction rank $K$")
    ax.set_ylabel("Energy fraction")
    _clean_axes(ax)
    _add_subcaption(ax, "(f)", "Rank-ablation trend for interaction-energy fraction", y=-0.38)

    out_path = out_dir / "figure_swellex96_two_event_metrics_diagnostics_and_rank_ablation.png"
    fig.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.18, hspace=0.60, wspace=0.35)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# -----------------------------------------------------------------------------
# Figure 5: transfer study (1 row x 3 columns)
# -----------------------------------------------------------------------------


def _comparison_bracket(ax, x0: float, x1: float, y: float, h: float, text: str, *, color: str) -> None:
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], color=color, lw=1.7, clip_on=False)
    ax.text((x0 + x1) / 2.0, y + h + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]), text, ha="center", va="bottom", fontsize=10.8, color=color)


def _better_badge(ax, text: str, *, color: str) -> None:
    ax.text(
        0.02,
        0.97,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.6,
        color=color,
        bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="0.75", alpha=0.96),
    )


def render_transfer_study(results_dir: Path, out_dir: Path) -> Path:
    transfer_dir = results_dir / "transfer_study"
    curve_img = _load_image(transfer_dir / "figure_transfer_phi_f.png")
    
    curve_panel = _crop_rel(curve_img, 0.000, 0.000, 1.000, 1.000)

    rows = _read_csv_rows(transfer_dir / "table_transfer.csv")
    if len(rows) < 2:
        raise ValueError("Transfer table must contain both scratch and transfer rows.")

    labels = ["Scratch", "Transfer"]
    rmse = np.array([float(rows[0]["rmse_test"]), float(rows[1]["rmse_test"])], dtype=float)
    ev = np.array([float(rows[0]["ev_test"]), float(rows[1]["ev_test"])], dtype=float)
    x = np.arange(len(labels), dtype=float)
    colors = [PALETTE.accent, PALETTE.sonarkad_rand]

    rmse_delta = rmse[1] - rmse[0]
    rmse_rel = (rmse[1] - rmse[0]) / rmse[0] * 100.0
    ev_delta = ev[1] - ev[0]
    ev_rel = (ev[1] - ev[0]) / ev[0] * 100.0

    set_plot_style()
    fig = plt.figure(figsize=(18.0, 5.2), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.0, 1.0], wspace=0.15)

    # (a) basis transfer comparison
    ax0 = fig.add_subplot(gs[0, 0])
    _show_image(ax0, curve_panel)
    _add_subcaption(ax0, "(a)", "Transferred source-event frequency basis\nversus target-event fits", y=-0.14)

    # (b) RMSE bar chart
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(x, rmse, color=colors, width=0.58)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("RMSE (dB)")
    ax1.set_ylim(0.0, max(rmse) * 1.28)
    _clean_axes(ax1)
    _better_badge(ax1, "↓ lower is better", color=PALETTE.sonarkad_rand)
    for xi, yi in zip(x, rmse):
        ax1.text(xi, yi + 0.02, f"{yi:.3f}", ha="center", va="bottom", fontsize=10.6)
    yb = max(rmse) * 1.05
    _comparison_bracket(ax1, x[0], x[1], yb, 0.10, f"Δ = {rmse_delta:+.3f} dB ({rmse_rel:+.1f}%)", color=PALETTE.sonarkad_rand)
    _add_subcaption(ax1, "(b)", "Target-event predictive error", y=-0.14)

    # (c) EV bar chart
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.bar(x, ev, color=colors, width=0.58)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Explained variance")
    ax2.set_ylim(0.0, max(ev) * 1.35)
    _clean_axes(ax2)
    _better_badge(ax2, "↑ higher is better", color=PALETTE.sonarkad_rand)
    for xi, yi in zip(x, ev):
        ax2.text(xi, yi + 0.010, f"{yi:.3f}", ha="center", va="bottom", fontsize=10.6)
    yc = max(ev) * 1.08
    _comparison_bracket(ax2, x[0], x[1], yc, 0.03, f"Δ = {ev_delta:+.3f} ({ev_rel:+.1f}%)", color=PALETTE.sonarkad_rand)
    _add_subcaption(ax2, "(c)", "Target-event explained variance", y=-0.14)

    out_path = out_dir / "figure_transfer_learning_frequency_basis_and_target_event_comparison.png"
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.20)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def build_outputs(results_dir: Path, out_dir: Path, *, clean: bool = True) -> List[Path]:
    if clean:
        _clean_output_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[Path] = []
    outputs.append(render_method_overview(out_dir))
    
    # Combined Figures 2 and 3 into one operation
    outputs.append(render_combined_data_and_decomposition(results_dir, out_dir))
    
    outputs.append(render_two_event_metrics_and_rank_ablation(results_dir, out_dir))
    outputs.append(render_transfer_study(results_dir, out_dir))

    outputs.append(_copy_table(results_dir / "table_metrics_two_events.csv", out_dir / "table_swellex96_two_event_metrics.csv"))
    outputs.append(_copy_table(results_dir / "table_diagnostics_two_events.csv", out_dir / "table_swellex96_two_event_diagnostics.csv"))

    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render figures and tables from existing result files only.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs" / "results", help="Directory containing existing result assets.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "figures", help="Directory where rebuilt figure/table assets will be written.")
    parser.add_argument("--no-clean", action="store_true", help="Do not delete the output directory before writing new files.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build_outputs(args.results_dir.resolve(), args.out_dir.resolve(), clean=not args.no_clean)
    print(f"[OK] figure/table assets written to: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())