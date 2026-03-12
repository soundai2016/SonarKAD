"""Plotting utilities for the SWellEx-96 experiment.

Design goal
-----------
The plots produced here are intentionally aligned with what the JASA/JOE ocean
acoustics community expects to see:

1) A clear view of the *measured* intensity-domain structure (tonal LOFAR-type
   features and the range track).
2) Interpretable model components (range term, frequency term, and the learned
   range–frequency coupling map).
3) A "traditional acoustics" baseline chain (parametric TL, spline GAM,
   waveguide-invariant striation fit, and modal striation fits) with quantitative
   metrics.

All figures are written into the experiment's ``outputs/.../plots`` directory.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from ..utils.torch_compat import torch_load_compat


def _load_components(path: Path) -> dict:
    """Load SonarKAD components from .pt (torch.save) or legacy .npz."""
    path = Path(path)
    if path.suffix.lower() == ".pt":
        # PyTorch 2.6 defaults `weights_only=True`, which can break legacy
        # artifacts containing numpy arrays.
        obj = torch_load_compat(path, map_location="cpu")
        if isinstance(obj, dict):
            return obj
        raise TypeError(f"Unexpected object type in {path}: {type(obj)}")
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}
    raise ValueError(f"Unsupported components file: {path}")

from ..utils.plotting import clean_axes, set_plot_style


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _load_json(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _float_or_none(v: object) -> Optional[float]:
    try:
        x = float(v)  # type: ignore[arg-type]
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return None


def _method_keypairs() -> List[Tuple[str, str]]:
    """Stable method order for metric bar charts."""
    return [
        ("Parametric TL", "parametric_tl"),
        ("Additive-only", "additive"),
        ("Spline GAM", "gam"),
        ("Waveguide-invariant", "waveguide_invariant"),
        ("Modal striation (c0)", "modal_striation"),
        ("Modal striation (CTD)", "modal_striation_profile"),
        ("Unconstrained MLP", "mlp"),
        ("Proposed (SonarKAD)", "sonarkad"),
    ]


def _bar(
    ax,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    ylabel: str,
    title: str,
    yerr: Optional[Sequence[float]] = None,
) -> None:
    if yerr is not None and any(float(e) > 0.0 for e in yerr):
        ax.bar(np.arange(len(values)), list(values), yerr=list(yerr), capsize=3)
    else:
        ax.bar(np.arange(len(values)), list(values))
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(list(labels), rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    clean_axes(ax)


# -----------------------------------------------------------------------------
# Event-level plots
# -----------------------------------------------------------------------------


def plot_processed_overview(processed_npz: str | Path, out_path: str | Path, *, title: str) -> Path:
    """Plot tonal RL heatmap + range track from the prepared dataset."""
    set_plot_style()
    processed_npz = Path(processed_npz)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(processed_npz, allow_pickle=True)
    t_sec = d["t_sec"].astype(np.float64)
    r_m = d["r_m"].astype(np.float64)
    f_hz = d["f_hz"].astype(np.float64)
    rl_db = d["rl_db"].astype(np.float64)

    t_min = t_sec / 60.0

    fig = plt.figure(figsize=(10.5, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])

    # (a) tonal RL heatmap (time × tone frequency)
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(
        rl_db.T,
        aspect="auto",
        origin="lower",
        extent=[t_min.min(), t_min.max(), f_hz.min(), f_hz.max()],
    )
    ax0.set_xlabel("Time (min)")
    ax0.set_ylabel("Frequency (Hz)")
    ax0.set_title("Tonal received level (pooled across channels)")
    fig.colorbar(im, ax=ax0, label="dB")

    # (b) range track
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t_min, r_m / 1000.0)
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Range (km)")
    ax1.set_title("Source–receiver range")
    clean_axes(ax1)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_decomposition(components_npz: str | Path, out_path: str | Path, *, title: str) -> Path:
    """Plot φ_f(f), φ_r(r), and ψ(r,f) as a JOE-style multi-panel figure."""
    set_plot_style()
    components_npz = Path(components_npz)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    d = _load_components(components_npz)
    r = d["grid_r_m"].astype(np.float64)
    f = d["grid_f_hz"].astype(np.float64)
    phi_r = d["phi_r"].astype(np.float64)
    phi_f = d["phi_f"].astype(np.float64)
    phi_abs_rf = d.get("phi_abs_rf", np.zeros((0, 0), dtype=np.float64))
    psi = d.get("psi_rf", np.zeros((0, 0), dtype=np.float64))
    has_psi = np.asarray(psi).size > 0
    has_abs = np.asarray(phi_abs_rf).size > 0

    # Optional waveguide-invariant diagnostics saved during training.
    def _scalar(key: str) -> Optional[float]:
        if key not in d:
            return None
        try:
            v = float(np.asarray(d[key]).reshape(-1)[0])
            return v if np.isfinite(v) else None
        except Exception:
            return None

    beta_wgi_best = _scalar("beta_waveguide_invariant_best")
    beta_inter = _scalar("beta_interaction_tensor")

    fig = plt.figure(figsize=(10.5, 5.2 if has_abs else 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[1.0, 1.0])

    # φ_f
    ax_f = fig.add_subplot(gs[0, 0])
    ax_f.plot(f, phi_f)
    ax_f.set_xlabel("Frequency (Hz)")
    ax_f.set_ylabel(r"$\phi_f(f)$ (dB, centered)")
    ax_f.set_title("Frequency component")
    clean_axes(ax_f)

    # φ_r
    ax_r = fig.add_subplot(gs[1, 0])
    ax_r.plot(r / 1000.0, phi_r)
    ax_r.set_xlabel("Range (km)")
    ax_r.set_ylabel(r"$\phi_r(r)$ (dB, centered)")
    ax_r.set_title("Range component")
    clean_axes(ax_r)

    # Right column: optional absorption heatmap + ψ
    if has_abs:
        ax_abs = fig.add_subplot(gs[0, 1])
        abs2 = np.asarray(phi_abs_rf, dtype=np.float64)
        im_abs = ax_abs.imshow(
            abs2,
            aspect="auto",
            origin="lower",
            extent=[f.min(), f.max(), r.min() / 1000.0, r.max() / 1000.0],
        )
        ax_abs.set_xlabel("Frequency (Hz)")
        ax_abs.set_ylabel("Range (km)")
        ax_abs.set_title(r"Absorption coupling $\phi_{abs}(r,f)$")
        fig.colorbar(im_abs, ax=ax_abs, label="dB")

        ax_p = fig.add_subplot(gs[1, 1])
    else:
        ax_p = fig.add_subplot(gs[:, 1])

    if has_psi:
        psi2 = np.asarray(psi, dtype=np.float64)
        im = ax_p.imshow(
            psi2,
            aspect="auto",
            origin="lower",
            extent=[f.min(), f.max(), r.min() / 1000.0, r.max() / 1000.0],
        )
        ax_p.set_xlabel("Frequency (Hz)")
        ax_p.set_ylabel("Range (km)")
        # Include β (if available) in the title for quick interpretability.
        if beta_wgi_best is not None:
            ax_p.set_title(
                rf"Learned coupling $\psi(r,f)$ with $\xi=\log f+\beta\log r$ overlay ($\beta$={beta_wgi_best:.2f})"
            )
        else:
            ax_p.set_title(r"Learned coupling $\psi(r,f)$")
        fig.colorbar(im, ax=ax_p, label="dB")

        # ------------------------------------------------------------
        # Overlay constant-ξ curves on ψ.
        # ξ = log f + β log r  =>  f(r) = exp(ξ) r^{-β}
        # ------------------------------------------------------------
        beta = beta_wgi_best if beta_wgi_best is not None else beta_inter
        if beta is not None and np.isfinite(beta) and float(beta) > 0.0:
            try:
                r_ref = float(np.median(r))
                fmin = float(np.min(f))
                fmax = float(np.max(f))

                # Choose reference frequencies on a log grid so overlay curves
                # spread evenly across the band.
                n_lines = 5
                f_refs = np.exp(
                    np.linspace(np.log(max(fmin, 1e-6)), np.log(fmax), n_lines + 2)[1:-1]
                )
                xi_vals = np.log(f_refs) + float(beta) * np.log(max(r_ref, 1e-6))

                r_line = np.asarray(r, dtype=np.float64)
                for xi in xi_vals:
                    f_line = np.exp(xi - float(beta) * np.log(np.maximum(r_line, 1e-6)))
                    m = (f_line >= fmin) & (f_line <= fmax)
                    if int(np.sum(m)) >= 2:
                        ax_p.plot(
                            f_line[m],
                            (r_line[m] / 1000.0),
                            linestyle="--",
                            linewidth=1.0,
                            color="w",
                            alpha=0.6,
                        )
            except Exception:
                # Overlay is a diagnostic: never fail the plot.
                pass
    else:
        ax_p.axis("off")
        ax_p.text(0.1, 0.5, "No interaction term (K=0)", fontsize=12)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_metrics(results_json: str | Path, out_path: str | Path, *, title: str) -> Path:
    """Plot RMSE/EV bar charts plus coupling/β diagnostics (if present)."""
    set_plot_style()
    results_json = Path(results_json)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    res = _load_json(results_json)

    # Collect metric vectors
    rmse_labels: List[str] = []
    rmse_vals: List[float] = []
    rmse_stds: List[float] = []
    ev_labels: List[str] = []
    ev_vals: List[float] = []
    ev_stds: List[float] = []

    for lbl, key in _method_keypairs():
        v_rmse = _float_or_none(res.get(f"rmse_test_{key}"))
        v_rmse_std = _float_or_none(res.get(f"rmse_test_{key}_std"))
        v_ev = _float_or_none(res.get(f"ev_test_{key}"))
        v_ev_std = _float_or_none(res.get(f"ev_test_{key}_std"))
        if v_rmse is not None:
            rmse_labels.append(lbl)
            rmse_vals.append(v_rmse)
            rmse_stds.append(float(v_rmse_std) if v_rmse_std is not None else 0.0)
        if v_ev is not None:
            ev_labels.append(lbl)
            ev_vals.append(v_ev)
            ev_stds.append(float(v_ev_std) if v_ev_std is not None else 0.0)

    # Diagnostics
    cpl = _float_or_none(res.get("coupling_energy_frac_test"))
    cpl_wgi = _float_or_none(res.get("coupling_energy_frac_waveguide_invariant_test"))
    cpl_modal = _float_or_none(res.get("coupling_energy_frac_modal_striation_test"))
    cpl_profile = _float_or_none(res.get("coupling_energy_frac_modal_striation_profile_test"))
    beta = res.get("beta_diagnostics", {}) if isinstance(res, dict) else {}
    beta_inter = _float_or_none(beta.get("beta_interaction_tensor")) if isinstance(beta, dict) else None
    beta_wgi = _float_or_none(beta.get("beta_waveguide_invariant_best")) if isinstance(beta, dict) else None
    beta_modal = _float_or_none(beta.get("beta_modal_constant_tensor")) if isinstance(beta, dict) else None
    beta_prof = _float_or_none(beta.get("beta_modal_profile_tensor")) if isinstance(beta, dict) else None

    fig = plt.figure(figsize=(11.2, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85])

    ax0 = fig.add_subplot(gs[0, 0])
    if rmse_vals:
        _bar(ax0, rmse_labels, rmse_vals, ylabel="RMSE (dB)", title="Test RMSE", yerr=rmse_stds)
    else:
        ax0.axis("off")
        ax0.text(0.1, 0.5, "RMSE metrics not found", fontsize=12)

    ax1 = fig.add_subplot(gs[0, 1])
    if ev_vals:
        _bar(ax1, ev_labels, ev_vals, ylabel="Explained variance", title="Test EV", yerr=ev_stds)
    else:
        ax1.axis("off")
        ax1.text(0.1, 0.5, "EV metrics not found", fontsize=12)

    # Diagnostics panel (text + optional β bars)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis("off")
    y0 = 0.80
    dy = 0.17
    if cpl is not None:
        ax2.text(0.02, y0, f"Coupling energy fraction (proposed): {cpl:.3f}", fontsize=12)
        y0 -= dy
    if cpl_wgi is not None:
        ax2.text(0.02, y0, f"Coupling fraction (waveguide-invariant baseline): {cpl_wgi:.3f}", fontsize=11)
        y0 -= dy
    if cpl_modal is not None:
        ax2.text(0.02, y0, f"Coupling fraction (modal striation baseline, const. c): {cpl_modal:.3f}", fontsize=11)
        y0 -= dy
    if cpl_profile is not None:
        ax2.text(0.02, y0, f"Coupling fraction (modal striation baseline, CTD profile): {cpl_profile:.3f}", fontsize=11)
        y0 -= dy

    beta_items: List[Tuple[str, Optional[float]]] = [
        ("β from learned interaction (structure tensor)", beta_inter),
        ("β from waveguide-invariant grid search", beta_wgi),
        ("β from modal baseline (structure tensor)", beta_modal),
        ("β from profile modal baseline (structure tensor)", beta_prof),
    ]
    beta_labels = [lab for lab, val in beta_items if val is not None]
    beta_vals = [float(val) for lab, val in beta_items if val is not None]
    if beta_vals:
        # Draw a small inset bar chart on the right
        inset = ax2.inset_axes([0.55, 0.15, 0.43, 0.78])
        inset.bar(np.arange(len(beta_vals)), beta_vals)
        inset.set_xticks(np.arange(len(beta_vals)))
        inset.set_xticklabels(beta_labels, rotation=25, ha="right")
        inset.set_ylabel("β")
        inset.set_title("Waveguide-invariant diagnostics")
        clean_axes(inset)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_event_report(
    *,
    processed_npz: str | Path,
    components_npz: str | Path,
    results_json: str | Path,
    out_dir: str | Path,
    prefix: str,
    title: str,
) -> Dict[str, Path]:
    """Generate the full set of event-level figures used by the paper."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}
    paths["overview"] = plot_processed_overview(
        processed_npz,
        out_dir / f"{prefix}_overview.png",
        title=title,
    )
    paths["decomposition"] = plot_decomposition(
        components_npz,
        out_dir / f"{prefix}_decomposition.png",
        title=title,
    )
    paths["metrics"] = plot_metrics(
        results_json,
        out_dir / f"{prefix}_metrics.png",
        title=title,
    )
    return paths


# -----------------------------------------------------------------------------
# Paper-level multi-event figures
# -----------------------------------------------------------------------------


def plot_two_event_baseline_summary(
    *,
    results_a: str | Path,
    label_a: str,
    results_b: str | Path,
    label_b: str,
    out_path: str | Path,
) -> Path:
    """Create a compact two-event RMSE/EV comparison (JOE-style)."""
    set_plot_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ra = _load_json(Path(results_a))
    rb = _load_json(Path(results_b))

    methods = _method_keypairs()
    labels = [m[0] for m in methods]

    def vec(res: Dict[str, object], prefix: str) -> Tuple[List[float], List[float]]:
        """Return (mean, std) vectors if available."""
        out: List[float] = []
        out_std: List[float] = []
        for _, key in methods:
            v = _float_or_none(res.get(f"{prefix}_test_{key}"))
            v_std = _float_or_none(res.get(f"{prefix}_test_{key}_std"))
            out.append(float(v) if v is not None else float("nan"))
            out_std.append(float(v_std) if v_std is not None else 0.0)
        return out, out_std

    rmse_a, rmse_a_std = vec(ra, "rmse")
    rmse_b, rmse_b_std = vec(rb, "rmse")
    ev_a, ev_a_std = vec(ra, "ev")
    ev_b, ev_b_std = vec(rb, "ev")

    fig = plt.figure(figsize=(12.0, 6.0))
    gs = fig.add_gridspec(2, 2)

    ax00 = fig.add_subplot(gs[0, 0])
    _bar(ax00, labels, rmse_a, ylabel="RMSE (dB)", title=f"{label_a}: test RMSE", yerr=rmse_a_std)
    ax01 = fig.add_subplot(gs[0, 1])
    _bar(ax01, labels, ev_a, ylabel="Explained variance", title=f"{label_a}: test EV", yerr=ev_a_std)

    ax10 = fig.add_subplot(gs[1, 0])
    _bar(ax10, labels, rmse_b, ylabel="RMSE (dB)", title=f"{label_b}: test RMSE", yerr=rmse_b_std)
    ax11 = fig.add_subplot(gs[1, 1])
    _bar(ax11, labels, ev_b, ylabel="Explained variance", title=f"{label_b}: test EV", yerr=ev_b_std)

    fig.suptitle("Baseline comparison on SWellEx-96 (two events)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_two_event_metrics_csv(
    *,
    results_a: Dict[str, object],
    event_a: str,
    results_b: Dict[str, object],
    event_b: str,
    out_path: str | Path,
    float_fmt: str = "{:.3f}",
) -> Path:
    """Write a two-event metrics summary as a single CSV file.

    Columns are grouped by event and include mean and std across CV folds when
    available (``*_std`` keys).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def get(res: Dict[str, object], k: str) -> Tuple[str, str]:
        v = _float_or_none(res.get(k))
        v_std = _float_or_none(res.get(k + "_std"))
        if v is None:
            return "", ""
        v_str = float_fmt.format(v)
        vstd_str = "" if v_std is None else float_fmt.format(v_std)
        return v_str, vstd_str

    methods = [
        ("Parametric TL", "parametric_tl"),
        ("Additive-only", "additive"),
        ("Spline GAM", "gam"),
        ("Waveguide-invariant", "waveguide_invariant"),
        ("Modal striation (c0)", "modal_striation"),
        ("Modal striation (CTD)", "modal_striation_profile"),
        ("Unconstrained MLP", "mlp"),
        ("Proposed (SonarKAD)", "sonarkad"),
    ]

    header = [
        "method",
        f"rmse_{event_a}_mean", f"rmse_{event_a}_std", f"ev_{event_a}_mean", f"ev_{event_a}_std",
        f"rmse_{event_b}_mean", f"rmse_{event_b}_std", f"ev_{event_b}_mean", f"ev_{event_b}_std",
    ]

    rows: List[List[str]] = []
    for name, key in methods:
        rmse_a, rmse_a_std = get(results_a, f"rmse_test_{key}")
        ev_a, ev_a_std = get(results_a, f"ev_test_{key}")
        rmse_b, rmse_b_std = get(results_b, f"rmse_test_{key}")
        ev_b, ev_b_std = get(results_b, f"ev_test_{key}")
        rows.append([name, rmse_a, rmse_a_std, ev_a, ev_a_std, rmse_b, rmse_b_std, ev_b, ev_b_std])

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return out_path


def write_two_event_diagnostics_csv(
    *,
    results_a: Dict[str, object],
    event_a: str,
    results_b: Dict[str, object],
    event_b: str,
    out_path: str | Path,
    float_fmt: str = "{:.3f}",
) -> Path:
    """Write a two-event diagnostics summary as a single CSV file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def get(res: Dict[str, object], k: str) -> Tuple[str, str]:
        v = _float_or_none(res.get(k))
        v_std = _float_or_none(res.get(k + "_std"))
        if v is None:
            return "", ""
        v_str = float_fmt.format(v)
        vstd_str = "" if v_std is None else float_fmt.format(v_std)
        return v_str, vstd_str

    def get_beta(res: Dict[str, object], k: str) -> Tuple[str, str]:
        bd = res.get("beta_diagnostics", {})
        if not isinstance(bd, dict):
            return "", ""
        v = _float_or_none(bd.get(k))
        v_std = _float_or_none(bd.get(k + "_std"))
        if v is None:
            return "", ""
        v_str = float_fmt.format(v)
        vstd_str = "" if v_std is None else float_fmt.format(v_std)
        return v_str, vstd_str

    items = [
        ("coupling_energy_frac_test", "Coupling fraction (proposed interaction)", "metric"),
        ("coupling_energy_frac_waveguide_invariant_test", "Coupling fraction (waveguide-invariant baseline)", "metric"),
        ("coupling_energy_frac_modal_striation_test", "Coupling fraction (modal striation baseline, const. c)", "metric"),
        ("coupling_energy_frac_modal_striation_profile_test", "Coupling fraction (modal striation baseline, CTD profile)", "metric"),
        ("beta_interaction_tensor", "beta (interaction map, structure tensor)", "beta"),
        ("beta_waveguide_invariant_best", "beta (waveguide-invariant grid search)", "beta"),
        ("beta_modal_constant_tensor", "beta (modal baseline, structure tensor)", "beta"),
        ("beta_modal_profile_tensor", "beta (profile modal baseline, structure tensor)", "beta"),
    ]

    header = ["diagnostic", f"{event_a}_mean", f"{event_a}_std", f"{event_b}_mean", f"{event_b}_std"]
    rows: List[List[str]] = []
    for key, name, kind in items:
        if kind == "beta":
            a, a_std = get_beta(results_a, key)
            b, b_std = get_beta(results_b, key)
        else:
            a, a_std = get(results_a, key)
            b, b_std = get(results_b, key)
        rows.append([name, a, a_std, b, b_std])

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return out_path

def plot_two_event_overview(
    *,
    processed_a: str | Path,
    label_a: str,
    processed_b: str | Path,
    label_b: str,
    out_path: str | Path,
) -> Path:
    """Two-event overview: tonal RL heatmaps and range tracks (4 panels)."""
    set_plot_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def load(proc: str | Path):
        d = np.load(Path(proc), allow_pickle=True)
        return (
            d["t_sec"].astype(np.float64) / 60.0,
            d["r_m"].astype(np.float64) / 1000.0,
            d["f_hz"].astype(np.float64),
            d["rl_db"].astype(np.float64),
        )

    tA, rA, fA, rlA = load(processed_a)
    tB, rB, fB, rlB = load(processed_b)

    fig = plt.figure(figsize=(12.2, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0])

    # A heatmap
    ax00 = fig.add_subplot(gs[0, 0])
    rlA_m = np.ma.masked_invalid(rlA.T)
    imA = ax00.imshow(
        rlA_m,
        aspect="auto",
        origin="lower",
        extent=[tA.min(), tA.max(), fA.min(), fA.max()],
    )
    ax00.set_ylabel("Frequency (Hz)")
    ax00.set_title(f"{label_a}: tonal RL")
    fig.colorbar(imA, ax=ax00, label="dB")

    ax01 = fig.add_subplot(gs[0, 1])
    ax01.plot(tA, rA)
    ax01.set_ylabel("Range (km)")
    ax01.set_title(f"{label_a}: range")
    clean_axes(ax01)

    # B heatmap
    ax10 = fig.add_subplot(gs[1, 0])
    rlB_m = np.ma.masked_invalid(rlB.T)
    imB = ax10.imshow(
        rlB_m,
        aspect="auto",
        origin="lower",
        extent=[tB.min(), tB.max(), fB.min(), fB.max()],
    )
    ax10.set_xlabel("Time (min)")
    ax10.set_ylabel("Frequency (Hz)")
    ax10.set_title(f"{label_b}: tonal RL")
    fig.colorbar(imB, ax=ax10, label="dB")

    ax11 = fig.add_subplot(gs[1, 1])
    ax11.plot(tB, rB)
    ax11.set_xlabel("Time (min)")
    ax11.set_ylabel("Range (km)")
    ax11.set_title(f"{label_b}: range")
    clean_axes(ax11)

    fig.suptitle("SWellEx-96 data overview")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_two_event_decomposition(
    *,
    components_a: str | Path,
    label_a: str,
    components_b: str | Path,
    label_b: str,
    out_path: str | Path,
) -> Path:
    """Two-event decomposition: overlay φ_f/φ_r and show ψ maps (4 panels)."""
    set_plot_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def load(comp: str | Path):
        d = _load_components(Path(comp))

        def _scalar(key: str) -> Optional[float]:
            if key not in d:
                return None
            try:
                v = float(np.asarray(d[key]).reshape(-1)[0])
                return v if np.isfinite(v) else None
            except Exception:
                return None

        return (
            d["grid_r_m"].astype(np.float64),
            d["grid_f_hz"].astype(np.float64),
            d["phi_r"].astype(np.float64),
            d["phi_f"].astype(np.float64),
            np.asarray(d.get("psi_rf", np.zeros((0, 0), dtype=np.float64)), dtype=np.float64),
            _scalar("beta_waveguide_invariant_best"),
            _scalar("beta_interaction_tensor"),
        )

    rA, fA, prA, pfA, psiA, betaA_wgi, betaA_inter = load(components_a)
    rB, fB, prB, pfB, psiB, betaB_wgi, betaB_inter = load(components_b)

    def overlay_xi(ax, r_m: np.ndarray, f_hz: np.ndarray, beta: Optional[float]):
        if beta is None or (not np.isfinite(beta)) or float(beta) <= 0.0:
            return
        try:
            r_ref = float(np.median(r_m))
            fmin = float(np.min(f_hz))
            fmax = float(np.max(f_hz))
            n_lines = 5
            f_refs = np.exp(
                np.linspace(np.log(max(fmin, 1e-6)), np.log(fmax), n_lines + 2)[1:-1]
            )
            xi_vals = np.log(f_refs) + float(beta) * np.log(max(r_ref, 1e-6))
            r_line = np.asarray(r_m, dtype=np.float64)
            for xi in xi_vals:
                f_line = np.exp(xi - float(beta) * np.log(np.maximum(r_line, 1e-6)))
                m = (f_line >= fmin) & (f_line <= fmax)
                if int(np.sum(m)) >= 2:
                    ax.plot(
                        f_line[m],
                        (r_line[m] / 1000.0),
                        linestyle="--",
                        linewidth=1.0,
                        color="w",
                        alpha=0.6,
                    )
        except Exception:
            return

    fig = plt.figure(figsize=(12.2, 6.2))
    gs = fig.add_gridspec(2, 2)

    ax00 = fig.add_subplot(gs[0, 0])
    ax00.plot(fA, pfA, label=label_a)
    ax00.plot(fB, pfB, label=label_b)
    ax00.set_xlabel("Frequency (Hz)")
    ax00.set_ylabel(r"$\phi_f(f)$ (dB, centered)")
    ax00.set_title("Frequency component")
    ax00.legend(loc="best")
    clean_axes(ax00)

    ax01 = fig.add_subplot(gs[1, 0])
    ax01.plot(rA / 1000.0, prA, label=label_a)
    ax01.plot(rB / 1000.0, prB, label=label_b)
    ax01.set_xlabel("Range (km)")
    ax01.set_ylabel(r"$\phi_r(r)$ (dB, centered)")
    ax01.set_title("Range component")
    ax01.legend(loc="best")
    clean_axes(ax01)

    ax10 = fig.add_subplot(gs[0, 1])
    if psiA.size:
        imA = ax10.imshow(
            psiA,
            aspect="auto",
            origin="lower",
            extent=[fA.min(), fA.max(), rA.min() / 1000.0, rA.max() / 1000.0],
        )
        ax10.set_xlabel("Frequency (Hz)")
        ax10.set_ylabel("Range (km)")
        betaA = betaA_wgi if betaA_wgi is not None else betaA_inter
        if betaA is not None:
            ax10.set_title(fr"{label_a}: $\psi(r,f)$ (β={float(betaA):.2f})")
        else:
            ax10.set_title(fr"{label_a}: $\psi(r,f)$")
        fig.colorbar(imA, ax=ax10, label="dB")
        overlay_xi(ax10, rA, fA, betaA)
    else:
        ax10.axis("off")
        ax10.text(0.1, 0.5, f"{label_a}: no interaction", fontsize=12)

    ax11 = fig.add_subplot(gs[1, 1])
    if psiB.size:
        imB = ax11.imshow(
            psiB,
            aspect="auto",
            origin="lower",
            extent=[fB.min(), fB.max(), rB.min() / 1000.0, rB.max() / 1000.0],
        )
        ax11.set_xlabel("Frequency (Hz)")
        ax11.set_ylabel("Range (km)")
        betaB = betaB_wgi if betaB_wgi is not None else betaB_inter
        if betaB is not None:
            ax11.set_title(fr"{label_b}: $\psi(r,f)$ (β={float(betaB):.2f})")
        else:
            ax11.set_title(fr"{label_b}: $\psi(r,f)$")
        fig.colorbar(imB, ax=ax11, label="dB")
        overlay_xi(ax11, rB, fB, betaB)
    else:
        ax11.axis("off")
        ax11.text(0.1, 0.5, f"{label_b}: no interaction", fontsize=12)

    fig.suptitle("Interpretable decomposition on SWellEx-96")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_two_event_diagnostics_summary(
    *,
    results_a: str | Path,
    label_a: str,
    results_b: str | Path,
    label_b: str,
    out_path: str | Path,
) -> Path:
    """Two-event diagnostic summary: coupling fractions and β estimates.

    This figure is intended to make the ``waveguide invariant'' connection
    explicit: if the learned interaction map captures striation geometry, the
    estimated β values should be consistent with traditional striation fits.
    """
    set_plot_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ra = _load_json(Path(results_a))
    rb = _load_json(Path(results_b))

    def coupling_vec(res: Dict[str, object]) -> Tuple[List[str], List[float]]:
        items = [
            ("Proposed", _float_or_none(res.get("coupling_energy_frac_test"))),
            ("WGI", _float_or_none(res.get("coupling_energy_frac_waveguide_invariant_test"))),
            ("Modal (c0)", _float_or_none(res.get("coupling_energy_frac_modal_striation_test"))),
            ("Modal (CTD)", _float_or_none(res.get("coupling_energy_frac_modal_striation_profile_test"))),
        ]
        labs = [k for k, v in items if v is not None]
        vals = [float(v) for k, v in items if v is not None]
        return labs, vals

    def beta_vec(res: Dict[str, object]) -> Tuple[List[str], List[float]]:
        bd = res.get("beta_diagnostics", {})
        if not isinstance(bd, dict):
            return [], []
        items = [
            (r"$\beta$ (interaction)", _float_or_none(bd.get("beta_interaction_tensor"))),
            (r"$\beta$ (WGI grid)", _float_or_none(bd.get("beta_waveguide_invariant_best"))),
            (r"$\beta$ (modal c0)", _float_or_none(bd.get("beta_modal_constant_tensor"))),
            (r"$\beta$ (modal CTD)", _float_or_none(bd.get("beta_modal_profile_tensor"))),
        ]
        labs = [k for k, v in items if v is not None]
        vals = [float(v) for k, v in items if v is not None]
        return labs, vals

    labs_c_a, vals_c_a = coupling_vec(ra)
    labs_c_b, vals_c_b = coupling_vec(rb)
    labs_b_a, vals_b_a = beta_vec(ra)
    labs_b_b, vals_b_b = beta_vec(rb)

    fig = plt.figure(figsize=(12.6, 6.2))
    gs = fig.add_gridspec(2, 2)

    ax00 = fig.add_subplot(gs[0, 0])
    if vals_c_a:
        _bar(ax00, labs_c_a, vals_c_a, ylabel="Energy fraction", title=f"{label_a}: coupling fraction")
    else:
        ax00.axis("off")
        ax00.text(0.1, 0.5, "Coupling metrics not found", fontsize=12)

    ax01 = fig.add_subplot(gs[0, 1])
    if vals_c_b:
        _bar(ax01, labs_c_b, vals_c_b, ylabel="Energy fraction", title=f"{label_b}: coupling fraction")
    else:
        ax01.axis("off")
        ax01.text(0.1, 0.5, "Coupling metrics not found", fontsize=12)

    ax10 = fig.add_subplot(gs[1, 0])
    if vals_b_a:
        _bar(ax10, labs_b_a, vals_b_a, ylabel=r"$\beta$", title=f"{label_a}: waveguide-invariant diagnostics")
    else:
        ax10.axis("off")
        ax10.text(0.1, 0.5, r"$\beta$ diagnostics not found", fontsize=12)

    ax11 = fig.add_subplot(gs[1, 1])
    if vals_b_b:
        _bar(ax11, labs_b_b, vals_b_b, ylabel=r"$\beta$", title=f"{label_b}: waveguide-invariant diagnostics")
    else:
        ax11.axis("off")
        ax11.text(0.1, 0.5, r"$\beta$ diagnostics not found", fontsize=12)

    fig.suptitle("Coupling and waveguide-invariant diagnostics (two events)")
    try:
        fig.tight_layout()
    except Exception as e:
        # tight_layout can fail if mathtext parsing fails or labels are too long;
        # fall back to a conservative layout rather than crashing the whole pipeline.
        print(f"[WARN] tight_layout failed for diagnostics summary: {e}")
        fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.10, wspace=0.25, hspace=0.35)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path

def plot_two_event_summary(
    *,
    results_a: str | Path,
    label_a: str,
    results_b: str | Path,
    label_b: str,
    out_path: str | Path,
) -> Path:
    """Combined two-event summary: predictive metrics + physics diagnostics.

    The output is a single 2×2 panel figure intended for the manuscript to
    keep the overall paper artifact count small (metrics and diagnostics in
    one place).
    """
    set_plot_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ra = _load_json(Path(results_a))
    rb = _load_json(Path(results_b))

    methods = [
        ("Parametric TL", "parametric_tl"),
        ("Additive-only", "additive"),
        ("Spline GAM", "gam"),
        ("Waveguide-invariant", "waveguide_invariant"),
        ("Modal striation (c0)", "modal_striation"),
        ("Modal striation (CTD)", "modal_striation_profile"),
        ("Unconstrained MLP", "mlp"),
        ("Proposed (SonarKAD)", "sonarkad"),
    ]

    def _get(res: Dict[str, object], k: str) -> Tuple[Optional[float], Optional[float]]:
        v = _float_or_none(res.get(k))
        v_std = _float_or_none(res.get(k + "_std"))
        return v, v_std

    # --- Build metric arrays (only keep methods present in both events) ---
    keep = []
    rmse_a = []
    rmse_b = []
    rmse_a_std = []
    rmse_b_std = []
    ev_a = []
    ev_b = []
    ev_a_std = []
    ev_b_std = []
    for name, key in methods:
        va, va_std = _get(ra, f"rmse_test_{key}")
        vb, vb_std = _get(rb, f"rmse_test_{key}")
        ea, ea_std = _get(ra, f"ev_test_{key}")
        eb, eb_std = _get(rb, f"ev_test_{key}")
        if va is None or vb is None or ea is None or eb is None:
            continue
        keep.append(name)
        rmse_a.append(float(va)); rmse_b.append(float(vb))
        rmse_a_std.append(0.0 if va_std is None else float(va_std))
        rmse_b_std.append(0.0 if vb_std is None else float(vb_std))
        ev_a.append(float(ea)); ev_b.append(float(eb))
        ev_a_std.append(0.0 if ea_std is None else float(ea_std))
        ev_b_std.append(0.0 if eb_std is None else float(eb_std))

    x = np.arange(len(keep), dtype=float)
    w = 0.38

    # Publication-facing layout: give extra vertical room for long x tick labels.
    fig, axs = plt.subplots(2, 2, figsize=(13.6, 7.8), constrained_layout=True)

    # RMSE panel
    ax = axs[0, 0]
    ax.bar(x - w / 2, rmse_a, w, yerr=rmse_a_std, capsize=2.5, label=label_a)
    ax.bar(x + w / 2, rmse_b, w, yerr=rmse_b_std, capsize=2.5, label=label_b)
    ax.set_xticks(x)
    ax.set_xticklabels(keep, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("RMSE (dB)")
    ax.set_title("Predictive error")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9, frameon=False)

    # EV panel
    ax = axs[0, 1]
    ax.bar(x - w / 2, ev_a, w, yerr=ev_a_std, capsize=2.5, label=label_a)
    ax.bar(x + w / 2, ev_b, w, yerr=ev_b_std, capsize=2.5, label=label_b)
    ax.set_xticks(x)
    ax.set_xticklabels(keep, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Explained variance")
    ax.set_title("Predictive skill")
    ax.grid(True, axis="y", alpha=0.3)

    # Coupling fraction panel
    ax = axs[1, 0]
    c_items = [
        ("Proposed", "coupling_energy_frac_test"),
        ("WGI", "coupling_energy_frac_waveguide_invariant_test"),
        ("Modal (c0)", "coupling_energy_frac_modal_striation_test"),
        ("Modal (CTD)", "coupling_energy_frac_modal_striation_profile_test"),
    ]
    labs = []
    ca = []; cb = []
    ca_std = []; cb_std = []
    for lab, key in c_items:
        va, va_std = _get(ra, key)
        vb, vb_std = _get(rb, key)
        if va is None or vb is None:
            continue
        labs.append(lab)
        ca.append(float(va)); cb.append(float(vb))
        ca_std.append(0.0 if va_std is None else float(va_std))
        cb_std.append(0.0 if vb_std is None else float(vb_std))
    if labs:
        xx = np.arange(len(labs), dtype=float)
        ax.bar(xx - w / 2, ca, w, yerr=ca_std, capsize=2.5, label=label_a)
        ax.bar(xx + w / 2, cb, w, yerr=cb_std, capsize=2.5, label=label_b)
        ax.set_xticks(xx)
        ax.set_xticklabels(labs, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Energy fraction")
        ax.set_title("Estimated interaction energy")
        ax.grid(True, axis="y", alpha=0.3)
    else:
        ax.axis("off")
        ax.text(0.1, 0.5, "Coupling metrics not found", fontsize=12)

    # Beta panel
    ax = axs[1, 1]
    bd_a = ra.get("beta_diagnostics", {}) if isinstance(ra.get("beta_diagnostics", {}), dict) else {}
    bd_b = rb.get("beta_diagnostics", {}) if isinstance(rb.get("beta_diagnostics", {}), dict) else {}
    # β estimates can be ill-conditioned when the learned interaction energy is tiny.
    # To keep the manuscript figure interpretable, we only plot *reasonable* β values
    # (typical shallow-water striation slopes are O(1)).
    b_items = [
        (r"$\beta$ (WGI grid)", "beta_waveguide_invariant_best"),
        (r"$\beta$ (interaction)", "beta_interaction_tensor"),
        (r"$\beta$ (modal c0)", "beta_modal_constant_tensor"),
        (r"$\beta$ (modal CTD)", "beta_modal_profile_tensor"),
    ]
    labs = []
    ba = []; bb = []
    ba_std = []; bb_std = []
    def _beta_ok(v: Optional[float]) -> bool:
        if v is None:
            return False
        if not np.isfinite(float(v)):
            return False
        vv = float(v)
        return (vv >= 0.05) and (vv <= 5.0)

    for lab, key in b_items:
        va = _float_or_none(bd_a.get(key))
        vb = _float_or_none(bd_b.get(key))
        if (not _beta_ok(va)) or (not _beta_ok(vb)):
            continue
        labs.append(lab)
        ba.append(float(va)); bb.append(float(vb))
        va_std = _float_or_none(bd_a.get(key + "_std"))
        vb_std = _float_or_none(bd_b.get(key + "_std"))
        ba_std.append(0.0 if va_std is None else float(va_std))
        bb_std.append(0.0 if vb_std is None else float(vb_std))
    if labs:
        xx = np.arange(len(labs), dtype=float)
        ax.bar(xx - w / 2, ba, w, yerr=ba_std, capsize=2.5, label=label_a)
        ax.bar(xx + w / 2, bb, w, yerr=bb_std, capsize=2.5, label=label_b)
        ax.set_xticks(xx)
        ax.set_xticklabels(labs, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(r"$\beta$")
        ax.set_title("Waveguide-invariant diagnostics")
        ax.grid(True, axis="y", alpha=0.3)
    else:
        ax.axis("off")
        ax.text(0.1, 0.5, r"$\beta$ diagnostics not found", fontsize=12)

    fig.suptitle("Two-event summary (blocked cross-validation)", fontsize=14)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
