"""Higher-level studies built on top of the SWellEx-96 experiment pipeline.

This module adds two paper-oriented studies that reviewers commonly request:

1) **Rank ablation / model selection** over the nonseparable interaction rank K.
2) **Cross-event transfer** (e.g., reuse the learned source spectrum term φ_f
   from a clean event when fitting a more challenging event).

The functions here are thin orchestration layers; the heavy lifting is done by
`train_swellex96()` and `train_swellex96_cv()`.
"""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ..utils.paths import ensure_dir
from ..utils.plotting import clean_axes, set_plot_style
from ..utils.torch_compat import torch_load_compat
from .swellex96_experiment import prepare_swellex96, train_swellex96, train_swellex96_cv


# -----------------------------------------------------------------------------
# Rank ablation
# -----------------------------------------------------------------------------


def _default_ranks() -> List[int]:
    return [0, 1, 2, 4, 8, 16]


def clone_cfg_with_rank(exp_cfg: Dict[str, object], rank: int) -> Dict[str, object]:
    """Return a deep-copied experiment config with ``model.interaction_rank`` set."""

    cfg_k = copy.deepcopy(exp_cfg)
    cfg_k.setdefault("model", {})
    if not isinstance(cfg_k["model"], dict):
        cfg_k["model"] = {}
    cfg_k["model"]["interaction_rank"] = int(rank)
    return cfg_k


def _select_rank_1se(ranks: Sequence[int], rmse_mean: Sequence[float], rmse_std: Sequence[float]) -> int:
    """1-SE rule: choose smallest rank within 1 std of the best mean."""
    arr_m = np.asarray(rmse_mean, dtype=np.float64)
    arr_s = np.asarray(rmse_std, dtype=np.float64)
    best_idx = int(np.nanargmin(arr_m))
    thresh = float(arr_m[best_idx] + (arr_s[best_idx] if np.isfinite(arr_s[best_idx]) else 0.0))
    # Smallest K whose mean <= thresh.
    for k, m in sorted(zip(list(ranks), list(arr_m)), key=lambda x: x[0]):
        if np.isfinite(m) and float(m) <= thresh:
            return int(k)
    return int(ranks[best_idx])


def _write_rank_ablation_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    cols = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_rank_ablation(
    *,
    ranks: Sequence[int],
    rmse_mean: Sequence[float],
    rmse_std: Sequence[float],
    ev_mean: Optional[Sequence[float]],
    ev_std: Optional[Sequence[float]],
    out_path: Path,
    title: str,
) -> None:
    set_plot_style()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ks = np.asarray(list(ranks), dtype=np.int64)
    rm = np.asarray(list(rmse_mean), dtype=np.float64)
    rs = np.asarray(list(rmse_std), dtype=np.float64)

    fig = plt.figure(figsize=(7.4, 3.4))
    ax = fig.add_subplot(1, 2, 1)
    ax.errorbar(ks, rm, yerr=rs, marker="o", linestyle="-")
    ax.set_xlabel("Interaction rank K")
    ax.set_ylabel("RMSE (dB)")
    ax.set_title("Blocked-CV RMSE")
    clean_axes(ax)

    ax2 = fig.add_subplot(1, 2, 2)
    if ev_mean is not None:
        em = np.asarray(list(ev_mean), dtype=np.float64)
        es = np.asarray(list(ev_std), dtype=np.float64) if ev_std is not None else None
        ax2.errorbar(ks, em, yerr=es, marker="o", linestyle="-")
        ax2.set_ylabel("Explained variance")
        ax2.set_title("Blocked-CV EV")
    else:
        ax2.plot(ks, rm, marker="o", linestyle="-")
        ax2.set_ylabel("RMSE (dB)")
        ax2.set_title("(EV unavailable)")
    ax2.set_xlabel("Interaction rank K")
    clean_axes(ax2)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_rank_ablation(
    exp_cfg: Dict[str, object],
    out_dir: str | Path,
    *,
    ranks: Optional[Sequence[int]] = None,
    selection_rule: str = "1se",
    write_into_paper_dir: bool = False,
    force: bool = False,
) -> Dict[str, object]:
    """Run a blocked-CV rank ablation for a single SWellEx-96 experiment.

    Parameters
    ----------
    exp_cfg:
        A single experiment config dict (type=swellex96).
    out_dir:
        Directory where ablation runs and summary assets are written.
    ranks:
        Iterable of interaction ranks K to evaluate. If None, defaults to
        [0,1,2,4,8,16].
    selection_rule:
        '1se' or 'best'.

    Returns
    -------
    summary dict (also written as JSON).
    """

    out_dir = ensure_dir(out_dir)

    # Idempotency: if a completed summary already exists, reuse it.
    summary_path = out_dir / "rank_ablation_summary.json"
    if summary_path.exists() and (not force):
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            # Fall through to recompute.
            pass
    ranks_list = list(_default_ranks() if ranks is None else [int(x) for x in ranks])

    # Ensure dataset is prepared once.
    processed_dir = out_dir / "processed"
    ensure_dir(processed_dir)
    try:
        prepare_swellex96(exp_cfg, processed_dir)
    except Exception:
        # Prepare is idempotent; if it fails we still want the error.
        raise

    rows: List[Dict[str, object]] = []

    for k in ranks_list:
        cfg_k = clone_cfg_with_rank(exp_cfg, int(k))

        run_dir_k = out_dir / f"K{k:02d}"
        ensure_dir(run_dir_k)

        # Blocked CV for this rank
        res = train_swellex96_cv(cfg_k, run_dir_k, force=force)

        rows.append(
            {
                "K": int(k),
                "rmse_mean": float(res.get("rmse_test_sonarkad", np.nan)),
                "rmse_std": float(res.get("rmse_test_sonarkad_std", np.nan)),
                "ev_mean": float(res.get("ev_test_sonarkad", np.nan)),
                "ev_std": float(res.get("ev_test_sonarkad_std", np.nan)),
                "coupling_energy_frac_mean": float(res.get("coupling_energy_frac_test", np.nan)),
                "coupling_energy_frac_std": float(res.get("coupling_energy_frac_test_std", np.nan)),
            }
        )

    # Select rank
    rmse_mean = [float(r["rmse_mean"]) for r in rows]
    rmse_std = [float(r["rmse_std"]) for r in rows]

    if str(selection_rule).strip().lower() == "best":
        sel = int(rows[int(np.nanargmin(np.asarray(rmse_mean)))]["K"])
    else:
        sel = _select_rank_1se([int(r["K"]) for r in rows], rmse_mean, rmse_std)

    # Write CSV
    csv_path = out_dir / "rank_ablation.csv"
    _write_rank_ablation_csv(rows, csv_path)

    # Plot
    fig_path = out_dir / "figure_rank_ablation.png"
    _plot_rank_ablation(
        ranks=[int(r["K"]) for r in rows],
        rmse_mean=rmse_mean,
        rmse_std=rmse_std,
        ev_mean=[float(r["ev_mean"]) for r in rows],
        ev_std=[float(r["ev_std"]) for r in rows],
        out_path=fig_path,
        title=str(exp_cfg.get("name", "SWellEx-96")) + " (rank ablation)",
    )

    summary: Dict[str, object] = {
        "exp_name": str(exp_cfg.get("name", "swellex96")),
        "ranks": ranks_list,
        "selected_rank": int(sel),
        "selection_rule": str(selection_rule),
        "csv": str(csv_path),
        "figure": str(fig_path),
    }

    (out_dir / "rank_ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Optional: convenience copies when writing directly into paper output.
    # (No-op by default; out_dir is usually already within outputs/paper/*.)
    _ = write_into_paper_dir

    return summary


def materialize_selected_rank_run(
    exp_name: str,
    exp_cfg: Dict[str, object],
    out_dir: str | Path,
    *,
    ranks: Optional[Sequence[int]] = None,
    selection_rule: str = "1se",
    force: bool = False,
) -> Dict[str, object]:
    """Run rank selection, then materialize the selected model artifacts.

    This helper is meant for paper-generation workflows: it guarantees that the
    decomposition figure (single fitted model) and the summary table/figure
    (blocked-CV metrics) are produced from the same automatically selected
    interaction rank.
    """

    out_dir = ensure_dir(out_dir)
    manifest_path = out_dir / "selected_model_manifest.json"
    if manifest_path.exists() and (not force):
        try:
            obj = json.loads(manifest_path.read_text(encoding="utf-8"))
            needed = [
                Path(str(obj.get("rank_ablation_summary", ""))),
                Path(str(obj.get("run_dir", ""))) / "components.pt",
                Path(str(obj.get("run_dir", ""))) / "results_cv.json",
            ]
            if all(p.exists() for p in needed if str(p)):
                return obj
        except Exception:
            pass

    rank_dir = ensure_dir(Path(out_dir) / "rank_ablation")
    summary = run_rank_ablation(
        exp_cfg,
        rank_dir,
        ranks=ranks,
        selection_rule=selection_rule,
        force=force,
    )
    selected_rank = int(summary.get("selected_rank", 0))

    cfg_sel = clone_cfg_with_rank(exp_cfg, selected_rank)
    run_dir = ensure_dir(Path(out_dir) / f"selected_K{selected_rank:02d}" / "run")

    # Single fitted model for decomposition components.
    train_swellex96(cfg_sel, run_dir, force=force)
    # Blocked-CV summary for the paper's main metrics.
    train_swellex96_cv(cfg_sel, run_dir, force=force)

    manifest: Dict[str, object] = {
        "exp_name": str(exp_name),
        "selected_rank": int(selected_rank),
        "selection_rule": str(selection_rule),
        "ranks": list(_default_ranks() if ranks is None else [int(x) for x in ranks]),
        "rank_ablation_summary": str(rank_dir / "rank_ablation_summary.json"),
        "run_dir": str(run_dir),
        "components": str(run_dir / "components.pt"),
        "results": str(run_dir / "results.json"),
        "results_cv": str(run_dir / "results_cv.json"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# -----------------------------------------------------------------------------
# Cross-event transfer study
# -----------------------------------------------------------------------------


def _load_json(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _plot_phi_f_comparison(
    *,
    f_hz: np.ndarray,
    phi_f_source: np.ndarray,
    phi_f_target_scratch: np.ndarray,
    phi_f_target_transfer: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    set_plot_style()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.0, 3.6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(f_hz, phi_f_source, label="Source (trained)")
    ax.plot(f_hz, phi_f_target_scratch, label="Target (scratch)")
    ax.plot(f_hz, phi_f_target_transfer, label="Target (transfer/frozen)")
    ax.set_xlabel("Frequency (Hz)")
    # Use a single backslash for Matplotlib mathtext.
    ax.set_ylabel(r"$\phi_f(f)$ (dB)")
    ax.set_title(title)
    ax.legend(loc="best")
    clean_axes(ax)
    try:
        fig.tight_layout()
    except Exception:
        # tight_layout() can fail for some font/mathtext backends; the figure is still usable.
        pass
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_transfer_study(
    *,
    source_exp_name: str,
    source_cfg: Dict[str, object],
    target_exp_name: str,
    target_cfg: Dict[str, object],
    out_dir: str | Path,
    freeze_parts: Optional[Sequence[str]] = None,
    load_parts: Optional[Sequence[str]] = None,
    write_into_paper_dir: bool = False,
    force: bool = False,
) -> Dict[str, object]:
    """Cross-event transfer study.

    Default: transfer φ_f from `source` to `target` (copy weights and freeze).

    Outputs
    -------
    out_dir/
      source/        (trained source model + components)
      target_scratch/
      target_transfer/
      transfer_summary.json
      figure_transfer_phi_f.png
      table_transfer.csv

    Notes
    -----
    This study is intentionally lightweight (single train/test split). For
    publication-grade statistics you can re-run with blocked-CV on top.
    """

    out_dir = ensure_dir(out_dir)

    # Idempotency: if a completed summary already exists, reuse it.
    summary_path = Path(out_dir) / "transfer_summary.json"
    if summary_path.exists() and (not force):
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            # Fall through to recompute.
            pass

    freeze_parts = list(freeze_parts) if freeze_parts is not None else ["phi_f"]
    load_parts = list(load_parts) if load_parts is not None else ["phi_f"]

    # ------------------------------------------------------------------
    # 1) Train source model (single split)
    # ------------------------------------------------------------------
    src_dir = ensure_dir(Path(out_dir) / "source")
    src_processed = ensure_dir(src_dir / "processed")
    src_run = ensure_dir(src_dir / "run")

    # Prepare + train (idempotent)
    prepare_swellex96(source_cfg, src_processed, force=force)
    src_results_path = src_run / "results.json"
    if not src_results_path.exists():
        train_swellex96(source_cfg, src_run, force=force)

    src_bundle = src_run / "sonarkad_model.pt"
    if not src_bundle.exists():
        raise FileNotFoundError(f"Missing source model bundle: {src_bundle}")

    # ------------------------------------------------------------------
    # 2) Train target from scratch
    # ------------------------------------------------------------------
    tgt_dir = ensure_dir(Path(out_dir) / "target_scratch")
    tgt_processed = ensure_dir(tgt_dir / "processed")
    tgt_run = ensure_dir(tgt_dir / "run")

    prepare_swellex96(target_cfg, tgt_processed, force=force)
    tgt_results_path = tgt_run / "results.json"
    if not tgt_results_path.exists():
        train_swellex96(target_cfg, tgt_run, force=force)

    # ------------------------------------------------------------------
    # 3) Train target with transfer init + freezing
    # ------------------------------------------------------------------
    tr_dir = ensure_dir(Path(out_dir) / "target_transfer")
    tr_processed = ensure_dir(tr_dir / "processed")
    tr_run = ensure_dir(tr_dir / "run")

    prepare_swellex96(target_cfg, tr_processed, force=force)

    transfer_cfg = {
        "bundle_path": str(src_bundle),
        "load_parts": list(load_parts),
        "freeze_parts": list(freeze_parts),
        # Keep φ_f unchanged by interaction gauge-fixing.
        "disable_gauge_fix_interaction": True,
    }

    tgt_cfg_transfer = copy.deepcopy(target_cfg)
    tgt_cfg_transfer.setdefault("training", {})
    if not isinstance(tgt_cfg_transfer["training"], dict):
        tgt_cfg_transfer["training"] = {}
    tgt_cfg_transfer["training"]["transfer"] = transfer_cfg

    tr_results_path = tr_run / "results.json"
    if not tr_results_path.exists():
        train_swellex96(tgt_cfg_transfer, tr_run, force=force)

    # ------------------------------------------------------------------
    # Collect and summarize
    # ------------------------------------------------------------------
    src_res = _load_json(src_results_path)
    tgt_res = _load_json(tgt_results_path)
    tr_res = _load_json(tr_results_path)

    summary = {
        "source": source_exp_name,
        "target": target_exp_name,
        "freeze_parts": list(freeze_parts),
        "load_parts": list(load_parts),
        "paths": {
            "source_bundle": str(src_bundle),
            "target_scratch_results": str(tgt_results_path),
            "target_transfer_results": str(tr_results_path),
        },
        "rmse_test_scratch": tgt_res.get("rmse_test_sonarkad"),
        "ev_test_scratch": tgt_res.get("ev_test_sonarkad"),
        "rmse_test_transfer": tr_res.get("rmse_test_sonarkad"),
        "ev_test_transfer": tr_res.get("ev_test_sonarkad"),
    }

    # ------------------------------------------------------------------
    # Plot φ_f comparison
    # ------------------------------------------------------------------
    src_comp = torch_load_compat(src_run / "components.pt", map_location="cpu")
    tgt_comp = torch_load_compat(tgt_run / "components.pt", map_location="cpu")
    tr_comp = torch_load_compat(tr_run / "components.pt", map_location="cpu")

    f_grid = np.asarray(src_comp.get("grid_f_hz"), dtype=np.float64)
    phi_f_src = np.asarray(src_comp.get("phi_f"), dtype=np.float64)
    phi_f_tgt = np.asarray(tgt_comp.get("phi_f"), dtype=np.float64)
    phi_f_tr = np.asarray(tr_comp.get("phi_f"), dtype=np.float64)

    fig_path = Path(out_dir) / "figure_transfer_phi_f.png"
    try:
        _plot_phi_f_comparison(
            f_hz=f_grid,
            phi_f_source=phi_f_src,
            phi_f_target_scratch=phi_f_tgt,
            phi_f_target_transfer=phi_f_tr,
            out_path=fig_path,
            title=f"Transfer study: {source_exp_name} -> {target_exp_name}",
        )
    except Exception as e:
        # Plotting should never invalidate the computed results.
        print(f"[WARN] transfer plot failed: {e}")

    # CSV summary table (submission-friendly, avoids auto-generated LaTeX).
    csv_path = Path(out_dir) / "table_transfer.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "rmse_test", "ev_test"]) 
        w.writerow(["scratch", summary.get("rmse_test_scratch", ""), summary.get("ev_test_scratch", "")])
        w.writerow([
            "transfer_frozen(" + ",".join(freeze_parts) + ")",
            summary.get("rmse_test_transfer", ""),
            summary.get("ev_test_transfer", ""),
        ])

    summary["figure_phi_f"] = str(fig_path)
    summary["csv_table"] = str(csv_path)

    # Persist full summary (including figure/table paths).
    (Path(out_dir) / "transfer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Optional: convenience copies when writing directly into paper output.
    _ = write_into_paper_dir

    return summary
