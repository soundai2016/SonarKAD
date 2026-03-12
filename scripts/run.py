"""Unified command-line entrypoint.
This script provides a single CLI to:
(i) validate the SWellEx-96 dataset paths,
(ii) prepare processed tonal datasets,
(iii) train models and run blocked cross-validation,
(iv) run targeted studies (rank ablation and cross-event transfer), and
(v) generate figures/tables for the manuscript.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

# Ensure local src/ is importable (repo-friendly without installation).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import yaml

from sonarkad.data.validate_swellex96 import validate_swellex96_dataset
from sonarkad.experiments.surrogate_experiment import run_surrogate_experiment
from sonarkad.experiments.swellex96_experiment import (
    _resolve_swellex96_dataset_config,
    prepare_swellex96,
    train_swellex96,
    train_swellex96_cv,
)
from sonarkad.experiments.swellex96_studies import (
    materialize_selected_rank_run,
    run_rank_ablation,
    run_transfer_study,
)
from sonarkad.plots.method_overview import draw_method_overview
from sonarkad.plots.surrogate_benchmark import plot_surrogate_benchmark
from sonarkad.plots.swellex96 import (
    plot_event_report,
    plot_two_event_decomposition,
    plot_two_event_overview,
    plot_two_event_summary,
    write_two_event_diagnostics_csv,
    write_two_event_metrics_csv,
)


def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must parse to a dict")
    return cfg


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_experiments(cfg: Dict[str, Any]) -> None:
    exps = cfg.get("experiments", {})
    if not isinstance(exps, dict):
        print("No experiments found in config.")
        return
    for k, v in exps.items():
        if not isinstance(v, dict):
            continue
        t = v.get("type", "?")
        print(f"- {k} (type={t})")


def _apply_device_override(exp_cfg: Dict[str, Any], *, device: str | None, gpu_flag: bool) -> None:
    if gpu_flag and not device:
        device = "cuda"
    if device:
        exp_cfg.setdefault("training", {})
        exp_cfg["training"]["device"] = str(device)
        if str(device).startswith("cuda") and "amp" not in exp_cfg["training"]:
            exp_cfg["training"]["amp"] = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        choices=[
            "check",
            "method-overview",
            "surrogate-run",
            "surrogate-plot",
            "data-validate",
            "swellex96-check",
            "swellex96-prepare",
            "swellex96-train",
            "swellex96-train-cv",
            "swellex96-rank-ablation",
            "swellex96-select-rank",
            "swellex96-transfer",
            "swellex96-plot",
            "paper-assets",
        ],
        help="Task to execute.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--exp",
        default=None,
        help="Experiment name (required for swellex96-* tasks; optional otherwise)",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Source experiment name (for swellex96-transfer)",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target experiment name (for swellex96-transfer)",
    )
    parser.add_argument(
        "--ranks",
        default=None,
        help="Comma-separated interaction ranks for rank ablation (e.g. 0,1,2,4,8,16).",
    )
    parser.add_argument(
        "--selection-rule",
        default="1se",
        choices=["1se", "best"],
        help="Model-selection rule for interaction rank (default: 1se).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration / rerun when supported by the task.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device override for training/inference: auto|cpu|cuda|cuda:0|mps",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Shortcut for --device cuda (use GPU if available).",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    outputs_root = Path(cfg.get("outputs_root", "outputs"))
    ensure_dir(outputs_root)

    if args.task == "check":
        print("Config loaded.")
        print(f"outputs_root: {outputs_root}")
        print("Experiments:")
        list_experiments(cfg)
        return 0

    # -----------------
    # Data validation
    # -----------------

    if args.task == "data-validate":
        exps = cfg.get("experiments", {})
        if not isinstance(exps, dict) or not exps:
            raise SystemExit("No experiments found in config.")

        any_fail = False
        for name, exp_cfg in exps.items():
            if not isinstance(exp_cfg, dict) or exp_cfg.get("type") != "swellex96":
                continue

            ds = _resolve_swellex96_dataset_config(exp_cfg.get("dataset", {}) or {})

            report = validate_swellex96_dataset(
                sio_path=ds.get("sio_path"),
                range_table_path=ds.get("range_table_path"),
                positions_vla_path=ds.get("positions_vla_path"),
                expected_nc=ds.get("expected_nc", None),
                event=ds.get("event", None),
                range_file_hint=ds.get("range_file_hint", None),
            )

            status = "OK" if report.ok else "FAIL"
            print(f"[{status}] {name}")

            if report.errors:
                any_fail = True
                for k, v in report.errors.items():
                    print(f"  [error] {k}: {v}")
            if report.warnings:
                for k, v in report.warnings.items():
                    print(f"  [warn]  {k}: {v}")

            # Compact info summary
            try:
                sio_i = report.info.get("sio", {})
                rng_i = report.info.get("range_table", {})
                off = report.info.get("inferred_range_time_offset_sec", None)
                print(
                    f"  sio: {Path(str(sio_i.get('path',''))).name} nc={sio_i.get('nc')} np={sio_i.get('np_per_channel')} endian={sio_i.get('endian')}"
                )
                print(
                    f"  range: {Path(str(rng_i.get('path',''))).name} inferred_offset_sec={off}"
                )
            except Exception:
                pass

        if any_fail:
            raise SystemExit(2)
        return 0

    # -----------------
    # Method overview
    # -----------------

    if args.task == "method-overview":
        paper_cfg = cfg.get("paper", {}) if isinstance(cfg.get("paper", {}), dict) else {}
        out_dir = Path(paper_cfg.get("out_dir", outputs_root / "paper"))
        ensure_dir(out_dir)
        fig_name = paper_cfg.get("figures", {}).get("method_overview", "figure_method_overview.png")
        fig_cfg = cfg.get("method_overview", {}) if isinstance(cfg.get("method_overview", {}), dict) else {}
        out_path = out_dir / fig_name
        draw_method_overview(out_path, fig_cfg)
        print(f"[OK] wrote {out_path}")
        return 0

    # -----------------
    # Surrogate (optional)
    # -----------------

    if args.task in {"surrogate-run", "surrogate-plot"}:
        exp_name = args.exp or "surrogate_benchmark"
        exp_cfg = cfg.get("experiments", {}).get(exp_name, None)
        if exp_cfg is None:
            raise SystemExit(f"Experiment not found in config: {exp_name}")

        out_dir = outputs_root / exp_name
        ensure_dir(out_dir)

        if args.task == "surrogate-run":
            run_dir = Path(exp_cfg.get("outputs", {}).get("run_dir", out_dir / "run"))
            ensure_dir(run_dir)
            run_surrogate_experiment(exp_cfg, run_dir)
            print(f"[OK] surrogate results in {run_dir}")
            return 0

        if args.task == "surrogate-plot":
            run_dir = Path(exp_cfg.get("outputs", {}).get("run_dir", out_dir / "run"))
            results_npz = run_dir / "results_aggregate.npz"
            if not results_npz.exists():
                raise SystemExit(f"Missing {results_npz}. Run surrogate-run first.")
            paper_cfg = cfg.get("paper", {}) if isinstance(cfg.get("paper", {}), dict) else {}
            out_dir2 = Path(paper_cfg.get("out_dir", outputs_root / "paper"))
            ensure_dir(out_dir2)
            out_path = out_dir2 / "figure_surrogate_benchmark.png"
            plot_surrogate_benchmark(results_npz, out_path)
            print(f"[OK] wrote {out_path}")
            return 0

    # -----------------
    # SWellEx-96 tasks
    # -----------------

    if args.task.startswith("swellex96-"):
        if args.task == "swellex96-transfer":
            # This task uses --source and --target.
            if not args.source or not args.target:
                raise SystemExit("swellex96-transfer requires --source and --target experiment names")
            exps = cfg.get("experiments", {})
            if args.source not in exps or args.target not in exps:
                raise SystemExit(f"Unknown experiments: source={args.source} target={args.target}")

            src_cfg = exps[args.source]
            tgt_cfg = exps[args.target]
            if src_cfg.get("type") != "swellex96" or tgt_cfg.get("type") != "swellex96":
                raise SystemExit("Both source and target must be type=swellex96")

            _apply_device_override(src_cfg, device=args.device, gpu_flag=args.gpu)
            _apply_device_override(tgt_cfg, device=args.device, gpu_flag=args.gpu)

            out_dir_tr = outputs_root / f"transfer_{args.source}_to_{args.target}"
            ensure_dir(out_dir_tr)

            summary = run_transfer_study(
                source_exp_name=args.source,
                source_cfg=src_cfg,
                target_exp_name=args.target,
                target_cfg=tgt_cfg,
                out_dir=out_dir_tr,
                force=args.force,
            )
            print(f"[OK] transfer study outputs in {out_dir_tr}")
            if isinstance(summary, dict):
                for k in ("rmse_test_transfer", "ev_test_transfer"):
                    if k in summary:
                        print(f"      {k}={summary.get(k)}")
            return 0

        # Remaining swellex96-* tasks use --exp.
        if not args.exp:
            raise SystemExit("--exp is required for swellex96-* tasks")

        exps = cfg.get("experiments", {})
        if args.exp not in exps:
            raise SystemExit(f"Experiment not found in config: {args.exp}")
        exp_cfg = exps[args.exp]

        # Device override from CLI
        _apply_device_override(exp_cfg, device=args.device, gpu_flag=args.gpu)

        if exp_cfg.get("type", "") != "swellex96":
            raise SystemExit(f"Experiment {args.exp} is not type=swellex96")

        exp_outputs = exp_cfg.get("outputs", {}) if isinstance(exp_cfg.get("outputs", {}), dict) else {}
        processed_dir = Path(exp_outputs.get("processed_dir", outputs_root / args.exp / "processed"))
        run_dir = Path(exp_outputs.get("run_dir", outputs_root / args.exp / "run"))
        plot_dir = Path(exp_outputs.get("plot_dir", run_dir / "plots"))
        ensure_dir(processed_dir)
        ensure_dir(run_dir)

        if args.task == "swellex96-check":
            ds = exp_cfg.get("dataset", {})
            print(f"Experiment: {args.exp}")
            print(f"  data_root: {ds.get('data_root')}")
            print(f"  event: {ds.get('event')}  array: {ds.get('array')}  tone_set: {ds.get('tone_set')}")
            print(f"  processed_dir: {processed_dir}")
            print(f"  run_dir: {run_dir}")
            return 0

        if args.task == "swellex96-prepare":
            out_npz = prepare_swellex96(exp_cfg, processed_dir, force=args.force)
            if out_npz.exists() and not args.force:
                print(f"[OK] prepared dataset exists: {out_npz}")
            else:
                print(f"[OK] wrote {out_npz}")
            return 0

        if args.task == "swellex96-train":
            results = train_swellex96(exp_cfg, run_dir, force=args.force)
            print(
                f"[OK] trained. Proposed RMSE(test)={results.get('rmse_test_sonarkad', float('nan')):.3f} dB, "
                f"EV={results.get('ev_test_sonarkad', float('nan')):.3f}."
            )
            print(f"      outputs in {run_dir}")
            return 0

        if args.task == "swellex96-train-cv":
            results = train_swellex96_cv(exp_cfg, run_dir, force=args.force)
            print(
                f"[OK] block-CV finished. Proposed RMSE(test)={results.get('rmse_test_sonarkad', float('nan')):.3f} dB "
                f"(std={results.get('rmse_test_sonarkad_std', float('nan')):.3f}), "
                f"EV={results.get('ev_test_sonarkad', float('nan')):.3f} "
                f"(std={results.get('ev_test_sonarkad_std', float('nan')):.3f})."
            )
            print(f"      wrote {run_dir / 'results_cv.json'}")
            return 0

        if args.task == "swellex96-rank-ablation":
            ranks = None
            if args.ranks:
                try:
                    ranks = [int(x) for x in str(args.ranks).split(",") if str(x).strip()]
                except Exception:
                    ranks = None

            out_dir_ab = run_dir / "rank_ablation"
            ensure_dir(out_dir_ab)
            summary = run_rank_ablation(
                exp_cfg,
                out_dir_ab,
                ranks=ranks,
                selection_rule=args.selection_rule,
                force=args.force,
            )
            print(f"[OK] rank ablation written to {out_dir_ab}")
            if isinstance(summary, dict):
                sel = summary.get("selected_rank", None)
                if sel is not None:
                    print(f"      selected_rank={sel}")
            return 0

        if args.task == "swellex96-select-rank":
            ranks = None
            if args.ranks:
                try:
                    ranks = [int(x) for x in str(args.ranks).split(",") if str(x).strip()]
                except Exception:
                    ranks = None

            sel_root = run_dir / "selected_model"
            ensure_dir(sel_root)
            manifest = materialize_selected_rank_run(
                args.exp,
                exp_cfg,
                sel_root,
                ranks=ranks,
                selection_rule=args.selection_rule,
                force=args.force,
            )
            print(f"[OK] selected-rank artifacts written to {sel_root}")
            print(f"      selected_rank={manifest.get('selected_rank')}")
            print(f"      results_cv={manifest.get('results_cv')}")
            return 0

        if args.task == "swellex96-plot":
            comp_npz = run_dir / "components.pt"
            results_json = (run_dir / "results_cv.json") if (run_dir / "results_cv.json").exists() else (run_dir / "results.json")
            processed_npz = Path(exp_cfg.get("dataset", {}).get("processed_npz", ""))

            if not processed_npz.exists():
                raise SystemExit(
                    f"Processed dataset not found: {processed_npz}. "
                    f"Run 'swellex96-prepare' first (or fix dataset.processed_npz in config)."
                )
            if not comp_npz.exists():
                raise SystemExit(f"Missing components file: {comp_npz}. Run 'swellex96-train' first.")
            if not results_json.exists():
                raise SystemExit(f"Missing results file: {results_json}. Run train / train-cv first.")

            ensure_dir(plot_dir)
            plot_event_report(
                processed_npz=processed_npz,
                components_npz=comp_npz,
                results_json=results_json,
                out_dir=plot_dir,
                prefix=args.exp,
                title=f"SWellEx-96 {args.exp}",
            )
            print(f"[OK] plots in {plot_dir}")
            return 0

    # --------------------------
    # Paper assets (two-event)
    # --------------------------

    if args.task == "paper-assets":
        paper_cfg = cfg.get("paper", {}) if isinstance(cfg.get("paper", {}), dict) else {}
        out_dir = Path(paper_cfg.get("out_dir", outputs_root / "paper"))
        ensure_dir(out_dir)

        fig_cfg = paper_cfg.get("figures", {}) if isinstance(paper_cfg.get("figures", {}), dict) else {}
        tab_cfg = paper_cfg.get("tables", {}) if isinstance(paper_cfg.get("tables", {}), dict) else {}

        # Which two experiments to include
        events = paper_cfg.get("events", ["swellex96_s5_vla", "swellex96_s59_vla"])
        if not isinstance(events, list) or len(events) != 2:
            raise SystemExit("paper.events must be a list of two experiment names")
        exp_a, exp_b = events[0], events[1]

        exps = cfg.get("experiments", {})
        if exp_a not in exps or exp_b not in exps:
            raise SystemExit(f"Config must define both experiments: {exp_a}, {exp_b}")

        a_cfg = exps[exp_a]
        b_cfg = exps[exp_b]
        if a_cfg.get("type") != "swellex96" or b_cfg.get("type") != "swellex96":
            raise SystemExit("paper.events must refer to swellex96 experiments")

        def _paths(exp_cfg: Dict[str, Any], exp_name: str) -> Dict[str, Path]:
            """Resolve standard artifact locations for one experiment."""
            exp_out = exp_cfg.get("outputs", {}) if isinstance(exp_cfg.get("outputs", {}), dict) else {}
            processed_dir = Path(exp_out.get("processed_dir", outputs_root / exp_name / "processed"))
            run_dir = Path(exp_out.get("run_dir", outputs_root / exp_name / "run"))
            return {
                "processed_dir": processed_dir,
                "run_dir": run_dir,
                "processed_npz": Path(exp_cfg.get("dataset", {}).get("processed_npz", "")),
                "components_npz": run_dir / "components.pt",
                "results_cv": run_dir / "results_cv.json",
                "results": run_dir / "results.json",
            }

        pa = _paths(a_cfg, exp_a)
        pb = _paths(b_cfg, exp_b)

        # Labels for plots
        label_a = paper_cfg.get("labels", {}).get("event_a", "Event A") if isinstance(paper_cfg.get("labels", {}), dict) else "Event A"
        label_b = paper_cfg.get("labels", {}).get("event_b", "Event B") if isinstance(paper_cfg.get("labels", {}), dict) else "Event B"
        tag_a = paper_cfg.get("tags", {}).get("event_a", "A") if isinstance(paper_cfg.get("tags", {}), dict) else "A"
        tag_b = paper_cfg.get("tags", {}).get("event_b", "B") if isinstance(paper_cfg.get("tags", {}), dict) else "B"

        # Optional: use automatically selected interaction ranks for the paper's
        # main decomposition + summary assets instead of the fixed config ranks.
        ms_cfg = paper_cfg.get("model_selection", {}) if isinstance(paper_cfg.get("model_selection", {}), dict) else {}
        use_selected_rank = bool(ms_cfg.get("enabled", False))

        def _selection_opts(exp_name: str) -> Dict[str, Any]:
            ev_cfg = ms_cfg.get("events", {}) if isinstance(ms_cfg.get("events", {}), dict) else {}
            one = ev_cfg.get(exp_name, {}) if isinstance(ev_cfg.get(exp_name, {}), dict) else {}
            ranks = one.get("ranks", ms_cfg.get("default_ranks", None))
            if isinstance(ranks, list):
                try:
                    ranks = [int(x) for x in ranks]
                except Exception:
                    ranks = None
            else:
                ranks = None
            return {
                "ranks": ranks,
                "selection_rule": str(one.get("selection_rule", ms_cfg.get("selection_rule", "1se"))),
            }

        if use_selected_rank:
            sel_root = out_dir / "selected_models"
            ensure_dir(sel_root)

            a_sel = materialize_selected_rank_run(
                exp_a,
                a_cfg,
                sel_root / exp_a,
                force=args.force,
                **_selection_opts(exp_a),
            )
            b_sel = materialize_selected_rank_run(
                exp_b,
                b_cfg,
                sel_root / exp_b,
                force=args.force,
                **_selection_opts(exp_b),
            )

            pa["run_dir"] = Path(str(a_sel["run_dir"]))
            pa["components_npz"] = Path(str(a_sel["components"]))
            pa["results_cv"] = Path(str(a_sel["results_cv"]))
            pa["results"] = Path(str(a_sel["results"]))
            pa["selected_rank"] = int(a_sel["selected_rank"])

            pb["run_dir"] = Path(str(b_sel["run_dir"]))
            pb["components_npz"] = Path(str(b_sel["components"]))
            pb["results_cv"] = Path(str(b_sel["results_cv"]))
            pb["results"] = Path(str(b_sel["results"]))
            pb["selected_rank"] = int(b_sel["selected_rank"])

            selected_manifest = {
                exp_a: {"selected_rank": pa["selected_rank"], "results_cv": str(pa["results_cv"]), "components": str(pa["components_npz"])},
                exp_b: {"selected_rank": pb["selected_rank"], "results_cv": str(pb["results_cv"]), "components": str(pb["components_npz"])},
            }
            (out_dir / "selected_models_summary.json").write_text(json.dumps(selected_manifest, indent=2), encoding="utf-8")

        # ------------------------------------------------------------
        # Decide which assets to (re)build. We intentionally do *not*
        # force heavy training runs unless an asset that depends on
        # those artifacts is missing.
        # ------------------------------------------------------------

        method_name = fig_cfg.get("method_overview", "figure_method_overview.png")
        overview_name = fig_cfg.get("data_overview", "figure_swellex96_data_overview.png")
        decomp_name = fig_cfg.get("decomposition", "figure_swellex96_decomposition.png")
        summary_name = fig_cfg.get("summary", "figure_swellex96_summary.png")
        metrics_name = tab_cfg.get("metrics_csv", "table_metrics_two_events.csv")
        diagnostics_name = tab_cfg.get("diagnostics_csv", "table_diagnostics_two_events.csv")

        method_path = out_dir / method_name
        overview_path = out_dir / overview_name
        decomp_path = out_dir / decomp_name
        summary_path = out_dir / summary_name
        metrics_csv_path = out_dir / metrics_name
        diagnostics_csv_path = out_dir / diagnostics_name

        # NOTE: --force here means "force re-render paper assets".
        # It intentionally does *not* force expensive recomputation of
        # datasets / training artifacts (run swellex96-train / swellex96-cv
        # with --force for that).
        need_method = (not method_path.exists()) or args.force
        need_overview = (not overview_path.exists()) or args.force
        need_decomp = (not decomp_path.exists()) or args.force
        need_summary = (not summary_path.exists()) or args.force
        need_metrics = (not metrics_csv_path.exists()) or args.force
        need_diagnostics = (not diagnostics_csv_path.exists()) or args.force

        if not any([need_method, need_overview, need_decomp, need_summary, need_metrics, need_diagnostics]):
            print(f"[OK] paper assets already exist in {out_dir} (nothing to do).")
            return 0

        # Ensure required intermediate artifacts only when needed.
        if need_overview:
            ensure_dir(pa["processed_dir"])
            ensure_dir(pb["processed_dir"])
            if (not pa["processed_npz"].exists()) or args.force:
                pa["processed_npz"] = prepare_swellex96(a_cfg, pa["processed_dir"], force=False)
            if (not pb["processed_npz"].exists()) or args.force:
                pb["processed_npz"] = prepare_swellex96(b_cfg, pb["processed_dir"], force=False)

        if need_decomp:
            ensure_dir(pa["run_dir"])
            ensure_dir(pb["run_dir"])
            if (not pa["components_npz"].exists()) or args.force:
                train_swellex96(a_cfg, pa["run_dir"], force=False)
            if (not pb["components_npz"].exists()) or args.force:
                train_swellex96(b_cfg, pb["run_dir"], force=False)

        if need_summary or need_metrics or need_diagnostics:
            ensure_dir(pa["run_dir"])
            ensure_dir(pb["run_dir"])
            if (not pa["results_cv"].exists()) or args.force:
                train_swellex96_cv(a_cfg, pa["run_dir"], force=False)
            if (not pb["results_cv"].exists()) or args.force:
                train_swellex96_cv(b_cfg, pb["run_dir"], force=False)

        # ------------------------------------------------------------
        # Build assets
        # ------------------------------------------------------------

        # 1) Method overview figure
        method_cfg = cfg.get("method_overview", {}) if isinstance(cfg.get("method_overview", {}), dict) else {}
        if need_method:
            draw_method_overview(method_path, method_cfg)

        # 2) Data overview (two events)
        if need_overview:
            plot_two_event_overview(
                processed_a=pa["processed_npz"],
                label_a=label_a,
                processed_b=pb["processed_npz"],
                label_b=label_b,
                out_path=overview_path,
            )

        # 3) Decomposition + invariant overlay (two events)
        if need_decomp:
            plot_two_event_decomposition(
                components_a=pa["components_npz"],
                label_a=label_a,
                components_b=pb["components_npz"],
                label_b=label_b,
                out_path=decomp_path,
            )

        # 4) Summary (metrics + diagnostics)
        if need_summary:
            plot_two_event_summary(
                results_a=pa["results_cv"],
                label_a=label_a,
                results_b=pb["results_cv"],
                label_b=label_b,
                out_path=summary_path,
            )

        # Tables (CSV)
        if need_metrics or need_diagnostics:
            with pa["results_cv"].open("r", encoding="utf-8") as f:
                ra = json.load(f)
            with pb["results_cv"].open("r", encoding="utf-8") as f:
                rb = json.load(f)

            if need_metrics:
                write_two_event_metrics_csv(
                    results_a=ra,
                    event_a=tag_a,
                    results_b=rb,
                    event_b=tag_b,
                    out_path=metrics_csv_path,
                )

            if need_diagnostics:
                write_two_event_diagnostics_csv(
                    results_a=ra,
                    event_a=tag_a,
                    results_b=rb,
                    event_b=tag_b,
                    out_path=diagnostics_csv_path,
                )

        # -----------------
        # Optional extra studies
        # -----------------
        extras = cfg.get("paper_extras", {}) if isinstance(cfg.get("paper_extras", {}), dict) else {}

        # Rank ablation (can be one event or both events)
        ab_cfg = extras.get("rank_ablation", {}) if isinstance(extras.get("rank_ablation", {}), dict) else {}
        ab_events = ab_cfg.get("events", None)
        if not isinstance(ab_events, list) or not ab_events:
            ab_events = [ab_cfg.get("exp", exp_a)]
        ab_ranks = ab_cfg.get("ranks", None)
        ab_rule = str(ab_cfg.get("selection_rule", "1se"))
        for ab_exp in ab_events:
            if ab_exp in exps:
                ab_out = out_dir / f"rank_ablation_{ab_exp}"
                ensure_dir(ab_out)
                run_rank_ablation(
                    exps[ab_exp],
                    ab_out,
                    ranks=ab_ranks,
                    selection_rule=ab_rule,
                    write_into_paper_dir=True,
                    force=args.force,
                )

        # Transfer study (defaults exp_a -> exp_b)
        tr_cfg = extras.get("transfer", {}) if isinstance(extras.get("transfer", {}), dict) else {}
        src = tr_cfg.get("source", exp_a)
        tgt = tr_cfg.get("target", exp_b)
        if src in exps and tgt in exps:
            tr_out = out_dir / "transfer_study"
            ensure_dir(tr_out)
            run_transfer_study(
                source_exp_name=src,
                source_cfg=exps[src],
                target_exp_name=tgt,
                target_cfg=exps[tgt],
                out_dir=tr_out,
                write_into_paper_dir=True,
                force=args.force,
            )

        print(f"[OK] paper assets written to {out_dir}")
        if use_selected_rank:
            print(f"      selected ranks: {exp_a} -> K={pa.get('selected_rank')}, {exp_b} -> K={pb.get('selected_rank')}")
        print("      (Copy figures from outputs/results -> paper/figures before compiling the manuscript.)")
        return 0

    raise SystemExit("Unknown task")


if __name__ == "__main__":
    raise SystemExit(main())
