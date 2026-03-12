"""Surrogate experiment runner (Fig.2-style).

This module replaces the original standalone script `run_surrogate_experiment.py` by
providing a callable function that can be invoked from a unified CLI.

Key additions over the original
------------------------------
- Supports the extended SonarKAD model with optional low-rank interaction ψ(r,f)
  (set ``model.interaction_rank > 0`` in the config).
- Keeps deterministic, full-batch training for transparency (good for peer review).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Optional progress bars (training + inference)
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

from ..models import (
    BSplineLayerConfig,
    AbsorptionTermConfig,
    SonarKAD,
    SonarKADConfig,
    SmallMLP,
    count_parameters,
    bspline_layer_config_from_dict,
)
from ..surrogate import AcousticSurrogate, AcousticSurrogateConfig
from ..utils.paths import ensure_dir
from ..utils.seed import set_global_seed


def _rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(nn.functional.mse_loss(pred, target)).item())


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b)))
    if denom == 0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def _epochs_to_within(loss: np.ndarray, frac: float = 0.05) -> int:
    loss = np.asarray(loss, dtype=np.float64).reshape(-1)
    final = float(loss[-1])
    thr = final * (1.0 + float(frac))
    for i, v in enumerate(loss):
        if float(v) <= thr:
            return int(i)
    return int(len(loss) - 1)


def _stack_field(results: List[Dict[str, np.ndarray]], key: str) -> np.ndarray:
    return np.stack([r[key] for r in results], axis=0)


def run_surrogate_experiment(exp_cfg: dict, run_dir: str | Path) -> Path:
    """Run the controlled surrogate experiment and write aggregate NPZ + summary.

    Parameters
    ----------
    exp_cfg:
        Experiment config block (merged YAML).
    run_dir:
        Output directory.

    Returns
    -------
    Path to the aggregate NPZ file.
    """
    run_dir = ensure_dir(run_dir)

    seeds = exp_cfg.get("repeats", {}).get("seeds", None) or [int(exp_cfg.get("seed", 42))]
    seeds = [int(s) for s in seeds]

    per_seed: List[Dict[str, np.ndarray]] = []
    for seed in seeds:
        set_global_seed(seed)

        # Surrogate
        sur_cfg_dict = dict(exp_cfg["surrogate"])
        sur_cfg_dict["seed"] = int(seed)
        surrogate = AcousticSurrogate(AcousticSurrogateConfig(**sur_cfg_dict))

        # Data (noiseless)
        n_train = int(exp_cfg["training"]["n_train"])
        n_test = int(exp_cfg["training"]["n_test"])
        X_train, y_train, _ = surrogate.generate_data(n_train, noise_std_db=0.0)
        X_test, y_test, _ = surrogate.generate_data(n_test, noise_std_db=0.0)

        # Models
        spline_cfg = bspline_layer_config_from_dict(exp_cfg["model"].get("spline", {}))

        abs_cfg_raw = exp_cfg["model"].get("absorption", exp_cfg["model"].get("absorption_term", {}))
        if isinstance(abs_cfg_raw, bool):
            abs_cfg = {"enabled": bool(abs_cfg_raw)}
        elif isinstance(abs_cfg_raw, dict):
            abs_cfg = dict(abs_cfg_raw)
        else:
            abs_cfg = {}

        abs_spline_cfg = bspline_layer_config_from_dict(abs_cfg.get("spline", {}))
        absorption_term_cfg = AbsorptionTermConfig(
            enabled=bool(abs_cfg.get("enabled", False)),
            mode=str(abs_cfg.get("mode", "thorp_scale")),
            reference_fc=bool(abs_cfg.get("reference_fc", True)),
            init_log_scale=float(abs_cfg.get("init_log_scale", 0.0)),
            spline=abs_spline_cfg,
            alpha_floor_db_per_km=float(abs_cfg.get("alpha_floor_db_per_km", 0.0)),
        )

        sonarkad_cfg = SonarKADConfig(
            spline=spline_cfg,
            physics_init_grid_n=int(exp_cfg["model"].get("physics_init_grid_n", 256)),
            SL_db=float(exp_cfg["model"].get("SL_db", surrogate.cfg.SL_db)),
            fc_hz=float(exp_cfg["model"].get("fc_hz", surrogate.cfg.fc_hz)),
            use_absorption=bool(exp_cfg["model"].get("use_absorption", True)),
            f_min_hz=float(getattr(surrogate.cfg, "f_min_hz", 0.0)),
            f_max_hz=float(getattr(surrogate.cfg, "f_max_hz", 0.0)),
            absorption=absorption_term_cfg,
            gauge_fix_each_epoch=bool(exp_cfg["model"].get("gauge_fix_each_epoch", True)),
            gauge_fix_grid_n=int(exp_cfg["model"].get("gauge_fix_grid_n", 200)),
            gauge_fix_interaction=bool(exp_cfg["model"].get("gauge_fix_interaction", True)),
            interaction_rank=int(exp_cfg["model"].get("interaction_rank", 0)),
        )

        sonarkad_phys = SonarKAD(r_min_m=surrogate.r_min, r_max_m=surrogate.r_max, cfg=sonarkad_cfg)
        init_stats = sonarkad_phys.physics_init()
        sonarkad_rand = SonarKAD(r_min_m=surrogate.r_min, r_max_m=surrogate.r_max, cfg=sonarkad_cfg)

        mlp = SmallMLP(hidden=int(exp_cfg["training"].get("hidden_mlp", 16)))

        epochs = int(exp_cfg["training"]["epochs"])
        lr = float(exp_cfg["training"]["lr"])
        criterion = nn.MSELoss()

        progress = bool(exp_cfg.get("training", {}).get("progress_bar", True))
        mp = exp_cfg.get("training", {}).get("matmul_precision", None)
        if mp is not None and hasattr(torch, "set_float32_matmul_precision"):
            try:
                torch.set_float32_matmul_precision(str(mp))
            except Exception:
                pass

        opt_phys = optim.Adam(sonarkad_phys.parameters(), lr=lr)
        opt_rand = optim.Adam(sonarkad_rand.parameters(), lr=lr)
        opt_mlp = optim.Adam(mlp.parameters(), lr=lr)

        loss_phys = np.zeros((epochs,), dtype=np.float64)
        loss_rand = np.zeros((epochs,), dtype=np.float64)
        loss_mlp = np.zeros((epochs,), dtype=np.float64)

        # Optional torch.compile acceleration (PyTorch 2.x)
        if bool(exp_cfg.get("training", {}).get("compile", False)) and hasattr(torch, "compile"):
            try:
                sonarkad_phys = torch.compile(sonarkad_phys)  # type: ignore[attr-defined]
                sonarkad_rand = torch.compile(sonarkad_rand)  # type: ignore[attr-defined]
                mlp = torch.compile(mlp)  # type: ignore[attr-defined]
            except Exception:
                # fall back to eager
                pass

        pbar = tqdm(range(epochs), desc=f"train-surrogate(seed={seed})", disable=not progress)
        for ep in pbar:
            opt_phys.zero_grad()
            l_phys = criterion(sonarkad_phys(X_train), y_train)
            l_phys.backward()
            opt_phys.step()
            loss_phys[ep] = float(l_phys.item())
            if sonarkad_cfg.gauge_fix_each_epoch:
                sonarkad_phys.gauge_fix()

            opt_rand.zero_grad()
            l_rand = criterion(sonarkad_rand(X_train), y_train)
            l_rand.backward()
            opt_rand.step()
            loss_rand[ep] = float(l_rand.item())
            if sonarkad_cfg.gauge_fix_each_epoch:
                sonarkad_rand.gauge_fix()

            opt_mlp.zero_grad()
            l_mlp = criterion(mlp(X_train), y_train)
            l_mlp.backward()
            opt_mlp.step()
            loss_mlp[ep] = float(l_mlp.item())

            if (ep + 1) % int(exp_cfg.get("training", {}).get("log_every", 50)) == 0:
                try:
                    pbar.set_postfix(
                        loss_phys=f"{loss_phys[ep]:.3g}",
                        loss_rand=f"{loss_rand[ep]:.3g}",
                        loss_mlp=f"{loss_mlp[ep]:.3g}",
                    )
                except Exception:
                    pass

        # Clean test RMSE
        sonarkad_phys.eval()
        sonarkad_rand.eval()
        mlp.eval()
        with torch.no_grad():
            rmse_phys_clean = _rmse(sonarkad_phys(X_test), y_test)
            rmse_rand_clean = _rmse(sonarkad_rand(X_test), y_test)
            rmse_mlp_clean = _rmse(mlp(X_test), y_test)

        # Label-noise SNR sweep
        snr_levels = np.array(exp_cfg.get("snr_sweep", {}).get("levels_db", [-5, 0, 5, 10, 15, 20]), dtype=np.float64)
        rmse_phys = np.zeros_like(snr_levels)
        rmse_rand = np.zeros_like(snr_levels)
        rmse_mlp_snr = np.zeros_like(snr_levels)

        sig_var = float(y_train.var().item())
        for i, snr in enumerate(snr_levels):
            noise_var = sig_var / (10.0 ** (snr / 10.0))
            noise_std = float(np.sqrt(noise_var))
            Xn, yn, _ = surrogate.generate_data(n_test, noise_std_db=noise_std)
            with torch.no_grad():
                rmse_phys[i] = _rmse(sonarkad_phys(Xn), yn)
                rmse_rand[i] = _rmse(sonarkad_rand(Xn), yn)
                rmse_mlp_snr[i] = _rmse(mlp(Xn), yn)

        # Component recovery diagnostics
        grid_n = int(exp_cfg.get("diagnostics", {}).get("grid_n", 200))
        r_norm = torch.linspace(0.0, 1.0, grid_n).reshape(-1, 1)
        f_norm = torch.linspace(0.0, 1.0, grid_n).reshape(-1, 1)

        r_phys = r_norm.numpy().reshape(-1) * (surrogate.r_max - surrogate.r_min) + surrogate.r_min
        f_phys = f_norm.numpy().reshape(-1) * (surrogate.f_max - surrogate.f_min) + surrogate.f_min

        tl_base = surrogate.get_transmission_loss_base(r_phys)
        tl_full = surrogate.get_transmission_loss(r_phys)
        theory_r_base = -(tl_base)
        theory_r_full = -(tl_full)
        theory_r_base -= theory_r_base.mean()
        theory_r_full -= theory_r_full.mean()

        ts = surrogate.get_target_strength(f_phys)
        theory_f = ts - ts.mean()

        with torch.no_grad():
            learned_r_phys = sonarkad_phys.phi_r(r_norm).numpy().reshape(-1)
            learned_f_phys = sonarkad_phys.phi_f(f_norm).numpy().reshape(-1)
            learned_r_rand = sonarkad_rand.phi_r(r_norm).numpy().reshape(-1)
            learned_f_rand = sonarkad_rand.phi_f(f_norm).numpy().reshape(-1)

        learned_r_phys -= learned_r_phys.mean()
        learned_f_phys -= learned_f_phys.mean()
        learned_r_rand -= learned_r_rand.mean()
        learned_f_rand -= learned_f_rand.mean()

        comp_rmse_r_full_phys = float(np.sqrt(np.mean((learned_r_phys - theory_r_full) ** 2)))
        comp_rmse_r_full_rand = float(np.sqrt(np.mean((learned_r_rand - theory_r_full) ** 2)))
        comp_corr_r_full_phys = _pearson_corr(learned_r_phys, theory_r_full)
        comp_corr_r_full_rand = _pearson_corr(learned_r_rand, theory_r_full)

        comp_rmse_f_phys = float(np.sqrt(np.mean((learned_f_phys - theory_f) ** 2)))
        comp_rmse_f_rand = float(np.sqrt(np.mean((learned_f_rand - theory_f) ** 2)))
        comp_corr_f_phys = _pearson_corr(learned_f_phys, theory_f)
        comp_corr_f_rand = _pearson_corr(learned_f_rand, theory_f)

        ep_within5_phys = _epochs_to_within(loss_phys, frac=0.05)
        ep_within5_rand = _epochs_to_within(loss_rand, frac=0.05)
        ep_within5_mlp = _epochs_to_within(loss_mlp, frac=0.05)

        res = dict(
            loss_sonarkad_phys=loss_phys,
            loss_sonarkad_rand=loss_rand,
            loss_mlp=loss_mlp,
            rmse_phys_clean=np.array([rmse_phys_clean], dtype=np.float64),
            rmse_rand_clean=np.array([rmse_rand_clean], dtype=np.float64),
            rmse_mlp_clean=np.array([rmse_mlp_clean], dtype=np.float64),
            snr_levels=snr_levels,
            rmse_phys=rmse_phys,
            rmse_rand=rmse_rand,
            rmse_mlp_snr=rmse_mlp_snr,
            r_phys=r_phys,
            f_phys=f_phys,
            theory_r_base=theory_r_base,
            theory_r_full=theory_r_full,
            theory_f=theory_f,
            learned_r_phys=learned_r_phys,
            learned_f_phys=learned_f_phys,
            learned_r_rand=learned_r_rand,
            learned_f_rand=learned_f_rand,
            init_rmse=np.array([float(init_stats["rmse"])], dtype=np.float64),
            init_mean_tl=np.array([float(init_stats["mean_tl"])], dtype=np.float64),
            params_sonarkad=np.array([count_parameters(sonarkad_phys)], dtype=np.int64),
            params_mlp=np.array([count_parameters(mlp)], dtype=np.int64),
            comp_rmse_r_full_phys=np.array([comp_rmse_r_full_phys], dtype=np.float64),
            comp_rmse_r_full_rand=np.array([comp_rmse_r_full_rand], dtype=np.float64),
            comp_corr_r_full_phys=np.array([comp_corr_r_full_phys], dtype=np.float64),
            comp_corr_r_full_rand=np.array([comp_corr_r_full_rand], dtype=np.float64),
            comp_rmse_f_phys=np.array([comp_rmse_f_phys], dtype=np.float64),
            comp_rmse_f_rand=np.array([comp_rmse_f_rand], dtype=np.float64),
            comp_corr_f_phys=np.array([comp_corr_f_phys], dtype=np.float64),
            comp_corr_f_rand=np.array([comp_corr_f_rand], dtype=np.float64),
            ep_within5_phys=np.array([ep_within5_phys], dtype=np.int64),
            ep_within5_rand=np.array([ep_within5_rand], dtype=np.int64),
            ep_within5_mlp=np.array([ep_within5_mlp], dtype=np.int64),
        )
        per_seed.append(res)

        seed_dir = ensure_dir(run_dir / f"seed_{seed}")
        np.savez(seed_dir / "results_seed.npz", **res)

    # Aggregate
    results_agg_path = run_dir / "results_aggregate.npz"
    np.savez(
        results_agg_path,
        seeds=np.array(seeds, dtype=np.int64),
        snr_levels=per_seed[0]["snr_levels"],
        loss_sonarkad_phys=_stack_field(per_seed, "loss_sonarkad_phys"),
        loss_sonarkad_rand=_stack_field(per_seed, "loss_sonarkad_rand"),
        loss_mlp=_stack_field(per_seed, "loss_mlp"),
        rmse_phys_clean=np.concatenate([r["rmse_phys_clean"] for r in per_seed], axis=0),
        rmse_rand_clean=np.concatenate([r["rmse_rand_clean"] for r in per_seed], axis=0),
        rmse_mlp_clean=np.concatenate([r["rmse_mlp_clean"] for r in per_seed], axis=0),
        rmse_phys=_stack_field(per_seed, "rmse_phys"),
        rmse_rand=_stack_field(per_seed, "rmse_rand"),
        rmse_mlp_snr=_stack_field(per_seed, "rmse_mlp_snr"),
        r_phys=per_seed[0]["r_phys"],
        f_phys=per_seed[0]["f_phys"],
        theory_r_base=per_seed[0]["theory_r_base"],
        theory_r_full=per_seed[0]["theory_r_full"],
        theory_f=per_seed[0]["theory_f"],
        learned_r_phys=_stack_field(per_seed, "learned_r_phys"),
        learned_f_phys=_stack_field(per_seed, "learned_f_phys"),
        learned_r_rand=_stack_field(per_seed, "learned_r_rand"),
        learned_f_rand=_stack_field(per_seed, "learned_f_rand"),
        init_rmse=np.concatenate([r["init_rmse"] for r in per_seed], axis=0),
        init_mean_tl=np.concatenate([r["init_mean_tl"] for r in per_seed], axis=0),
        params_sonarkad=per_seed[0]["params_sonarkad"],
        params_mlp=per_seed[0]["params_mlp"],
        comp_rmse_r_full_phys=np.concatenate([r["comp_rmse_r_full_phys"] for r in per_seed], axis=0),
        comp_rmse_r_full_rand=np.concatenate([r["comp_rmse_r_full_rand"] for r in per_seed], axis=0),
        comp_corr_r_full_phys=np.concatenate([r["comp_corr_r_full_phys"] for r in per_seed], axis=0),
        comp_corr_r_full_rand=np.concatenate([r["comp_corr_r_full_rand"] for r in per_seed], axis=0),
        comp_rmse_f_phys=np.concatenate([r["comp_rmse_f_phys"] for r in per_seed], axis=0),
        comp_rmse_f_rand=np.concatenate([r["comp_rmse_f_rand"] for r in per_seed], axis=0),
        comp_corr_f_phys=np.concatenate([r["comp_corr_f_phys"] for r in per_seed], axis=0),
        comp_corr_f_rand=np.concatenate([r["comp_corr_f_rand"] for r in per_seed], axis=0),
        ep_within5_phys=np.concatenate([r["ep_within5_phys"] for r in per_seed], axis=0),
        ep_within5_rand=np.concatenate([r["ep_within5_rand"] for r in per_seed], axis=0),
        ep_within5_mlp=np.concatenate([r["ep_within5_mlp"] for r in per_seed], axis=0),
    )

    def _mean_std(x: np.ndarray) -> Tuple[float, float]:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        return float(np.mean(x)), float(np.std(x, ddof=1) if x.size > 1 else 0.0)

    summary = {
        "n_seeds": int(len(seeds)),
        "seeds": seeds,
        "params_sonarkad": int(per_seed[0]["params_sonarkad"][0]),
        "params_mlp": int(per_seed[0]["params_mlp"][0]),
        "rmse_clean": {
            "sonarkad_phys_mean_std": _mean_std(np.concatenate([r["rmse_phys_clean"] for r in per_seed], axis=0)),
            "sonarkad_rand_mean_std": _mean_std(np.concatenate([r["rmse_rand_clean"] for r in per_seed], axis=0)),
            "mlp_mean_std": _mean_std(np.concatenate([r["rmse_mlp_clean"] for r in per_seed], axis=0)),
        },
        "component_recovery": {
            "phi_r_rmse_full": {
                "sonarkad_phys_mean_std": _mean_std(np.concatenate([r["comp_rmse_r_full_phys"] for r in per_seed], axis=0)),
                "sonarkad_rand_mean_std": _mean_std(np.concatenate([r["comp_rmse_r_full_rand"] for r in per_seed], axis=0)),
            },
            "phi_r_corr_full": {
                "sonarkad_phys_mean_std": _mean_std(np.concatenate([r["comp_corr_r_full_phys"] for r in per_seed], axis=0)),
                "sonarkad_rand_mean_std": _mean_std(np.concatenate([r["comp_corr_r_full_rand"] for r in per_seed], axis=0)),
            },
            "phi_f_rmse": {
                "sonarkad_phys_mean_std": _mean_std(np.concatenate([r["comp_rmse_f_phys"] for r in per_seed], axis=0)),
                "sonarkad_rand_mean_std": _mean_std(np.concatenate([r["comp_rmse_f_rand"] for r in per_seed], axis=0)),
            },
            "phi_f_corr": {
                "sonarkad_phys_mean_std": _mean_std(np.concatenate([r["comp_corr_f_phys"] for r in per_seed], axis=0)),
                "sonarkad_rand_mean_std": _mean_std(np.concatenate([r["comp_corr_f_rand"] for r in per_seed], axis=0)),
            },
        },
        "convergence_speed": {
            "epochs_to_within_5pct_final_loss_mean_std": {
                "sonarkad_phys": _mean_std(np.concatenate([r["ep_within5_phys"] for r in per_seed], axis=0)),
                "sonarkad_rand": _mean_std(np.concatenate([r["ep_within5_rand"] for r in per_seed], axis=0)),
                "mlp": _mean_std(np.concatenate([r["ep_within5_mlp"] for r in per_seed], axis=0)),
            }
        },
        "physics_init": {
            "ls_projection_rmse_mean_std": _mean_std(np.concatenate([r["init_rmse"] for r in per_seed], axis=0)),
            "mean_tl_mean_std": _mean_std(np.concatenate([r["init_mean_tl"] for r in per_seed], axis=0)),
        },
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return results_agg_path
