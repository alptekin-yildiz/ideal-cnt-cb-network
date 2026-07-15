#!/usr/bin/env python3
"""Affine-strain Simmons GF sweep for the companion manuscript.

This wrapper is the GF sibling of ``run_simmons_conductivity_regime_pub.py``.
It uses the same selected CNT/CB distributions and Simmons parameters, but
samples each RVE once and solves the deformed network over a small strain grid.

Outputs:
  data/processed/gf/gf_simmons_<tag>.csv
  data/processed/gf/gf_simmons_<tag>_realizations.csv

Use distinct ``--tag`` values under ``data/processed/gf`` for
mechanism checks, sensitivity sweeps, and exploratory weak-link runs.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# Put the repo root on sys.path so a bare checkout runs without pip install;
# the cntcb imports below intentionally follow this bootstrap (hence E402).
ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT
DATA_DIR = PAPER / "data" / "processed"
PHIC_DIR = DATA_DIR / "percolation"
CONDUCTIVITY_DIR = DATA_DIR / "conductivity"
OUT_DIR = DATA_DIR / "gf"

sys.path.insert(0, str(ROOT))

from cntcb.engines.gf_engine import (  # noqa: E402
    SimmonsGFResult,
    run_simmons_gf_realization,
)
from cntcb.engines.percolation_engine import (  # noqa: E402
    estimate_cb_n_max,
    estimate_cnt_n_max,
    solve_truncated_lognormal,
)


CB_MU_D_NM = 148.0
CB_SIGMA_D_NM = 83.0
CB_D_MIN_NM = 20.0
CB_D_MAX_NM = 500.0
CB_MONO_D_NM = 148.0

CNT_MU_L_NM = 500.0
CNT_SIGMA_L_NM = 300.0
CNT_L_MIN_NM = 50.0
CNT_L_MAX_NM = 1500.0
CNT_MONO_L_NM = 500.0
CNT_DIAM_NM = 10.0
CNT_WAVINESS = 0.7
SEG_LEN_UNIT_NM = 50.0

L_RVE_NM = 4000.0
L_RVE_CB_NM = 8000.0
L_RVE_CNT_NM = 4000.0
L_RVE_HYBRID_NM = 4000.0
TUNNEL_NM = 10.0
D_MIN_TUNNEL_NM = 0.34
G_CUTOFF_S = 1.0e-15
POISSON_NU = 0.36
STRAIN_GRID = (0.0, 0.0025, 0.005, 0.0075, 0.01)
SEED_BASE = 20260617
SAFETY = 2.5

DEFAULT_SYSTEMS = ("cb", "cnt", "hybrid")
DEFAULT_MULTIPLIERS = (2.0,)
DEFAULT_HYBRID_CB_WT = 1.0

CNT_PHIC_CSV = PHIC_DIR / "cnt_phic_sweep_fixedstep_Lmin50.csv"
CB_PHIC_CSV = PHIC_DIR / "cb_phic_sweep_Loverd60_momentmatched_D20_500_n300.csv"
HYBRID_PHIC_CSV = PHIC_DIR / "hybrid_phic_boundary_n400_modes.csv"
MONO_UNIVERSALITY_FITS_CSV = (
    CONDUCTIVITY_DIR / "conductivity_universality_mono_validation_n100_diag_fits.csv"
)


SUMMARY_HEADER = [
    "system",
    "system_label",
    "variable_filler",
    "population_mode",
    "phi_multiplier",
    "phi_c_ref_vol",
    "phi_cb_target_vol",
    "phi_cb_actual_mean_vol",
    "phi_cnt_target_vol",
    "phi_cnt_actual_mean_vol",
    "n_ok",
    "n_fail",
    "n_total",
    "failure_reasons",
    "GF_R_median",
    "GF_R_mean",
    "GF_R_std",
    "GF_R_p05",
    "GF_R_p25",
    "GF_R_p75",
    "GF_R_p95",
    "GF_sigma_median",
    "GF_sigma_mean",
    "GF_sigma_std",
    "GF_geom_median",
    "GF_R_from_sigma_median",
    "GF_R_minus_recon_median",
    "R2_lnG_median",
    "R2_lnG_p05",
    "R2_lnsigma_median",
    "R2_lnsigma_p05",
    "linearity_flag_threshold",
    "linearity_flag_count",
    "linearity_flag_fraction",
    "topology_switch_median",
    "topology_switch_p95",
    "log10_G_range_max_median",
    "residual_ratio_max_median",
    "opening_alpha_ss_nm_per_strain",
    "opening_alpha_sc_nm_per_strain",
    "opening_alpha_cc_nm_per_strain",
    "G_eff_eps0_mean_S",
    "G_eff_epsmax_mean_S",
    "sigma_x_eps0_mean_Sm",
    "sigma_x_epsmax_mean_Sm",
    "n_cb_mean",
    "n_cnt_mean",
    "n_total_segs_mean",
    "mean_d_realized_nm",
    "std_d_realized_nm",
    "mean_L_realized_nm",
    "std_L_realized_nm",
    "mean_n_seg_per_cnt",
    "max_n_seg_per_cnt",
    "elapsed_s",
    "L_RVE_nm",
    "strain_grid",
    "poisson_nu",
    "tunnel_nm",
    "d_min_tunnel_nm",
    "G_cutoff_S",
    "CB_mu_d_nm",
    "CB_sigma_d_nm",
    "CB_D_min_nm",
    "CB_D_max_nm",
    "CNT_mu_L_nm",
    "CNT_sigma_L_nm",
    "CNT_L_min_nm",
    "CNT_L_max_nm",
    "CNT_diam_nm",
    "CNT_waviness",
    "seg_len_unit_nm",
]


REAL_HEADER = [
    "system",
    "system_label",
    "variable_filler",
    "population_mode",
    "phi_multiplier",
    "realization",
    "seed",
    "success",
    "failure_reason",
    "failure_stage",
    "n_strain_ok",
    "GF_R",
    "GF_sigma",
    "GF_geom",
    "GF_R_from_sigma",
    "GF_R_minus_recon",
    "R2_lnG_vs_eps",
    "max_resid_lnG",
    "R2_lnsigma_vs_eps",
    "max_resid_lnsigma",
    "G_eff_eps0_S",
    "G_eff_epsmax_S",
    "sigma_x_eps0_Sm",
    "sigma_x_epsmax_Sm",
    "topology_switch_fraction",
    "log10_G_range_max",
    "residual_ratio_max",
    "opening_alpha_ss_nm_per_strain",
    "opening_alpha_sc_nm_per_strain",
    "opening_alpha_cc_nm_per_strain",
    "power_frac_ss_eps0",
    "power_frac_sc_eps0",
    "power_frac_cc_eps0",
    "power_frac_ss_epsmax",
    "power_frac_sc_epsmax",
    "power_frac_cc_epsmax",
    "phi_cb_target_vol",
    "phi_cb_actual_vol",
    "phi_cnt_target_vol",
    "phi_cnt_actual_vol",
    "n_cb",
    "n_cnt",
    "n_total_segs",
    "mean_d_realized_nm",
    "std_d_realized_nm",
    "mean_L_realized_nm",
    "std_L_realized_nm",
    "mean_n_seg_per_cnt",
    "max_n_seg_per_cnt",
    "strain_grid",
    "G_eff_series_S",
    "sigma_x_series_Sm",
    "n_edges_series",
    "power_total_series",
    "power_frac_ss_series",
    "power_frac_sc_series",
    "power_frac_cc_series",
    "largest_component_fraction_series",
    "log10_G_range_series",
    "residual_ratio_series",
    "failure_by_eps",
    "elapsed_s",
    "L_RVE_nm",
]


def _safe_tag(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    if not safe:
        raise ValueError("--tag must contain at least one safe character.")
    return safe


def _parse_csv_list(text: str, cast=str) -> tuple:
    values = []
    for chunk in text.split(","):
        item = chunk.strip()
        if item:
            values.append(cast(item))
    if not values:
        raise ValueError("List argument is empty.")
    return tuple(values)


def _parse_strain_grid(text: str) -> tuple[float, ...]:
    vals = tuple(float(v) for v in _parse_csv_list(text, float))
    if len(vals) < 3:
        raise argparse.ArgumentTypeError("strain grid must contain at least three values")
    if any(v < 0.0 for v in vals) or tuple(sorted(vals)) != vals:
        raise argparse.ArgumentTypeError("strain grid must be non-negative and sorted")
    return vals


def _fmt(value) -> str:
    if isinstance(value, str):
        return value
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "" if value is None else str(value)
    if not np.isfinite(value):
        return ""
    return f"{value:.8g}"


def _fmt_series(values) -> str:
    return ";".join(_fmt(v) for v in values)


def _output_paths(out_dir: Path, tag: str) -> tuple[Path, Path]:
    base = f"gf_simmons_{_safe_tag(tag)}"
    return out_dir / f"{base}.csv", out_dir / f"{base}_realizations.csv"


def _prepare_output(path: Path, header: list[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Use --overwrite or choose a different --tag."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        csv.writer(f).writerow(header)


def _find_row(path: Path, predicate) -> dict[str, str]:
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if predicate(row):
                return row
    raise ValueError(f"No matching row found in {path}.")


def _load_phi_c_cb(path: Path) -> float:
    row = _find_row(
        path,
        lambda r: abs(float(r["mu_d_nm"]) - CB_MU_D_NM) < 1e-9
        and abs(float(r["sigma_d_nm"]) - CB_SIGMA_D_NM) < 1e-9,
    )
    return float(row["phi_c_vol"])


def _load_phi_c_cnt(path: Path) -> float:
    row = _find_row(
        path,
        lambda r: abs(float(r["mu_L_nm"]) - CNT_MU_L_NM) < 1e-9
        and abs(float(r["sigma_L_nm"]) - CNT_SIGMA_L_NM) < 1e-9,
    )
    return float(row["phi_c_vol"])


def _load_phi_c_mono_fit(path: Path, system: str) -> float:
    row = _find_row(
        path,
        lambda r: r["system"] == system and r["fit_mode"] == "ok_ge_0p9_mean",
    )
    return float(row["phi_c_ref_vol"])


def _load_phi_c_hybrid(path: Path, w_cb_wt: float, mode: str) -> tuple[float, float]:
    row = _find_row(
        path,
        lambda r: r["mode"] == mode and abs(float(r["w_cb_wt_pct"]) - w_cb_wt) < 1e-9,
    )
    return float(row["phi_cb_vol"]), float(row["phi_cnt_c_median_vol"])


def _system_l_rve_nm(args: argparse.Namespace, system: str) -> float:
    if system == "cb" and args.l_rve_cb_nm is not None:
        return float(args.l_rve_cb_nm)
    if system == "cnt" and args.l_rve_cnt_nm is not None:
        return float(args.l_rve_cnt_nm)
    if system == "hybrid" and args.l_rve_hybrid_nm is not None:
        return float(args.l_rve_hybrid_nm)
    return float(args.l_rve_nm)


def _system_spec(
    system: str,
    hybrid_cb_wt: float,
    population_mode: str,
) -> dict[str, float | str]:
    system = system.lower()
    population_mode = population_mode.lower()
    if population_mode not in {"poly", "mono"}:
        raise ValueError("--population-mode must be 'poly' or 'mono'.")

    cb_mu = CB_MU_D_NM
    cb_sigma = CB_SIGMA_D_NM
    cb_label = "selected polydisperse"
    cnt_mu = CNT_MU_L_NM
    cnt_sigma = CNT_SIGMA_L_NM
    cnt_label = "selected polydisperse"
    hybrid_mode = "PP"
    cb_phi_c = _load_phi_c_cb(CB_PHIC_CSV)
    cnt_phi_c = _load_phi_c_cnt(CNT_PHIC_CSV)

    if population_mode == "mono":
        cb_mu = CB_MONO_D_NM
        cb_sigma = 0.0
        cb_label = "monodisperse"
        cnt_mu = CNT_MONO_L_NM
        cnt_sigma = 0.0
        cnt_label = "monodisperse"
        hybrid_mode = "MM"
        cb_phi_c = _load_phi_c_mono_fit(MONO_UNIVERSALITY_FITS_CSV, "cb_mono")
        cnt_phi_c = _load_phi_c_mono_fit(MONO_UNIVERSALITY_FITS_CSV, "cnt_mono_w07")

    if system == "cb":
        return {
            "system": "cb",
            "system_label": f"pure CB, {cb_label}",
            "variable_filler": "CB",
            "phi_cb_fixed": 0.0,
            "phi_cnt_fixed": 0.0,
            "phi_c_ref": cb_phi_c,
            "cb_mu_d_nm": cb_mu,
            "cb_sigma_d_nm": cb_sigma,
            "cnt_mu_L_nm": cnt_mu,
            "cnt_sigma_L_nm": cnt_sigma,
            "hybrid_mode": hybrid_mode,
        }
    if system == "cnt":
        return {
            "system": "cnt",
            "system_label": f"pure CNT, {cnt_label}",
            "variable_filler": "CNT",
            "phi_cb_fixed": 0.0,
            "phi_cnt_fixed": 0.0,
            "phi_c_ref": cnt_phi_c,
            "cb_mu_d_nm": cb_mu,
            "cb_sigma_d_nm": cb_sigma,
            "cnt_mu_L_nm": cnt_mu,
            "cnt_sigma_L_nm": cnt_sigma,
            "hybrid_mode": hybrid_mode,
        }
    if system == "hybrid":
        phi_cb, phi_cnt_c = _load_phi_c_hybrid(HYBRID_PHIC_CSV, hybrid_cb_wt, hybrid_mode)
        return {
            "system": "hybrid",
            "system_label": f"hybrid {hybrid_mode}, fixed CB={hybrid_cb_wt:g} wt%",
            "variable_filler": "CNT",
            "phi_cb_fixed": phi_cb,
            "phi_cnt_fixed": 0.0,
            "phi_c_ref": phi_cnt_c,
            "cb_mu_d_nm": cb_mu,
            "cb_sigma_d_nm": cb_sigma,
            "cnt_mu_L_nm": cnt_mu,
            "cnt_sigma_L_nm": cnt_sigma,
            "hybrid_mode": hybrid_mode,
        }
    raise ValueError(f"Unknown system {system!r}; use cb,cnt,hybrid.")


def _run_one(args: tuple) -> SimmonsGFResult:
    (
        cb_model,
        cnt_model,
        phi_cb_target,
        phi_cnt_target,
        l_rve_nm,
        tunnel_nm,
        d_min_nm,
        g_cutoff_s,
        strain_grid,
        poisson_nu,
        seed,
        n_cb_max,
        n_cnt_max,
        waviness,
        opening_alpha_ss,
        opening_alpha_sc,
        opening_alpha_cc,
    ) = args
    return run_simmons_gf_realization(
        cb_model=cb_model,
        cnt_model=cnt_model,
        phi_cb_target_vol=phi_cb_target,
        phi_cnt_target_vol=phi_cnt_target,
        l_rve_nm=l_rve_nm,
        cnt_diam_nm=CNT_DIAM_NM,
        waviness=waviness,
        tunnel_nm=tunnel_nm,
        seg_len_unit_nm=SEG_LEN_UNIT_NM,
        d_min_nm=d_min_nm,
        g_cutoff_s=g_cutoff_s,
        strain_grid=strain_grid,
        poisson_nu=poisson_nu,
        seed=seed,
        opening_alpha_ss_nm_per_strain=opening_alpha_ss,
        opening_alpha_sc_nm_per_strain=opening_alpha_sc,
        opening_alpha_cc_nm_per_strain=opening_alpha_cc,
        n_cb_max=n_cb_max,
        n_cnt_max=n_cnt_max,
    )


def _run_tasks(
    tasks: list[tuple],
    workers: int,
    *,
    seeds: list[int],
    on_result,
    progress_every: int,
) -> list[SimmonsGFResult]:
    results: list[SimmonsGFResult | None] = [None] * len(tasks)
    t0 = time.time()
    n_done = 0

    def record(idx: int, result: SimmonsGFResult) -> None:
        nonlocal n_done
        results[idx - 1] = result
        n_done += 1
        on_result(idx, seeds[idx - 1], result)
        if progress_every > 0 and (n_done % progress_every == 0 or n_done == len(tasks)):
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0.0
            eta = (len(tasks) - n_done) / rate if rate > 0 else float("nan")
            n_ok = sum(1 for r in results if r is not None and r.success)
            n_fail = sum(1 for r in results if r is not None and not r.success)
            print(
                f"    progress {n_done}/{len(tasks)}  ok={n_ok}  fail={n_fail}  "
                f"elapsed={elapsed:.1f}s  eta={eta:.1f}s",
                flush=True,
            )

    if workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            record(idx, _run_one(task))
        return [r for r in results if r is not None]

    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {
                pool.submit(_run_one, task): idx for idx, task in enumerate(tasks, start=1)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                record(idx, future.result())
    except PermissionError as exc:
        warnings.warn(
            "ProcessPoolExecutor could not start in this environment "
            f"({exc}); falling back to sequential execution.",
            RuntimeWarning,
        )
        for idx, task in enumerate(tasks, start=1):
            if results[idx - 1] is None:
                record(idx, _run_one(task))

    return [r for r in results if r is not None]


def _finite_values(results: list[SimmonsGFResult], attr: str) -> np.ndarray:
    values = np.asarray([float(getattr(r, attr)) for r in results], dtype=float)
    return values[np.isfinite(values)]


def _stat(values: np.ndarray, name: str) -> float:
    if len(values) == 0:
        return float("nan")
    if name == "mean":
        return float(np.mean(values))
    if name == "median":
        return float(np.median(values))
    if name == "std":
        return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    if name.startswith("p"):
        return float(np.percentile(values, float(name[1:])))
    raise ValueError(name)


def _realization_row(
    *,
    spec: dict[str, float | str],
    population_mode: str,
    multiplier: float,
    realization_idx: int,
    seed: int,
    phi_cb_target: float,
    phi_cnt_target: float,
    l_rve_nm: float,
    res: SimmonsGFResult,
) -> list:
    return [
        spec["system"],
        spec["system_label"],
        spec["variable_filler"],
        population_mode,
        _fmt(multiplier),
        realization_idx,
        seed,
        res.success,
        res.failure_reason,
        res.failure_stage,
        res.n_strain_ok,
        _fmt(res.gf_r),
        _fmt(res.gf_sigma),
        _fmt(res.gf_geom),
        _fmt(res.gf_r_from_sigma),
        _fmt(res.gf_r_minus_recon),
        _fmt(res.r2_lng_vs_eps),
        _fmt(res.max_resid_lng),
        _fmt(res.r2_lnsigma_vs_eps),
        _fmt(res.max_resid_lnsigma),
        _fmt(res.g_eff_eps0_s),
        _fmt(res.g_eff_epsmax_s),
        _fmt(res.sigma_x_eps0_s_per_m),
        _fmt(res.sigma_x_epsmax_s_per_m),
        _fmt(res.topology_switch_fraction),
        _fmt(res.log10_g_range_max),
        _fmt(res.residual_ratio_max),
        _fmt(res.opening_alpha_ss_nm_per_strain),
        _fmt(res.opening_alpha_sc_nm_per_strain),
        _fmt(res.opening_alpha_cc_nm_per_strain),
        _fmt(res.power_frac_ss_eps0),
        _fmt(res.power_frac_sc_eps0),
        _fmt(res.power_frac_cc_eps0),
        _fmt(res.power_frac_ss_epsmax),
        _fmt(res.power_frac_sc_epsmax),
        _fmt(res.power_frac_cc_epsmax),
        _fmt(phi_cb_target),
        _fmt(res.phi_cb_actual_vol),
        _fmt(phi_cnt_target),
        _fmt(res.phi_cnt_actual_vol),
        res.n_cb,
        res.n_cnt,
        res.n_total_segs,
        _fmt(res.mean_d_realized_nm),
        _fmt(res.std_d_realized_nm),
        _fmt(res.mean_L_realized_nm),
        _fmt(res.std_L_realized_nm),
        _fmt(res.mean_n_seg_per_cnt),
        res.max_n_seg_per_cnt,
        _fmt_series(res.strain_grid),
        _fmt_series(res.g_eff_series_s),
        _fmt_series(res.sigma_x_series_s_per_m),
        _fmt_series(res.n_edges_series),
        _fmt_series(res.power_total_series),
        _fmt_series(res.power_frac_ss_series),
        _fmt_series(res.power_frac_sc_series),
        _fmt_series(res.power_frac_cc_series),
        _fmt_series(res.largest_component_fraction_series),
        _fmt_series(res.log10_g_range_series),
        _fmt_series(res.residual_ratio_series),
        ";".join(res.failure_by_eps),
        f"{res.elapsed_s:.3f}",
        _fmt(l_rve_nm),
    ]


def _append_realization(
    path: Path,
    spec: dict[str, float | str],
    population_mode: str,
    multiplier: float,
    realization_idx: int,
    seed: int,
    phi_cb_target: float,
    phi_cnt_target: float,
    l_rve_nm: float,
    res: SimmonsGFResult,
) -> None:
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            _realization_row(
                spec=spec,
                population_mode=population_mode,
                multiplier=multiplier,
                realization_idx=realization_idx,
                seed=seed,
                phi_cb_target=phi_cb_target,
                phi_cnt_target=phi_cnt_target,
                l_rve_nm=l_rve_nm,
                res=res,
            )
        )
        f.flush()


def _summary_row(
    spec: dict[str, float | str],
    population_mode: str,
    multiplier: float,
    phi_cb_target: float,
    phi_cnt_target: float,
    results: list[SimmonsGFResult],
    elapsed_s: float,
    l_rve_nm: float,
    args: argparse.Namespace,
) -> dict[str, str]:
    valid = [r for r in results if r.success]
    failure_counts = Counter(r.failure_reason for r in results if not r.success)
    failures = "|".join(f"{k}:{v}" for k, v in sorted(failure_counts.items()))
    r2_thresh = float(args.linearity_r2_threshold)
    n_flag = sum(
        r.success
        and (
            (np.isfinite(r.r2_lng_vs_eps) and r.r2_lng_vs_eps < r2_thresh)
            or (np.isfinite(r.r2_lnsigma_vs_eps) and r.r2_lnsigma_vs_eps < r2_thresh)
        )
        for r in results
    )
    n_ok = len(valid)

    def vals(attr: str) -> np.ndarray:
        return _finite_values(valid, attr)

    def mean_all(attr: str) -> float:
        arr = _finite_values(results, attr)
        return float(np.mean(arr)) if len(arr) else float("nan")

    return {
        "system": str(spec["system"]),
        "system_label": str(spec["system_label"]),
        "variable_filler": str(spec["variable_filler"]),
        "population_mode": population_mode,
        "phi_multiplier": _fmt(multiplier),
        "phi_c_ref_vol": _fmt(spec["phi_c_ref"]),
        "phi_cb_target_vol": _fmt(phi_cb_target),
        "phi_cb_actual_mean_vol": _fmt(mean_all("phi_cb_actual_vol")),
        "phi_cnt_target_vol": _fmt(phi_cnt_target),
        "phi_cnt_actual_mean_vol": _fmt(mean_all("phi_cnt_actual_vol")),
        "n_ok": str(n_ok),
        "n_fail": str(len(results) - n_ok),
        "n_total": str(len(results)),
        "failure_reasons": failures,
        "GF_R_median": _fmt(_stat(vals("gf_r"), "median")),
        "GF_R_mean": _fmt(_stat(vals("gf_r"), "mean")),
        "GF_R_std": _fmt(_stat(vals("gf_r"), "std")),
        "GF_R_p05": _fmt(_stat(vals("gf_r"), "p5")),
        "GF_R_p25": _fmt(_stat(vals("gf_r"), "p25")),
        "GF_R_p75": _fmt(_stat(vals("gf_r"), "p75")),
        "GF_R_p95": _fmt(_stat(vals("gf_r"), "p95")),
        "GF_sigma_median": _fmt(_stat(vals("gf_sigma"), "median")),
        "GF_sigma_mean": _fmt(_stat(vals("gf_sigma"), "mean")),
        "GF_sigma_std": _fmt(_stat(vals("gf_sigma"), "std")),
        "GF_geom_median": _fmt(_stat(vals("gf_geom"), "median")),
        "GF_R_from_sigma_median": _fmt(_stat(vals("gf_r_from_sigma"), "median")),
        "GF_R_minus_recon_median": _fmt(_stat(vals("gf_r_minus_recon"), "median")),
        "R2_lnG_median": _fmt(_stat(vals("r2_lng_vs_eps"), "median")),
        "R2_lnG_p05": _fmt(_stat(vals("r2_lng_vs_eps"), "p5")),
        "R2_lnsigma_median": _fmt(_stat(vals("r2_lnsigma_vs_eps"), "median")),
        "R2_lnsigma_p05": _fmt(_stat(vals("r2_lnsigma_vs_eps"), "p5")),
        "linearity_flag_threshold": _fmt(r2_thresh),
        "linearity_flag_count": str(n_flag),
        "linearity_flag_fraction": _fmt(n_flag / n_ok if n_ok else float("nan")),
        "topology_switch_median": _fmt(_stat(vals("topology_switch_fraction"), "median")),
        "topology_switch_p95": _fmt(_stat(vals("topology_switch_fraction"), "p95")),
        "log10_G_range_max_median": _fmt(_stat(vals("log10_g_range_max"), "median")),
        "residual_ratio_max_median": _fmt(_stat(vals("residual_ratio_max"), "median")),
        "opening_alpha_ss_nm_per_strain": _fmt(args.opening_alpha_ss_nm_per_strain),
        "opening_alpha_sc_nm_per_strain": _fmt(args.opening_alpha_sc_nm_per_strain),
        "opening_alpha_cc_nm_per_strain": _fmt(args.opening_alpha_cc_nm_per_strain),
        "G_eff_eps0_mean_S": _fmt(_stat(vals("g_eff_eps0_s"), "mean")),
        "G_eff_epsmax_mean_S": _fmt(_stat(vals("g_eff_epsmax_s"), "mean")),
        "sigma_x_eps0_mean_Sm": _fmt(_stat(vals("sigma_x_eps0_s_per_m"), "mean")),
        "sigma_x_epsmax_mean_Sm": _fmt(_stat(vals("sigma_x_epsmax_s_per_m"), "mean")),
        "n_cb_mean": _fmt(mean_all("n_cb")),
        "n_cnt_mean": _fmt(mean_all("n_cnt")),
        "n_total_segs_mean": _fmt(mean_all("n_total_segs")),
        "mean_d_realized_nm": _fmt(mean_all("mean_d_realized_nm")),
        "std_d_realized_nm": _fmt(mean_all("std_d_realized_nm")),
        "mean_L_realized_nm": _fmt(mean_all("mean_L_realized_nm")),
        "std_L_realized_nm": _fmt(mean_all("std_L_realized_nm")),
        "mean_n_seg_per_cnt": _fmt(mean_all("mean_n_seg_per_cnt")),
        "max_n_seg_per_cnt": str(max((r.max_n_seg_per_cnt for r in results), default=0)),
        "elapsed_s": f"{elapsed_s:.3f}",
        "L_RVE_nm": _fmt(l_rve_nm),
        "strain_grid": _fmt_series(args.strain_grid),
        "poisson_nu": _fmt(args.poisson_nu),
        "tunnel_nm": _fmt(args.tunnel_nm),
        "d_min_tunnel_nm": _fmt(args.d_min_nm),
        "G_cutoff_S": _fmt(args.g_cutoff_s),
        "CB_mu_d_nm": _fmt(spec["cb_mu_d_nm"]),
        "CB_sigma_d_nm": _fmt(spec["cb_sigma_d_nm"]),
        "CB_D_min_nm": _fmt(CB_D_MIN_NM),
        "CB_D_max_nm": _fmt(CB_D_MAX_NM),
        "CNT_mu_L_nm": _fmt(spec["cnt_mu_L_nm"]),
        "CNT_sigma_L_nm": _fmt(spec["cnt_sigma_L_nm"]),
        "CNT_L_min_nm": _fmt(CNT_L_MIN_NM),
        "CNT_L_max_nm": _fmt(CNT_L_MAX_NM),
        "CNT_diam_nm": _fmt(CNT_DIAM_NM),
        "CNT_waviness": _fmt(CNT_WAVINESS),
        "seg_len_unit_nm": _fmt(SEG_LEN_UNIT_NM),
    }


def _write_summary(path: Path, row: dict[str, str]) -> None:
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=SUMMARY_HEADER).writerow(row)


def run(args: argparse.Namespace) -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    out_dir = Path(args.out_dir).expanduser().resolve()
    summary_csv, real_csv = _output_paths(out_dir, args.tag)
    _prepare_output(summary_csv, SUMMARY_HEADER, args.overwrite)
    _prepare_output(real_csv, REAL_HEADER, args.overwrite)

    systems = tuple(s.lower() for s in _parse_csv_list(args.systems, str))
    multipliers = tuple(float(v) for v in _parse_csv_list(args.phi_multipliers, float))

    print("=" * 78)
    print("Affine-strain Simmons GF sweep")
    print(f"  tag={args.tag}  N={args.n_real}  workers={args.workers}")
    print(f"  population_mode={args.population_mode}")
    print(f"  systems={','.join(systems)}  multipliers={','.join(f'{m:g}' for m in multipliers)}")
    print(
        f"  L_RVE: CB={args.l_rve_cb_nm:g} nm, CNT={args.l_rve_cnt_nm:g} nm, "
        f"hybrid={args.l_rve_hybrid_nm:g} nm"
    )
    print(
        f"  tunnel={args.tunnel_nm:g} nm  d_min={args.d_min_nm:g} nm  "
        f"G_cut={args.g_cutoff_s:g} S  nu={args.poisson_nu:g}"
    )
    print(
        "  opening_alpha [nm/strain]: "
        f"CB-CB={args.opening_alpha_ss_nm_per_strain:g}, "
        f"CB-CNT={args.opening_alpha_sc_nm_per_strain:g}, "
        f"CNT-CNT={args.opening_alpha_cc_nm_per_strain:g}"
    )
    print(f"  strain={list(args.strain_grid)}")
    print(f"  summary: {summary_csv}")
    print(f"  realizations: {real_csv}")
    print("=" * 78)

    for sys_idx, system in enumerate(systems):
        spec = _system_spec(system, args.hybrid_cb_wt, args.population_mode)
        l_rve_nm = _system_l_rve_nm(args, system)
        cb_model = solve_truncated_lognormal(
            float(spec["cb_mu_d_nm"]),
            float(spec["cb_sigma_d_nm"]),
            CB_D_MIN_NM,
            CB_D_MAX_NM,
        )
        cnt_model = solve_truncated_lognormal(
            float(spec["cnt_mu_L_nm"]),
            float(spec["cnt_sigma_L_nm"]),
            CNT_L_MIN_NM,
            CNT_L_MAX_NM,
        )
        phi_c = float(spec["phi_c_ref"])
        variable = str(spec["variable_filler"])
        print(
            f"\n--- {spec['system']}: {spec['system_label']} "
            f"(variable={variable}, phi_c={phi_c:.6g}, L_RVE={l_rve_nm:g} nm) ---"
        )

        for mult_idx, multiplier in enumerate(multipliers):
            if variable == "CB":
                phi_cb_target = phi_c * multiplier
                phi_cnt_target = 0.0
            else:
                phi_cb_target = float(spec["phi_cb_fixed"])
                phi_cnt_target = phi_c * multiplier

            n_cb_max = estimate_cb_n_max(
                cb_model,
                max(phi_cb_target, 1e-12),
                l_rve_nm,
                safety=SAFETY,
            )
            n_cnt_max = estimate_cnt_n_max(
                cnt_model,
                max(phi_cnt_target, 1e-12),
                l_rve_nm,
                CNT_DIAM_NM,
                safety=SAFETY,
            )
            seeds = [
                args.seed_base + sys_idx * 1_000_000 + mult_idx * 100_000 + r
                for r in range(args.n_real)
            ]
            tasks = [
                (
                    cb_model,
                    cnt_model,
                    phi_cb_target,
                    phi_cnt_target,
                    l_rve_nm,
                    args.tunnel_nm,
                    args.d_min_nm,
                    args.g_cutoff_s,
                    args.strain_grid,
                    args.poisson_nu,
                    seed,
                    n_cb_max,
                    n_cnt_max,
                    CNT_WAVINESS,
                    args.opening_alpha_ss_nm_per_strain,
                    args.opening_alpha_sc_nm_per_strain,
                    args.opening_alpha_cc_nm_per_strain,
                )
                for seed in seeds
            ]

            print(
                f"mult={multiplier:g}  phi_CB={phi_cb_target:.5g}  "
                f"phi_CNT={phi_cnt_target:.5g}  n_cb_max={n_cb_max}  n_cnt_max={n_cnt_max}"
            )
            t0 = time.time()
            def append_one(
                realization_idx: int,
                seed: int,
                result: SimmonsGFResult,
            ) -> None:
                _append_realization(
                    real_csv,
                    spec,
                    args.population_mode,
                    multiplier,
                    realization_idx,
                    seed,
                    phi_cb_target,
                    phi_cnt_target,
                    l_rve_nm,
                    result,
                )

            results = _run_tasks(
                tasks,
                args.workers,
                seeds=seeds,
                on_result=append_one,
                progress_every=args.progress_every,
            )
            elapsed = time.time() - t0
            row = _summary_row(
                spec,
                args.population_mode,
                multiplier,
                phi_cb_target,
                phi_cnt_target,
                results,
                elapsed,
                l_rve_nm,
                args,
            )
            _write_summary(summary_csv, row)
            print(
                f"  n_ok={row['n_ok']}/{row['n_total']}  "
                f"GF_R_med={row['GF_R_median'] or 'nan'}  "
                f"GF_sigma_med={row['GF_sigma_median'] or 'nan'}  "
                f"geom={row['GF_geom_median'] or 'nan'}  elapsed={elapsed:.1f}s"
            )
            if row["failure_reasons"]:
                print(f"  failures: {row['failure_reasons']}")

    print("\nDone.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="smoke")
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument(
        "--population-mode",
        choices=("poly", "mono"),
        default="poly",
    )
    parser.add_argument(
        "--phi-multipliers",
        default=",".join(f"{v:g}" for v in DEFAULT_MULTIPLIERS),
    )
    parser.add_argument("--n-real", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hybrid-cb-wt", type=float, default=DEFAULT_HYBRID_CB_WT)
    parser.add_argument("--l-rve-nm", type=float, default=L_RVE_NM)
    parser.add_argument("--l-rve-cb-nm", type=float, default=L_RVE_CB_NM)
    parser.add_argument("--l-rve-cnt-nm", type=float, default=L_RVE_CNT_NM)
    parser.add_argument("--l-rve-hybrid-nm", type=float, default=L_RVE_HYBRID_NM)
    parser.add_argument("--tunnel-nm", type=float, default=TUNNEL_NM)
    parser.add_argument("--d-min-nm", type=float, default=D_MIN_TUNNEL_NM)
    parser.add_argument("--g-cutoff-s", type=float, default=G_CUTOFF_S)
    parser.add_argument("--poisson-nu", type=float, default=POISSON_NU)
    parser.add_argument(
        "--opening-alpha-ss-nm-per-strain",
        type=float,
        default=0.0,
        help="Extra CB-CB effective tunnel opening alpha: d_eff += alpha*epsilon.",
    )
    parser.add_argument(
        "--opening-alpha-sc-nm-per-strain",
        type=float,
        default=0.0,
        help="Extra CB-CNT effective tunnel opening alpha: d_eff += alpha*epsilon.",
    )
    parser.add_argument(
        "--opening-alpha-cc-nm-per-strain",
        type=float,
        default=0.0,
        help="Extra CNT-CNT effective tunnel opening alpha: d_eff += alpha*epsilon.",
    )
    parser.add_argument(
        "--strain-grid",
        type=_parse_strain_grid,
        default=STRAIN_GRID,
    )
    parser.add_argument("--linearity-r2-threshold", type=float, default=0.90)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress after this many completed realizations; use 0 to disable.",
    )
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
