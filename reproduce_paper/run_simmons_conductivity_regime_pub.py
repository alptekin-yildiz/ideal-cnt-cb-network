#!/usr/bin/env python3
"""Simmons-weighted ideal conductivity regime sweep for the companion manuscript.

This wrapper is the physical-weight extension after the unit-conductance
Kirchhoff validation.  It uses the selected polydisperse CNT and CB
distributions from Figs. 4--6, builds the same penetrable graph topology, and
weights each accepted edge by the local Simmons tunneling conductance.
For supplementary controls, --population-mode mono uses monodisperse CB/CNT
counterparts while keeping the same graph and solver diagnostics.

Outputs:
  data/processed/conductivity/simmons_conductivity_<tag>.csv
  data/processed/conductivity/simmons_conductivity_<tag>_realizations.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# Put the repo root on sys.path so a bare checkout runs without pip install;
# the cntcb imports below intentionally follow this bootstrap (hence E402).
ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT
DATA_DIR = PAPER / "data" / "processed"
PHIC_DIR = DATA_DIR / "percolation"
OUT_DIR = DATA_DIR / "conductivity"

sys.path.insert(0, str(ROOT))

from cntcb.engines.percolation_engine import (  # noqa: E402
    estimate_cb_n_max,
    estimate_cnt_n_max,
    solve_truncated_lognormal,
)
from cntcb.engines.simmons_conductivity_engine import (  # noqa: E402
    SimmonsConductivityResult,
    run_simmons_conductivity_realization,
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
TUNNEL_NM = 10.0
D_MIN_TUNNEL_NM = 0.34
G_CUTOFF_S = 1.0e-15
SEED_BASE = 20260610
SAFETY = 2.5

DEFAULT_SYSTEMS = ("cnt", "cb", "hybrid")
DEFAULT_MULTIPLIERS = (1.05, 1.15, 1.30, 1.50)
DEFAULT_HYBRID_CB_WT = 5.0

CNT_PHIC_CSV = PHIC_DIR / "cnt_phic_sweep_fixedstep_Lmin50.csv"
CB_PHIC_CSV = PHIC_DIR / "cb_phic_sweep_Loverd60_momentmatched_D20_500_n300.csv"
HYBRID_PHIC_CSV = PHIC_DIR / "hybrid_phic_boundary_n400_modes.csv"
MONO_UNIVERSALITY_FITS_CSV = (
    OUT_DIR / "conductivity_universality_mono_validation_n100_diag_fits.csv"
)


SUMMARY_HEADER = [
    "system",
    "system_label",
    "variable_filler",
    "phi_multiplier",
    "phi_c_ref_vol",
    "phi_variable_target_vol",
    "phi_variable_actual_mean_vol",
    "phi_variable_actual_std_vol",
    "phi_cb_target_vol",
    "phi_cb_actual_mean_vol",
    "phi_cnt_target_vol",
    "phi_cnt_actual_mean_vol",
    "sigma_s_per_m_mean",
    "sigma_s_per_m_std",
    "sigma_s_per_m_median",
    "sigma_s_per_m_p25",
    "sigma_s_per_m_p75",
    "sigma_conditional_s_per_m_mean",
    "sigma_conditional_s_per_m_std",
    "n_ok",
    "n_fail",
    "n_total",
    "n_cb_mean",
    "n_cnt_mean",
    "n_total_segs_mean",
    "n_nodes_mean",
    "n_edges_mean",
    "n_edges_ss_mean",
    "n_edges_sc_mean",
    "n_edges_cc_mean",
    "n_components_mean",
    "largest_component_fraction_mean",
    "n_spanning_components_mean",
    "low_nodes_mean",
    "high_nodes_mean",
    "n_unknown_mean",
    "matrix_nnz_mean",
    "residual_ratio_mean",
    "residual_ratio_max",
    "G_min_mean",
    "G_max_mean",
    "log10_G_range_mean",
    "log10_G_range_max",
    "n_clamped_d_min_mean",
    "frac_clamped_d_min_mean",
    "n_overlap_edges_mean",
    "frac_overlap_edges_mean",
    "n_cutoff_edges_mean",
    "frac_cutoff_edges_mean",
    "n_finite_g_mean",
    "mean_d_realized_nm",
    "std_d_realized_nm",
    "mean_L_realized_nm",
    "std_L_realized_nm",
    "mean_n_seg_per_cnt",
    "max_n_seg_per_cnt",
    "elapsed_s",
    "L_RVE_nm",
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
    "phi_multiplier",
    "realization",
    "seed",
    "success",
    "failure_reason",
    "failure_stage",
    "sigma_s_per_nm",
    "sigma_s_per_m",
    "effective_conductance_s",
    "phi_cb_target_vol",
    "phi_cb_actual_vol",
    "phi_cnt_target_vol",
    "phi_cnt_actual_vol",
    "n_cb",
    "n_cnt",
    "n_total_segs",
    "n_nodes",
    "n_edges",
    "n_edges_ss",
    "n_edges_sc",
    "n_edges_cc",
    "n_components",
    "largest_component_size",
    "largest_component_fraction",
    "n_spanning_components",
    "low_nodes",
    "high_nodes",
    "n_unknown",
    "matrix_nnz",
    "residual_ratio",
    "G_min",
    "G_max",
    "G_dynamic_range",
    "log10_G_range",
    "log10_G_p0",
    "log10_G_p1",
    "log10_G_p5",
    "log10_G_p50",
    "log10_G_p95",
    "log10_G_p99",
    "log10_G_p100",
    "n_clamped_d_min",
    "frac_clamped_d_min",
    "n_overlap_edges",
    "frac_overlap_edges",
    "n_cutoff_edges",
    "frac_cutoff_edges",
    "n_finite_g",
    "mean_d_realized_nm",
    "std_d_realized_nm",
    "mean_L_realized_nm",
    "std_L_realized_nm",
    "mean_n_seg_per_cnt",
    "max_n_seg_per_cnt",
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


def _output_paths(out_dir: Path, tag: str) -> tuple[Path, Path]:
    base = f"simmons_conductivity_{_safe_tag(tag)}"
    return (
        out_dir / f"{base}.csv",
        out_dir / f"{base}_realizations.csv",
    )


def _system_l_rve_nm(args: argparse.Namespace, system: str) -> float:
    """Return the system-specific RVE size, falling back to the global value."""

    if system == "cb" and args.l_rve_cb_nm is not None:
        return float(args.l_rve_cb_nm)
    if system == "cnt" and args.l_rve_cnt_nm is not None:
        return float(args.l_rve_cnt_nm)
    if system == "hybrid" and args.l_rve_hybrid_nm is not None:
        return float(args.l_rve_hybrid_nm)
    return float(args.l_rve_nm)


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


def _load_phi_c_hybrid(path: Path, w_cb_wt: float, mode: str = "PP") -> tuple[float, float]:
    row = _find_row(
        path,
        lambda r: r["mode"] == mode and abs(float(r["w_cb_wt_pct"]) - w_cb_wt) < 1e-9,
    )
    return float(row["phi_cb_vol"]), float(row["phi_cnt_c_median_vol"])


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
            "cb_d_min_nm": CB_D_MIN_NM,
            "cb_d_max_nm": CB_D_MAX_NM,
            "cnt_mu_L_nm": cnt_mu,
            "cnt_sigma_L_nm": cnt_sigma,
            "cnt_L_min_nm": CNT_L_MIN_NM,
            "cnt_L_max_nm": CNT_L_MAX_NM,
            "cnt_waviness": CNT_WAVINESS,
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
            "cb_d_min_nm": CB_D_MIN_NM,
            "cb_d_max_nm": CB_D_MAX_NM,
            "cnt_mu_L_nm": cnt_mu,
            "cnt_sigma_L_nm": cnt_sigma,
            "cnt_L_min_nm": CNT_L_MIN_NM,
            "cnt_L_max_nm": CNT_L_MAX_NM,
            "cnt_waviness": CNT_WAVINESS,
        }
    if system == "hybrid":
        phi_cb, phi_cnt_c = _load_phi_c_hybrid(
            HYBRID_PHIC_CSV,
            hybrid_cb_wt,
            mode=hybrid_mode,
        )
        return {
            "system": "hybrid",
            "system_label": f"hybrid {hybrid_mode}, fixed CB={hybrid_cb_wt:g} wt%",
            "variable_filler": "CNT",
            "phi_cb_fixed": phi_cb,
            "phi_cnt_fixed": 0.0,
            "phi_c_ref": phi_cnt_c,
            "cb_mu_d_nm": cb_mu,
            "cb_sigma_d_nm": cb_sigma,
            "cb_d_min_nm": CB_D_MIN_NM,
            "cb_d_max_nm": CB_D_MAX_NM,
            "cnt_mu_L_nm": cnt_mu,
            "cnt_sigma_L_nm": cnt_sigma,
            "cnt_L_min_nm": CNT_L_MIN_NM,
            "cnt_L_max_nm": CNT_L_MAX_NM,
            "cnt_waviness": CNT_WAVINESS,
        }
    raise ValueError("Unknown system {!r}; use cb,cnt,hybrid.".format(system))


def _run_one(args: tuple) -> SimmonsConductivityResult:
    (
        cb_model,
        cnt_model,
        phi_cb_target,
        phi_cnt_target,
        l_rve_nm,
        tunnel_nm,
        d_min_nm,
        g_cutoff_s,
        seed,
        n_cb_max,
        n_cnt_max,
        waviness,
    ) = args
    return run_simmons_conductivity_realization(
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
        seed=seed,
        n_cb_max=n_cb_max,
        n_cnt_max=n_cnt_max,
    )


def _run_tasks(
    tasks: list[tuple],
    workers: int,
    progress_every: int,
) -> list[SimmonsConductivityResult]:
    results: list[SimmonsConductivityResult | None] = [None] * len(tasks)
    t0 = time.time()
    n_done = 0

    def record(idx: int, result: SimmonsConductivityResult) -> None:
        nonlocal n_done
        results[idx - 1] = result
        n_done += 1
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


def _finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def _finite_max(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if len(arr) else float("nan")


def _write_realizations(
    path: Path,
    spec: dict[str, float | str],
    multiplier: float,
    seeds: list[int],
    phi_cb_target: float,
    phi_cnt_target: float,
    l_rve_nm: float,
    results: list[SimmonsConductivityResult],
) -> None:
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        for idx, (seed, res) in enumerate(zip(seeds, results), start=1):
            writer.writerow(
                [
                    spec["system"],
                    spec["system_label"],
                    spec["variable_filler"],
                    f"{multiplier:.8g}",
                    idx,
                    seed,
                    res.success,
                    res.failure_reason,
                    res.failure_stage,
                    f"{res.sigma_s_per_nm:.8g}",
                    f"{res.sigma_s_per_m:.8g}",
                    f"{res.effective_conductance_s:.8g}",
                    f"{phi_cb_target:.8g}",
                    f"{res.phi_cb_actual_vol:.8g}",
                    f"{phi_cnt_target:.8g}",
                    f"{res.phi_cnt_actual_vol:.8g}",
                    res.n_cb,
                    res.n_cnt,
                    res.n_total_segs,
                    res.n_nodes,
                    res.n_edges,
                    res.n_edges_ss,
                    res.n_edges_sc,
                    res.n_edges_cc,
                    res.n_components,
                    res.largest_component_size,
                    f"{res.largest_component_fraction:.8g}",
                    res.n_spanning_components,
                    res.low_nodes,
                    res.high_nodes,
                    res.n_unknown,
                    res.matrix_nnz,
                    f"{res.residual_ratio:.8g}",
                    f"{res.g_min:.8g}",
                    f"{res.g_max:.8g}",
                    f"{res.g_dynamic_range:.8g}",
                    f"{res.log10_g_range:.8g}",
                    f"{res.log10_g_p0:.8g}",
                    f"{res.log10_g_p1:.8g}",
                    f"{res.log10_g_p5:.8g}",
                    f"{res.log10_g_p50:.8g}",
                    f"{res.log10_g_p95:.8g}",
                    f"{res.log10_g_p99:.8g}",
                    f"{res.log10_g_p100:.8g}",
                    res.n_clamped_d_min,
                    f"{res.frac_clamped_d_min:.8g}",
                    res.n_overlap_edges,
                    f"{res.frac_overlap_edges:.8g}",
                    res.n_cutoff_edges,
                    f"{res.frac_cutoff_edges:.8g}",
                    res.n_finite_g,
                    f"{res.mean_d_realized_nm:.8g}",
                    f"{res.std_d_realized_nm:.8g}",
                    f"{res.mean_L_realized_nm:.8g}",
                    f"{res.std_L_realized_nm:.8g}",
                    f"{res.mean_n_seg_per_cnt:.8g}",
                    res.max_n_seg_per_cnt,
                    f"{res.elapsed_s:.3f}",
                    f"{l_rve_nm:.8g}",
                ]
            )


def _summary_row(
    spec: dict[str, float | str],
    multiplier: float,
    phi_cb_target: float,
    phi_cnt_target: float,
    results: list[SimmonsConductivityResult],
    elapsed_s: float,
    l_rve_nm: float,
    tunnel_nm: float,
    d_min_nm: float,
    g_cutoff_s: float,
) -> dict[str, str]:
    variable = str(spec["variable_filler"])
    phi_var_actual = [
        res.phi_cb_actual_vol if variable == "CB" else res.phi_cnt_actual_vol
        for res in results
    ]
    sigmas = [res.sigma_s_per_m for res in results]
    sigmas_ok = [res.sigma_s_per_m for res in results if res.success]

    return {
        "system": str(spec["system"]),
        "system_label": str(spec["system_label"]),
        "variable_filler": variable,
        "phi_multiplier": f"{multiplier:.8g}",
        "phi_c_ref_vol": f"{float(spec['phi_c_ref']):.8g}",
        "phi_variable_target_vol": f"{(phi_cb_target if variable == 'CB' else phi_cnt_target):.8g}",
        "phi_variable_actual_mean_vol": f"{float(np.mean(phi_var_actual)):.8g}",
        "phi_variable_actual_std_vol": f"{float(np.std(phi_var_actual, ddof=1)) if len(phi_var_actual) > 1 else 0.0:.8g}",
        "phi_cb_target_vol": f"{phi_cb_target:.8g}",
        "phi_cb_actual_mean_vol": f"{float(np.mean([r.phi_cb_actual_vol for r in results])):.8g}",
        "phi_cnt_target_vol": f"{phi_cnt_target:.8g}",
        "phi_cnt_actual_mean_vol": f"{float(np.mean([r.phi_cnt_actual_vol for r in results])):.8g}",
        "sigma_s_per_m_mean": f"{float(np.mean(sigmas)):.8g}",
        "sigma_s_per_m_std": f"{float(np.std(sigmas, ddof=1)) if len(sigmas) > 1 else 0.0:.8g}",
        "sigma_s_per_m_median": f"{float(np.median(sigmas)):.8g}",
        "sigma_s_per_m_p25": f"{float(np.percentile(sigmas, 25)):.8g}",
        "sigma_s_per_m_p75": f"{float(np.percentile(sigmas, 75)):.8g}",
        "sigma_conditional_s_per_m_mean": f"{float(np.mean(sigmas_ok)) if sigmas_ok else 0.0:.8g}",
        "sigma_conditional_s_per_m_std": f"{float(np.std(sigmas_ok, ddof=1)) if len(sigmas_ok) > 1 else 0.0:.8g}",
        "n_ok": str(sum(res.success for res in results)),
        "n_fail": str(sum(not res.success for res in results)),
        "n_total": str(len(results)),
        "n_cb_mean": f"{float(np.mean([r.n_cb for r in results])):.8g}",
        "n_cnt_mean": f"{float(np.mean([r.n_cnt for r in results])):.8g}",
        "n_total_segs_mean": f"{float(np.mean([r.n_total_segs for r in results])):.8g}",
        "n_nodes_mean": f"{float(np.mean([r.n_nodes for r in results])):.8g}",
        "n_edges_mean": f"{float(np.mean([r.n_edges for r in results])):.8g}",
        "n_edges_ss_mean": f"{float(np.mean([r.n_edges_ss for r in results])):.8g}",
        "n_edges_sc_mean": f"{float(np.mean([r.n_edges_sc for r in results])):.8g}",
        "n_edges_cc_mean": f"{float(np.mean([r.n_edges_cc for r in results])):.8g}",
        "n_components_mean": f"{float(np.mean([r.n_components for r in results])):.8g}",
        "largest_component_fraction_mean": f"{float(np.mean([r.largest_component_fraction for r in results])):.8g}",
        "n_spanning_components_mean": f"{float(np.mean([r.n_spanning_components for r in results])):.8g}",
        "low_nodes_mean": f"{float(np.mean([r.low_nodes for r in results])):.8g}",
        "high_nodes_mean": f"{float(np.mean([r.high_nodes for r in results])):.8g}",
        "n_unknown_mean": f"{float(np.mean([r.n_unknown for r in results])):.8g}",
        "matrix_nnz_mean": f"{float(np.mean([r.matrix_nnz for r in results])):.8g}",
        "residual_ratio_mean": f"{_finite_mean([r.residual_ratio for r in results]):.8g}",
        "residual_ratio_max": f"{_finite_max([r.residual_ratio for r in results]):.8g}",
        "G_min_mean": f"{_finite_mean([r.g_min for r in results]):.8g}",
        "G_max_mean": f"{_finite_mean([r.g_max for r in results]):.8g}",
        "log10_G_range_mean": f"{_finite_mean([r.log10_g_range for r in results]):.8g}",
        "log10_G_range_max": f"{_finite_max([r.log10_g_range for r in results]):.8g}",
        "n_clamped_d_min_mean": f"{float(np.mean([r.n_clamped_d_min for r in results])):.8g}",
        "frac_clamped_d_min_mean": f"{_finite_mean([r.frac_clamped_d_min for r in results]):.8g}",
        "n_overlap_edges_mean": f"{float(np.mean([r.n_overlap_edges for r in results])):.8g}",
        "frac_overlap_edges_mean": f"{_finite_mean([r.frac_overlap_edges for r in results]):.8g}",
        "n_cutoff_edges_mean": f"{float(np.mean([r.n_cutoff_edges for r in results])):.8g}",
        "frac_cutoff_edges_mean": f"{_finite_mean([r.frac_cutoff_edges for r in results]):.8g}",
        "n_finite_g_mean": f"{float(np.mean([r.n_finite_g for r in results])):.8g}",
        "mean_d_realized_nm": f"{_finite_mean([r.mean_d_realized_nm for r in results]):.8g}",
        "std_d_realized_nm": f"{_finite_mean([r.std_d_realized_nm for r in results]):.8g}",
        "mean_L_realized_nm": f"{_finite_mean([r.mean_L_realized_nm for r in results]):.8g}",
        "std_L_realized_nm": f"{_finite_mean([r.std_L_realized_nm for r in results]):.8g}",
        "mean_n_seg_per_cnt": f"{_finite_mean([r.mean_n_seg_per_cnt for r in results]):.8g}",
        "max_n_seg_per_cnt": str(max(r.max_n_seg_per_cnt for r in results)),
        "elapsed_s": f"{elapsed_s:.3f}",
        "L_RVE_nm": f"{l_rve_nm:.8g}",
        "tunnel_nm": f"{tunnel_nm:.8g}",
        "d_min_tunnel_nm": f"{d_min_nm:.8g}",
        "G_cutoff_S": f"{g_cutoff_s:.8g}",
        "CB_mu_d_nm": f"{float(spec['cb_mu_d_nm']):.8g}",
        "CB_sigma_d_nm": f"{float(spec['cb_sigma_d_nm']):.8g}",
        "CB_D_min_nm": f"{float(spec['cb_d_min_nm']):.8g}",
        "CB_D_max_nm": f"{float(spec['cb_d_max_nm']):.8g}",
        "CNT_mu_L_nm": f"{float(spec['cnt_mu_L_nm']):.8g}",
        "CNT_sigma_L_nm": f"{float(spec['cnt_sigma_L_nm']):.8g}",
        "CNT_L_min_nm": f"{float(spec['cnt_L_min_nm']):.8g}",
        "CNT_L_max_nm": f"{float(spec['cnt_L_max_nm']):.8g}",
        "CNT_diam_nm": f"{CNT_DIAM_NM:.8g}",
        "CNT_waviness": f"{float(spec['cnt_waviness']):.8g}",
        "seg_len_unit_nm": f"{SEG_LEN_UNIT_NM:.8g}",
    }


def _write_summary(path: Path, row: dict[str, str]) -> None:
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=SUMMARY_HEADER).writerow(row)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser().resolve()
    summary_csv, real_csv = _output_paths(out_dir, args.tag)
    _prepare_output(summary_csv, SUMMARY_HEADER, args.overwrite)
    _prepare_output(real_csv, REAL_HEADER, args.overwrite)

    systems = tuple(s.lower() for s in _parse_csv_list(args.systems, str))
    multipliers = tuple(float(v) for v in _parse_csv_list(args.phi_multipliers, float))

    print("=" * 78)
    print("Simmons-weighted ideal conductivity regime sweep")
    print(f"  tag={args.tag}  N={args.n_real}  workers={args.workers}")
    print(f"  population_mode={args.population_mode}")
    print(f"  systems={','.join(systems)}  multipliers={','.join(f'{m:g}' for m in multipliers)}")
    print(
        f"  L_RVE={args.l_rve_nm:g} nm  tunnel={args.tunnel_nm:g} nm  "
        f"d_min={args.d_min_nm:g} nm  G_cut={args.g_cutoff_s:g} S"
    )
    if (
        args.l_rve_cb_nm is not None
        or args.l_rve_cnt_nm is not None
        or args.l_rve_hybrid_nm is not None
    ):
        print(
            "  system L_RVE overrides: "
            f"CB={args.l_rve_cb_nm or args.l_rve_nm:g} nm, "
            f"CNT={args.l_rve_cnt_nm or args.l_rve_nm:g} nm, "
            f"hybrid={args.l_rve_hybrid_nm or args.l_rve_nm:g} nm"
        )
    print(f"  summary: {summary_csv}")
    print(f"  realizations: {real_csv}")
    print("=" * 78)

    for sys_idx, system in enumerate(systems):
        spec = _system_spec(system, args.hybrid_cb_wt, args.population_mode)
        l_rve_nm = _system_l_rve_nm(args, system)
        cb_model = solve_truncated_lognormal(
            float(spec["cb_mu_d_nm"]),
            float(spec["cb_sigma_d_nm"]),
            float(spec["cb_d_min_nm"]),
            float(spec["cb_d_max_nm"]),
        )
        cnt_model = solve_truncated_lognormal(
            float(spec["cnt_mu_L_nm"]),
            float(spec["cnt_sigma_L_nm"]),
            float(spec["cnt_L_min_nm"]),
            float(spec["cnt_L_max_nm"]),
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
                    seed,
                    n_cb_max,
                    n_cnt_max,
                    float(spec["cnt_waviness"]),
                )
                for seed in seeds
            ]

            print(
                f"mult={multiplier:g}  phi_CB={phi_cb_target:.5g}  "
                f"phi_CNT={phi_cnt_target:.5g}  n_cb_max={n_cb_max}  n_cnt_max={n_cnt_max}"
            )
            t0 = time.time()
            results = _run_tasks(tasks, args.workers, args.progress_every)
            elapsed = time.time() - t0
            row = _summary_row(
                spec,
                multiplier,
                phi_cb_target,
                phi_cnt_target,
                results,
                elapsed,
                l_rve_nm,
                args.tunnel_nm,
                args.d_min_nm,
                args.g_cutoff_s,
            )
            _write_realizations(
                real_csv,
                spec,
                multiplier,
                seeds,
                phi_cb_target,
                phi_cnt_target,
                l_rve_nm,
                results,
            )
            _write_summary(summary_csv, row)
            print(
                f"  n_ok={row['n_ok']}/{row['n_total']}  "
                f"sigma={float(row['sigma_s_per_m_mean']):.3e} S/m  "
                f"log10G_range={float(row['log10_G_range_mean']):.2f}  "
                f"elapsed={elapsed:.1f}s"
            )

    print("\nDone.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="smoke")
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument(
        "--population-mode",
        choices=("poly", "mono"),
        default="poly",
        help="Use selected polydisperse populations or monodisperse support controls.",
    )
    parser.add_argument(
        "--phi-multipliers",
        default=",".join(f"{v:g}" for v in DEFAULT_MULTIPLIERS),
    )
    parser.add_argument("--n-real", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hybrid-cb-wt", type=float, default=DEFAULT_HYBRID_CB_WT)
    parser.add_argument("--l-rve-nm", type=float, default=L_RVE_NM)
    parser.add_argument("--l-rve-cb-nm", type=float)
    parser.add_argument("--l-rve-cnt-nm", type=float)
    parser.add_argument("--l-rve-hybrid-nm", type=float)
    parser.add_argument("--tunnel-nm", type=float, default=TUNNEL_NM)
    parser.add_argument("--d-min-nm", type=float, default=D_MIN_TUNNEL_NM)
    parser.add_argument("--g-cutoff-s", type=float, default=G_CUTOFF_S)
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
