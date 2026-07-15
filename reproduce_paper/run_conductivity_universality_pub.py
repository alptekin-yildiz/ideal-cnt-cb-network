#!/usr/bin/env python3
"""Step-network conductivity universality sweep for the paper.

This wrapper validates the Kirchhoff solver before the absolute tunneling
conductivity stage.  Every connected junction has unit conductance
(`G_ij = 1`); the reported conductivity is therefore a model-unit quantity.

Default systems:

* cb     : selected polydisperse CB population from Fig. 7;
* cnt    : selected polydisperse CNT population from Fig. 6;
* hybrid : selected PP hybrid population from Fig. 8 at fixed CB loading.

Outputs:
  data/processed/conductivity/conductivity_universality_<tag>.csv
  data/processed/conductivity/conductivity_universality_<tag>_realizations.csv
  data/processed/conductivity/conductivity_universality_<tag>_fits.csv
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

from cntcb.engines.conductivity_engine import (  # noqa: E402
    StepConductivityResult,
    run_step_conductivity_realization,
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

CNT_MU_L_NM = 500.0
CNT_SIGMA_L_NM = 300.0
CNT_L_MIN_NM = 50.0
CNT_L_MAX_NM = 1500.0
CNT_DIAM_NM = 10.0
CNT_WAVINESS = 0.7
SEG_LEN_UNIT_NM = 50.0

L_RVE_NM = 4000.0
TUNNEL_NM = 10.0
SEED_BASE = 20260609
SAFETY = 2.5

DEFAULT_SYSTEMS = ("cb", "cnt", "hybrid")
DEFAULT_MULTIPLIERS = (0.85, 0.95, 1.05, 1.15, 1.30, 1.50)
DEFAULT_HYBRID_CB_WT = 5.0
NU = 0.8774

CNT_PHIC_CSV = PHIC_DIR / "cnt_phic_sweep_fixedstep_Lmin50.csv"
CB_PHIC_CSV = PHIC_DIR / "cb_phic_sweep_Loverd60_momentmatched_D20_500_n300.csv"
HYBRID_PHIC_CSV = PHIC_DIR / "hybrid_phic_boundary_n400_modes.csv"
MONO_CB_FSS_CSV = PHIC_DIR / "fss_monodisperse_cb_d150.csv"
MONO_CNT_FSS_CSV = PHIC_DIR / "fss_monodisperse_cnt_L500_w_sweep.csv"

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
    "sigma_uncond_mean",
    "sigma_uncond_std",
    "sigma_conditional_mean",
    "sigma_conditional_std",
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
    "largest_component_size_mean",
    "largest_component_fraction_mean",
    "n_spanning_components_mean",
    "low_nodes_mean",
    "high_nodes_mean",
    "n_unknown_mean",
    "matrix_nnz_mean",
    "residual_ratio_mean",
    "residual_ratio_max",
    "mean_d_realized_nm",
    "std_d_realized_nm",
    "mean_L_realized_nm",
    "std_L_realized_nm",
    "mean_n_seg_per_cnt",
    "max_n_seg_per_cnt",
    "elapsed_s",
    "L_RVE_nm",
    "tunnel_nm",
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
    "sigma_uncond",
    "effective_conductance",
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
    "mean_d_realized_nm",
    "std_d_realized_nm",
    "mean_L_realized_nm",
    "std_L_realized_nm",
    "mean_n_seg_per_cnt",
    "max_n_seg_per_cnt",
    "elapsed_s",
]

FIT_HEADER = [
    "system",
    "system_label",
    "variable_filler",
    "fit_mode",
    "sigma_column",
    "min_success_fraction",
    "phi_c_ref_vol",
    "n_fit",
    "t_fit",
    "intercept_log_C",
    "r2_loglog",
    "phi_min_fit_vol",
    "phi_max_fit_vol",
]


def _safe_tag(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    if not safe:
        raise ValueError("--tag must contain at least one safe character.")
    return safe


def _parse_csv_list(text: str, cast=str) -> tuple:
    vals = []
    for chunk in text.split(","):
        item = chunk.strip()
        if item:
            vals.append(cast(item))
    if not vals:
        raise ValueError("List argument is empty.")
    return tuple(vals)


def _output_paths(out_dir: Path, tag: str) -> tuple[Path, Path, Path]:
    base = f"conductivity_universality_{_safe_tag(tag)}"
    return (
        out_dir / f"{base}.csv",
        out_dir / f"{base}_realizations.csv",
        out_dir / f"{base}_fits.csv",
    )


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


def _load_phi_c_hybrid(path: Path, w_cb_wt: float) -> tuple[float, float]:
    row = _find_row(
        path,
        lambda r: r["mode"] == "PP" and abs(float(r["w_cb_wt_pct"]) - w_cb_wt) < 1e-9,
    )
    return float(row["phi_cb_vol"]), float(row["phi_cnt_c_median_vol"])


def _fss_phi_inf(rows: list[dict[str, str]]) -> float:
    L = np.asarray([float(row["L_nm"]) for row in rows], dtype=float)
    phi = np.asarray([float(row["phi_c_mean"]) for row in rows], dtype=float)
    std = np.asarray([float(row["phi_c_std"]) for row in rows], dtype=float)
    ok = np.isfinite(L) & np.isfinite(phi) & np.isfinite(std) & (std > 0)
    if np.count_nonzero(ok) < 2:
        raise ValueError("Need at least two finite FSS points to extrapolate phi_inf.")
    x = L[ok] ** (-1.0 / NU)
    weights = 1.0 / std[ok] ** 2
    A = np.array([x, np.ones_like(x)]).T
    W = np.diag(weights)
    coeffs = np.linalg.lstsq(W @ A, W @ phi[ok], rcond=None)[0]
    return float(coeffs[1])


def _load_phi_c_cb_mono(path: Path) -> float:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return _fss_phi_inf(rows)


def _load_phi_c_cnt_mono(path: Path, waviness: float) -> float:
    with path.open(newline="") as f:
        rows = [
            row for row in csv.DictReader(f)
            if abs(float(row["w"]) - waviness) < 1e-9
        ]
    if not rows:
        raise ValueError(f"No monodisperse CNT FSS rows found for w={waviness}.")
    return _fss_phi_inf(rows)


def _system_spec(system: str, hybrid_cb_wt: float) -> dict[str, float | str]:
    system = system.lower()
    if system == "cb":
        return {
            "system": "cb",
            "system_label": "pure CB, selected polydisperse",
            "variable_filler": "CB",
            "phi_cb_fixed": 0.0,
            "phi_cnt_fixed": 0.0,
            "phi_c_ref": _load_phi_c_cb(CB_PHIC_CSV),
            "cb_mu_d_nm": CB_MU_D_NM,
            "cb_sigma_d_nm": CB_SIGMA_D_NM,
            "cb_d_min_nm": CB_D_MIN_NM,
            "cb_d_max_nm": CB_D_MAX_NM,
            "cnt_mu_L_nm": CNT_MU_L_NM,
            "cnt_sigma_L_nm": CNT_SIGMA_L_NM,
            "cnt_L_min_nm": CNT_L_MIN_NM,
            "cnt_L_max_nm": CNT_L_MAX_NM,
            "cnt_waviness": CNT_WAVINESS,
        }
    if system == "cnt":
        return {
            "system": "cnt",
            "system_label": "pure CNT, selected polydisperse",
            "variable_filler": "CNT",
            "phi_cb_fixed": 0.0,
            "phi_cnt_fixed": 0.0,
            "phi_c_ref": _load_phi_c_cnt(CNT_PHIC_CSV),
            "cb_mu_d_nm": CB_MU_D_NM,
            "cb_sigma_d_nm": CB_SIGMA_D_NM,
            "cb_d_min_nm": CB_D_MIN_NM,
            "cb_d_max_nm": CB_D_MAX_NM,
            "cnt_mu_L_nm": CNT_MU_L_NM,
            "cnt_sigma_L_nm": CNT_SIGMA_L_NM,
            "cnt_L_min_nm": CNT_L_MIN_NM,
            "cnt_L_max_nm": CNT_L_MAX_NM,
            "cnt_waviness": CNT_WAVINESS,
        }
    if system == "hybrid":
        phi_cb, phi_c_cnt = _load_phi_c_hybrid(HYBRID_PHIC_CSV, hybrid_cb_wt)
        return {
            "system": "hybrid",
            "system_label": f"hybrid PP, fixed CB={hybrid_cb_wt:g} wt%",
            "variable_filler": "CNT",
            "phi_cb_fixed": phi_cb,
            "phi_cnt_fixed": 0.0,
            "phi_c_ref": phi_c_cnt,
            "cb_mu_d_nm": CB_MU_D_NM,
            "cb_sigma_d_nm": CB_SIGMA_D_NM,
            "cb_d_min_nm": CB_D_MIN_NM,
            "cb_d_max_nm": CB_D_MAX_NM,
            "cnt_mu_L_nm": CNT_MU_L_NM,
            "cnt_sigma_L_nm": CNT_SIGMA_L_NM,
            "cnt_L_min_nm": CNT_L_MIN_NM,
            "cnt_L_max_nm": CNT_L_MAX_NM,
            "cnt_waviness": CNT_WAVINESS,
        }
    if system == "cb_mono":
        return {
            "system": "cb_mono",
            "system_label": "pure CB, monodisperse d=150 nm",
            "variable_filler": "CB",
            "phi_cb_fixed": 0.0,
            "phi_cnt_fixed": 0.0,
            "phi_c_ref": _load_phi_c_cb_mono(MONO_CB_FSS_CSV),
            "cb_mu_d_nm": 150.0,
            "cb_sigma_d_nm": 0.0,
            "cb_d_min_nm": CB_D_MIN_NM,
            "cb_d_max_nm": CB_D_MAX_NM,
            "cnt_mu_L_nm": CNT_MU_L_NM,
            "cnt_sigma_L_nm": CNT_SIGMA_L_NM,
            "cnt_L_min_nm": CNT_L_MIN_NM,
            "cnt_L_max_nm": CNT_L_MAX_NM,
            "cnt_waviness": CNT_WAVINESS,
        }
    mono_cnt_w = {
        "cnt_mono_w1": 1.0,
        "cnt_mono_w07": 0.7,
        "cnt_mono_w03": 0.3,
    }
    if system in mono_cnt_w:
        waviness = mono_cnt_w[system]
        label_w = f"{waviness:.1f}"
        return {
            "system": system,
            "system_label": f"pure CNT, monodisperse L=500 nm, w={label_w}",
            "variable_filler": "CNT",
            "phi_cb_fixed": 0.0,
            "phi_cnt_fixed": 0.0,
            "phi_c_ref": _load_phi_c_cnt_mono(MONO_CNT_FSS_CSV, waviness),
            "cb_mu_d_nm": CB_MU_D_NM,
            "cb_sigma_d_nm": CB_SIGMA_D_NM,
            "cb_d_min_nm": CB_D_MIN_NM,
            "cb_d_max_nm": CB_D_MAX_NM,
            "cnt_mu_L_nm": 500.0,
            "cnt_sigma_L_nm": 0.0,
            "cnt_L_min_nm": CNT_L_MIN_NM,
            "cnt_L_max_nm": CNT_L_MAX_NM,
            "cnt_waviness": waviness,
        }
    raise ValueError(
        "Unknown system {!r}; use cb,cnt,hybrid,cb_mono,cnt_mono_w1,"
        "cnt_mono_w07,cnt_mono_w03.".format(system)
    )


def _run_one(args: tuple) -> StepConductivityResult:
    (
        cb_model,
        cnt_model,
        phi_cb_target,
        phi_cnt_target,
        l_rve_nm,
        tunnel_nm,
        seed,
        n_cb_max,
        n_cnt_max,
        waviness,
    ) = args
    return run_step_conductivity_realization(
        cb_model=cb_model,
        cnt_model=cnt_model,
        phi_cb_target_vol=phi_cb_target,
        phi_cnt_target_vol=phi_cnt_target,
        l_rve_nm=l_rve_nm,
        cnt_diam_nm=CNT_DIAM_NM,
        waviness=waviness,
        tunnel_nm=tunnel_nm,
        seg_len_unit_nm=SEG_LEN_UNIT_NM,
        seed=seed,
        n_cb_max=n_cb_max,
        n_cnt_max=n_cnt_max,
    )


def _run_tasks(
    tasks: list[tuple],
    workers: int,
    progress_every: int,
) -> list[StepConductivityResult]:
    results: list[StepConductivityResult | None] = [None] * len(tasks)
    t0 = time.time()
    n_done = 0

    def record(idx: int, result: StepConductivityResult) -> None:
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
    results: list[StepConductivityResult],
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
                    f"{res.sigma_uncond:.8g}",
                    f"{res.effective_conductance:.8g}",
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
                    f"{res.mean_d_realized_nm:.8g}",
                    f"{res.std_d_realized_nm:.8g}",
                    f"{res.mean_L_realized_nm:.8g}",
                    f"{res.std_L_realized_nm:.8g}",
                    f"{res.mean_n_seg_per_cnt:.8g}",
                    res.max_n_seg_per_cnt,
                    f"{res.elapsed_s:.3f}",
                ]
            )


def _summary_row(
    spec: dict[str, float | str],
    multiplier: float,
    phi_cb_target: float,
    phi_cnt_target: float,
    results: list[StepConductivityResult],
    elapsed_s: float,
    l_rve_nm: float,
    tunnel_nm: float,
) -> dict[str, str]:
    variable = str(spec["variable_filler"])
    phi_var_actual = [
        res.phi_cb_actual_vol if variable == "CB" else res.phi_cnt_actual_vol
        for res in results
    ]
    sigmas = [res.sigma_uncond for res in results]
    sigmas_ok = [res.sigma_uncond for res in results if res.success]

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
        "sigma_uncond_mean": f"{float(np.mean(sigmas)):.8g}",
        "sigma_uncond_std": f"{float(np.std(sigmas, ddof=1)) if len(sigmas) > 1 else 0.0:.8g}",
        "sigma_conditional_mean": f"{float(np.mean(sigmas_ok)) if sigmas_ok else 0.0:.8g}",
        "sigma_conditional_std": f"{float(np.std(sigmas_ok, ddof=1)) if len(sigmas_ok) > 1 else 0.0:.8g}",
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
        "largest_component_size_mean": f"{float(np.mean([r.largest_component_size for r in results])):.8g}",
        "largest_component_fraction_mean": f"{float(np.mean([r.largest_component_fraction for r in results])):.8g}",
        "n_spanning_components_mean": f"{float(np.mean([r.n_spanning_components for r in results])):.8g}",
        "low_nodes_mean": f"{float(np.mean([r.low_nodes for r in results])):.8g}",
        "high_nodes_mean": f"{float(np.mean([r.high_nodes for r in results])):.8g}",
        "n_unknown_mean": f"{float(np.mean([r.n_unknown for r in results])):.8g}",
        "matrix_nnz_mean": f"{float(np.mean([r.matrix_nnz for r in results])):.8g}",
        "residual_ratio_mean": f"{_finite_mean([r.residual_ratio for r in results]):.8g}",
        "residual_ratio_max": f"{_finite_max([r.residual_ratio for r in results]):.8g}",
        "mean_d_realized_nm": f"{_finite_mean([r.mean_d_realized_nm for r in results]):.8g}",
        "std_d_realized_nm": f"{_finite_mean([r.std_d_realized_nm for r in results]):.8g}",
        "mean_L_realized_nm": f"{_finite_mean([r.mean_L_realized_nm for r in results]):.8g}",
        "std_L_realized_nm": f"{_finite_mean([r.std_L_realized_nm for r in results]):.8g}",
        "mean_n_seg_per_cnt": f"{_finite_mean([r.mean_n_seg_per_cnt for r in results]):.8g}",
        "max_n_seg_per_cnt": str(max(r.max_n_seg_per_cnt for r in results)),
        "elapsed_s": f"{elapsed_s:.3f}",
        "L_RVE_nm": f"{l_rve_nm:.8g}",
        "tunnel_nm": f"{tunnel_nm:.8g}",
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


def _fit_power_laws(summary_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    fits = []
    systems = sorted(set(row["system"] for row in summary_rows))
    fit_specs = (
        ("all_positive_mean", "sigma_uncond_mean", 0.0),
        ("ok_ge_0p9_mean", "sigma_uncond_mean", 0.9),
        ("ok_ge_0p9_conditional", "sigma_conditional_mean", 0.9),
        ("full_perc_mean", "sigma_uncond_mean", 1.0),
        ("full_perc_conditional", "sigma_conditional_mean", 1.0),
    )
    for system in systems:
        rows = [row for row in summary_rows if row["system"] == system]
        phi_c = float(rows[0]["phi_c_ref_vol"])
        for fit_mode, sigma_column, min_success_fraction in fit_specs:
            fit_rows = []
            for row in rows:
                n_total = int(row["n_total"])
                success_fraction = int(row["n_ok"]) / n_total if n_total else 0.0
                phi = float(row["phi_variable_actual_mean_vol"])
                sigma = float(row[sigma_column])
                if (
                    success_fraction >= min_success_fraction
                    and phi > phi_c
                    and sigma > 0.0
                    and np.isfinite(phi)
                    and np.isfinite(sigma)
                ):
                    fit_rows.append(row)
            if len(fit_rows) < 2:
                fits.append(
                    _empty_fit(
                        rows[0],
                        fit_mode,
                        sigma_column,
                        min_success_fraction,
                        phi_c,
                        len(fit_rows),
                    )
                )
                continue
            x = np.asarray(
                [float(row["phi_variable_actual_mean_vol"]) - phi_c for row in fit_rows]
            )
            y = np.asarray([float(row[sigma_column]) for row in fit_rows])
            lx = np.log(x)
            ly = np.log(y)
            slope, intercept = np.polyfit(lx, ly, 1)
            pred = slope * lx + intercept
            ss_res = float(np.sum((ly - pred) ** 2))
            ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
            fits.append(
                {
                    "system": rows[0]["system"],
                    "system_label": rows[0]["system_label"],
                    "variable_filler": rows[0]["variable_filler"],
                    "fit_mode": fit_mode,
                    "sigma_column": sigma_column,
                    "min_success_fraction": f"{min_success_fraction:.8g}",
                    "phi_c_ref_vol": f"{phi_c:.8g}",
                    "n_fit": str(len(fit_rows)),
                    "t_fit": f"{float(slope):.8g}",
                    "intercept_log_C": f"{float(intercept):.8g}",
                    "r2_loglog": f"{r2:.8g}",
                    "phi_min_fit_vol": f"{min(float(row['phi_variable_actual_mean_vol']) for row in fit_rows):.8g}",
                    "phi_max_fit_vol": f"{max(float(row['phi_variable_actual_mean_vol']) for row in fit_rows):.8g}",
                }
            )
    return fits


def _empty_fit(
    row: dict[str, str],
    fit_mode: str,
    sigma_column: str,
    min_success_fraction: float,
    phi_c: float,
    n_fit: int,
) -> dict[str, str]:
    return {
        "system": row["system"],
        "system_label": row["system_label"],
        "variable_filler": row["variable_filler"],
        "fit_mode": fit_mode,
        "sigma_column": sigma_column,
        "min_success_fraction": f"{min_success_fraction:.8g}",
        "phi_c_ref_vol": f"{phi_c:.8g}",
        "n_fit": str(n_fit),
        "t_fit": "",
        "intercept_log_C": "",
        "r2_loglog": "",
        "phi_min_fit_vol": "",
        "phi_max_fit_vol": "",
    }


def _write_fits(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIT_HEADER)
        for row in rows:
            writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser().resolve()
    summary_csv, real_csv, fit_csv = _output_paths(out_dir, args.tag)
    _prepare_output(summary_csv, SUMMARY_HEADER, args.overwrite)
    _prepare_output(real_csv, REAL_HEADER, args.overwrite)
    _prepare_output(fit_csv, FIT_HEADER, args.overwrite)

    systems = tuple(s.lower() for s in _parse_csv_list(args.systems, str))
    multipliers = tuple(float(v) for v in _parse_csv_list(args.phi_multipliers, float))

    print("=" * 78)
    print("Step-network conductivity universality sweep")
    print(f"  tag={args.tag}  N={args.n_real}  workers={args.workers}")
    print(f"  systems={','.join(systems)}  multipliers={','.join(f'{m:g}' for m in multipliers)}")
    print(f"  L_RVE={args.l_rve_nm:g} nm  tunnel={args.tunnel_nm:g} nm  G_ij=1")
    print(f"  summary: {summary_csv}")
    print(f"  realizations: {real_csv}")
    print(f"  fits: {fit_csv}")
    print("=" * 78)

    all_summary_rows: list[dict[str, str]] = []
    for sys_idx, system in enumerate(systems):
        spec = _system_spec(system, args.hybrid_cb_wt)
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
            f"(variable={variable}, phi_c={phi_c:.6g}) ---"
        )

        for mult_idx, multiplier in enumerate(multipliers):
            if variable == "CB":
                phi_cb_target = phi_c * multiplier
                phi_cnt_target = 0.0
            else:
                phi_cb_target = float(spec["phi_cb_fixed"])
                phi_cnt_target = phi_c * multiplier

            n_cb_max = estimate_cb_n_max(cb_model, max(phi_cb_target, 1e-12), args.l_rve_nm, safety=SAFETY)
            n_cnt_max = estimate_cnt_n_max(
                cnt_model,
                max(phi_cnt_target, 1e-12),
                args.l_rve_nm,
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
                    args.l_rve_nm,
                    args.tunnel_nm,
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
                args.l_rve_nm,
                args.tunnel_nm,
            )
            _write_realizations(real_csv, spec, multiplier, seeds, phi_cb_target, phi_cnt_target, results)
            _write_summary(summary_csv, row)
            all_summary_rows.append(row)
            print(
                f"  n_ok={row['n_ok']}/{row['n_total']}  "
                f"sigma_uncond={float(row['sigma_uncond_mean']):.3e}  "
                f"phi_act={float(row['phi_variable_actual_mean_vol']):.6g}  "
                f"elapsed={elapsed:.1f}s"
            )

    fit_rows = _fit_power_laws(all_summary_rows)
    _write_fits(fit_csv, fit_rows)
    print("\nFits:")
    for row in fit_rows:
        print(
            f"  {row['system']}: n_fit={row['n_fit']}  "
            f"t={row['t_fit'] or 'NA'}  R2={row['r2_loglog'] or 'NA'}"
        )
    print("\nDone.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="pilot")
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument(
        "--phi-multipliers",
        default=",".join(f"{v:g}" for v in DEFAULT_MULTIPLIERS),
    )
    parser.add_argument("--n-real", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hybrid-cb-wt", type=float, default=DEFAULT_HYBRID_CB_WT)
    parser.add_argument("--l-rve-nm", type=float, default=L_RVE_NM)
    parser.add_argument("--tunnel-nm", type=float, default=TUNNEL_NM)
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
