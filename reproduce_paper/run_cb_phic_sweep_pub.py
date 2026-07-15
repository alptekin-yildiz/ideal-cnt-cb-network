#!/usr/bin/env python3
"""Polydisperse CB percolation-threshold sweep for the paper.

This paper reproduction wrapper regenerates the manuscript's Fig. 5 CB
aggregate-size sweep with two explicit RVE conventions:

    ratio : L_RVE = 60 * mu_d, for the low-noise distribution benchmark.
    fixed : L_RVE = 4000 nm, for a production-scale spot-check.

The paper version treats ``mu_d`` and ``sigma_d`` as the target moments of the
truncated distribution actually sampled over [D_min, D_max]. The corresponding
parent log-normal moments are solved deterministically and recorded in the CSV.

This publication wrapper keeps spot-check switches separate from the paper
runs, records per-realization diagnostics, and writes all outputs under
``data/``.

Outputs:
  data/processed/percolation/cb_phic_sweep_Loverd60_momentmatched_D20_500_n300.csv
  data/processed/percolation/cb_phic_sweep_Loverd60_momentmatched_D20_500_n300_realizations.csv
  data/processed/percolation/cb_phic_sweep_L4000_momentmatched_D20_500_n300.csv
  data/processed/percolation/cb_phic_sweep_L4000_momentmatched_D20_500_n300_realizations.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np


# Put the repo root on sys.path so a bare checkout runs without pip install;
# the cntcb imports below intentionally follow this bootstrap (hence E402).
ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT
OUT_DIR = PAPER / "data" / "processed" / "percolation"

sys.path.insert(0, str(ROOT))

from cntcb.kernel.percolation_finder import _SpatialHash, _UnionFind, _sphere_flags  # noqa: E402


# Densities.
RHO_CB = 1.80
RHO_PEI = 1.27

# Simulation conventions.
PHI_MAX_WT = 50.0
TUNNEL_NM = 10.0
D_MIN_NM = 20.0
D_MAX_NM = 500.0
L_RVE_FIXED_NM = 4000.0
L_OVER_D_DEFAULT = 60.0
SEED_BASE = 20260529

# Fig. 7-style CB grid.
MU_D_VALS = (220, 180, 148, 120, 100)
SIGMA_D_VALS = (100, 83, 70, 50, 25)
SIGMA_CONSTRAINT_NM = 20

N_REAL_PRODUCTION = 300
N_REAL_QUICK = 3
N_WORKERS_DEFAULT = 10
PRODUCTION_TAG = "D20_500_n300"
SAFETY = 2.5

CB_BOUNDARY_MODE = "penetrable"
DIAMETER_MOMENT_MODE = "truncated_moment_matched"
NORMAL = NormalDist()

SUMMARY_CSV = OUT_DIR / f"cb_phic_sweep_Loverd60_momentmatched_{PRODUCTION_TAG}.csv"
REAL_CSV = OUT_DIR / f"cb_phic_sweep_Loverd60_momentmatched_{PRODUCTION_TAG}_realizations.csv"

SUMMARY_HEADER = [
    "mu_d_nm",
    "sigma_d_nm",
    "cv",
    "diameter_moment_mode",
    "parent_mu_d_nm",
    "parent_sigma_d_nm",
    "expected_trunc_mean_d_nm",
    "expected_trunc_std_d_nm",
    "expected_trunc_Ed3_nm3",
    "trunc_prob_low",
    "trunc_prob_high",
    "trunc_prob_kept",
    "phi_c_vol",
    "phi_c_wt_pct",
    "phi_c_q25_wt",
    "phi_c_q75_wt",
    "phi_c_ps_vol",
    "phi_c_ps_wt",
    "phi_c_psp_vol",
    "phi_c_psp_wt",
    "delta_phi_wt",
    "n_perc",
    "n_total",
    "n_censored",
    "phi_act_mean",
    "phi_act_std",
    "n_cb_mean",
    "n_cb_std",
    "mean_d_realized_nm",
    "std_d_realized_nm",
    "d_q05_realized_nm",
    "d_q50_realized_nm",
    "d_q95_realized_nm",
    "L_RVE_nm",
    "D_min_nm",
    "D_max_nm",
    "tunnel_nm",
    "phi_max_wt_pct",
    "cb_boundary_mode",
    "elapsed_s",
]

REAL_HEADER = [
    "mu_d_nm",
    "sigma_d_nm",
    "cv",
    "diameter_moment_mode",
    "parent_mu_d_nm",
    "parent_sigma_d_nm",
    "expected_trunc_mean_d_nm",
    "expected_trunc_std_d_nm",
    "realization",
    "seed",
    "percolated",
    "failure_reason",
    "phi_perc_vol_censored",
    "phi_perc_wt_pct_censored",
    "phi_stop_vol",
    "phi_stop_wt_pct",
    "n_cb",
    "mean_d_realized_nm",
    "std_d_realized_nm",
    "min_d_realized_nm",
    "d_q05_realized_nm",
    "d_q50_realized_nm",
    "d_q95_realized_nm",
    "max_d_realized_nm",
    "L_RVE_nm",
    "D_min_nm",
    "D_max_nm",
    "tunnel_nm",
    "phi_max_wt_pct",
    "cb_boundary_mode",
    "elapsed_s",
]


@dataclass(frozen=True)
class RealizationResult:
    percolated: bool
    failure_reason: str
    phi_stop_vol: float
    n_cb: int
    mean_d_realized_nm: float
    std_d_realized_nm: float
    min_d_realized_nm: float
    d_q05_realized_nm: float
    d_q50_realized_nm: float
    d_q95_realized_nm: float
    max_d_realized_nm: float
    elapsed_s: float


@dataclass(frozen=True)
class DiameterModel:
    target_mu_d_nm: float
    target_sigma_d_nm: float
    parent_mu_d_nm: float
    parent_sigma_d_nm: float
    mu_ln: float
    sigma_ln: float
    expected_trunc_mean_d_nm: float
    expected_trunc_std_d_nm: float
    expected_trunc_e_d3_nm3: float
    trunc_prob_low: float
    trunc_prob_high: float
    trunc_prob_kept: float


def wt_frac_to_phi(wt_frac: float) -> float:
    """Convert CB mass fraction to volume fraction."""
    return (wt_frac / RHO_CB) / (wt_frac / RHO_CB + (1.0 - wt_frac) / RHO_PEI)


def phi_to_wt_pct(phi: float) -> float:
    """Convert CB volume fraction to wt.%."""
    wt = phi * RHO_CB / (phi * RHO_CB + (1.0 - phi) * RHO_PEI)
    return 100.0 * wt


def _lognormal_params(mu: float, sigma: float) -> tuple[float, float]:
    """Log-normal parameters from linear-space mean and standard deviation."""
    if sigma <= 0.0:
        return float(np.log(mu)), 0.0
    cv2 = (sigma / mu) ** 2
    sigma_ln = float(np.sqrt(np.log(1.0 + cv2)))
    mu_ln = float(np.log(mu) - 0.5 * sigma_ln**2)
    return mu_ln, sigma_ln


def _normal_cdf(value: float) -> float:
    return float(NORMAL.cdf(value))


def _truncated_raw_moment(
    mu_ln: float,
    sigma_ln: float,
    order: int,
    d_min_nm: float,
    d_max_nm: float,
) -> float:
    """Raw moment E[d^order | D_MIN_NM <= d <= D_MAX_NM]."""
    if sigma_ln <= 1e-12:
        diameter = float(np.exp(mu_ln))
        if not (d_min_nm <= diameter <= d_max_nm):
            raise ValueError("Monodisperse diameter is outside truncation bounds.")
        return diameter**order

    z_lo = (math.log(d_min_nm) - mu_ln) / sigma_ln
    z_hi = (math.log(d_max_nm) - mu_ln) / sigma_ln
    prob_kept = _normal_cdf(z_hi) - _normal_cdf(z_lo)
    if prob_kept <= 0.0:
        raise ValueError("Log-normal truncation interval has zero probability.")

    shifted_lo = (math.log(d_min_nm) - mu_ln - order * sigma_ln**2) / sigma_ln
    shifted_hi = (math.log(d_max_nm) - mu_ln - order * sigma_ln**2) / sigma_ln
    raw_parent = math.exp(order * mu_ln + 0.5 * order**2 * sigma_ln**2)
    return raw_parent * (_normal_cdf(shifted_hi) - _normal_cdf(shifted_lo)) / prob_kept


def _truncated_stats_from_parent(
    parent_mu_d_nm: float,
    parent_sigma_d_nm: float,
    d_min_nm: float,
    d_max_nm: float,
) -> tuple[float, float, float, float, float, float]:
    """Return truncated mean, std, E[d^3], P(d<D_min), P(d>D_max), P(kept)."""
    mu_ln, sigma_ln = _lognormal_params(parent_mu_d_nm, parent_sigma_d_nm)
    if sigma_ln <= 1e-12:
        mean = float(parent_mu_d_nm)
        return mean, 0.0, mean**3, 0.0, 0.0, 1.0

    z_lo = (math.log(d_min_nm) - mu_ln) / sigma_ln
    z_hi = (math.log(d_max_nm) - mu_ln) / sigma_ln
    prob_low = _normal_cdf(z_lo)
    prob_high = 1.0 - _normal_cdf(z_hi)
    prob_kept = 1.0 - prob_low - prob_high

    e1 = _truncated_raw_moment(mu_ln, sigma_ln, 1, d_min_nm, d_max_nm)
    e2 = _truncated_raw_moment(mu_ln, sigma_ln, 2, d_min_nm, d_max_nm)
    e3 = _truncated_raw_moment(mu_ln, sigma_ln, 3, d_min_nm, d_max_nm)
    std = math.sqrt(max(0.0, e2 - e1**2))
    return e1, std, e3, prob_low, prob_high, prob_kept


def _solve_parent_for_truncated_moments(
    target_mu_d_nm: float,
    target_sigma_d_nm: float,
    d_min_nm: float,
    d_max_nm: float,
) -> DiameterModel:
    """Find parent log-normal moments that yield target truncated moments."""
    if target_mu_d_nm <= 0.0 or target_sigma_d_nm < 0.0:
        raise ValueError("CB diameter moments must be positive.")
    if target_sigma_d_nm <= 1e-12:
        mu_ln, sigma_ln = _lognormal_params(target_mu_d_nm, 0.0)
        return DiameterModel(
            target_mu_d_nm=target_mu_d_nm,
            target_sigma_d_nm=target_sigma_d_nm,
            parent_mu_d_nm=target_mu_d_nm,
            parent_sigma_d_nm=0.0,
            mu_ln=mu_ln,
            sigma_ln=sigma_ln,
            expected_trunc_mean_d_nm=target_mu_d_nm,
            expected_trunc_std_d_nm=0.0,
            expected_trunc_e_d3_nm3=target_mu_d_nm**3,
            trunc_prob_low=0.0,
            trunc_prob_high=0.0,
            trunc_prob_kept=1.0,
        )

    # Unknowns are log(parent mean) and log(parent std). Newton is stable for
    # the present grid because the truncation interval only clips the far tail.
    y = math.log(target_mu_d_nm)
    z = math.log(target_sigma_d_nm * 1.05)
    for _ in range(80):
        parent_mu = math.exp(y)
        parent_sigma = math.exp(z)
        mean, std, *_ = _truncated_stats_from_parent(
            parent_mu, parent_sigma, d_min_nm, d_max_nm
        )
        f0 = mean - target_mu_d_nm
        f1 = std - target_sigma_d_nm
        if abs(f0) < 1e-8 and abs(f1) < 1e-8:
            break

        h = 1e-5
        mean_y, std_y, *_ = _truncated_stats_from_parent(
            math.exp(y + h), parent_sigma, d_min_nm, d_max_nm
        )
        mean_z, std_z, *_ = _truncated_stats_from_parent(
            parent_mu, math.exp(z + h), d_min_nm, d_max_nm
        )
        j00 = (mean_y - mean) / h
        j10 = (std_y - std) / h
        j01 = (mean_z - mean) / h
        j11 = (std_z - std) / h
        det = j00 * j11 - j01 * j10
        if abs(det) < 1e-14:
            raise RuntimeError(
                f"Could not solve CB parent moments for {target_mu_d_nm}/{target_sigma_d_nm} nm."
            )

        dy = (-f0 * j11 + j01 * f1) / det
        dz = (j10 * f0 - j00 * f1) / det

        damping = 1.0
        current_error = f0**2 + f1**2
        for _try in range(24):
            next_y = y + damping * dy
            next_z = z + damping * dz
            next_mean, next_std, *_ = _truncated_stats_from_parent(
                math.exp(next_y), math.exp(next_z), d_min_nm, d_max_nm
            )
            next_error = (next_mean - target_mu_d_nm) ** 2 + (
                next_std - target_sigma_d_nm
            ) ** 2
            if math.isfinite(next_error) and next_error <= current_error:
                y = next_y
                z = next_z
                break
            damping *= 0.5
        else:
            raise RuntimeError(
                f"Moment solver stalled for {target_mu_d_nm}/{target_sigma_d_nm} nm."
            )

    parent_mu = math.exp(y)
    parent_sigma = math.exp(z)
    mu_ln, sigma_ln = _lognormal_params(parent_mu, parent_sigma)
    mean, std, e3, prob_low, prob_high, prob_kept = _truncated_stats_from_parent(
        parent_mu, parent_sigma, d_min_nm, d_max_nm
    )
    if abs(mean - target_mu_d_nm) > 1e-5 or abs(std - target_sigma_d_nm) > 1e-5:
        raise RuntimeError(
            f"Moment solver did not converge for {target_mu_d_nm}/{target_sigma_d_nm} nm."
        )

    return DiameterModel(
        target_mu_d_nm=target_mu_d_nm,
        target_sigma_d_nm=target_sigma_d_nm,
        parent_mu_d_nm=parent_mu,
        parent_sigma_d_nm=parent_sigma,
        mu_ln=mu_ln,
        sigma_ln=sigma_ln,
        expected_trunc_mean_d_nm=mean,
        expected_trunc_std_d_nm=std,
        expected_trunc_e_d3_nm3=e3,
        trunc_prob_low=prob_low,
        trunc_prob_high=prob_high,
        trunc_prob_kept=prob_kept,
    )


def phi_c_ps_mono(mu_d_nm: float) -> float:
    """Pike-Seager-style monodisperse Boolean-sphere threshold, by volume."""
    d_eff = mu_d_nm + TUNNEL_NM
    return 0.3418 * (mu_d_nm / d_eff) ** 3


def phi_c_ps_poly(model: DiameterModel, d_min_nm: float, d_max_nm: float) -> float:
    """Mean-field polydisperse Boolean-sphere guide for the sampled distribution."""
    if model.target_sigma_d_nm <= 0.0:
        return phi_c_ps_mono(model.target_mu_d_nm)

    e1 = model.expected_trunc_mean_d_nm
    e2 = _truncated_raw_moment(model.mu_ln, model.sigma_ln, 2, d_min_nm, d_max_nm)
    e3 = model.expected_trunc_e_d3_nm3
    e_deff3 = e3 + 3.0 * TUNNEL_NM * e2 + 3.0 * TUNNEL_NM**2 * e1 + TUNNEL_NM**3
    if e_deff3 < 1e-30:
        return float("nan")
    return 0.3418 * e3 / e_deff3


def _n_max_estimate(
    expected_e_d3_nm3: float,
    phi_max_vol: float,
    l_rve_nm: float,
) -> int:
    """Conservative particle-count cap at phi_max."""
    v_mean = (np.pi / 6.0) * expected_e_d3_nm3
    n_exp = phi_max_vol * l_rve_nm**3 / v_mean
    return max(200, int(math.ceil(n_exp * SAFETY)))


def _sample_diameter(
    rng: np.random.Generator,
    mu_ln: float,
    sigma_ln: float,
    cdf_lo: float,
    cdf_rng: float,
    d_min_nm: float,
    d_max_nm: float,
) -> float:
    if sigma_ln <= 1e-12:
        return float(np.exp(mu_ln))
    u = float(rng.uniform(0.0, 1.0))
    u_scaled = cdf_lo + u * cdf_rng
    u_scaled = max(1e-14, min(u_scaled, 1.0 - 1e-14))
    diameter = float(np.exp(mu_ln + sigma_ln * NORMAL.inv_cdf(u_scaled)))
    return max(d_min_nm, min(diameter, d_max_nm))


def _run_realization(
    args: tuple[float, float, int, float, float, float, float, int],
) -> RealizationResult:
    """One volume-stopped penetrable-CB realization."""
    mu_ln, sigma_ln, n_max, phi_max_vol, l_rve_nm, d_min_nm, d_max_nm, seed = args

    t0 = time.time()
    rng = np.random.default_rng(seed)

    if sigma_ln > 1e-12:
        cdf_lo = float(NORMAL.cdf((np.log(d_min_nm) - mu_ln) / sigma_ln))
        cdf_hi = float(NORMAL.cdf((np.log(d_max_nm) - mu_ln) / sigma_ln))
        cdf_rng = cdf_hi - cdf_lo
    else:
        cdf_lo = 0.0
        cdf_rng = 0.0

    centers = np.empty((n_max, 3), dtype=np.float64)
    diameters = np.empty(n_max, dtype=np.float64)
    uf = _UnionFind(n_max)
    shash = _SpatialHash(cell_size=d_max_nm + TUNNEL_NM)

    v_rve = l_rve_nm**3
    v_target = phi_max_vol * v_rve
    vol_total = 0.0
    n_act = 0

    for step in range(n_max):
        d_i = _sample_diameter(rng, mu_ln, sigma_ln, cdf_lo, cdf_rng, d_min_nm, d_max_nm)
        center = rng.uniform(0.0, l_rve_nm, size=3)

        r_i = d_i / 2.0
        flags = _sphere_flags(center, r_i, l_rve_nm)
        uf.set_boundary(step, flags)

        percolated = uf.is_root_percolating(step)
        for j in shash.query(center):
            d_j = diameters[j]
            connect_d = (d_i + d_j) / 2.0 + TUNNEL_NM
            dist = float(np.linalg.norm(center - centers[j]))
            if dist < connect_d and uf.union(step, j):
                percolated = True

        centers[step] = center
        diameters[step] = d_i
        shash.insert(step, center)
        n_act += 1
        vol_total += (np.pi / 6.0) * d_i**3

        if percolated:
            return _finalize_result(
                True, "", vol_total, n_act, diameters, l_rve_nm, t0
            )

        if vol_total >= v_target:
            return _finalize_result(
                False, "phi_max", vol_total, n_act, diameters, l_rve_nm, t0
            )

    return _finalize_result(False, "n_max", vol_total, n_act, diameters, l_rve_nm, t0)


def _finalize_result(
    percolated: bool,
    failure_reason: str,
    vol_total: float,
    n_cb: int,
    diameters: np.ndarray,
    l_rve_nm: float,
    t0: float,
) -> RealizationResult:
    if n_cb <= 0:
        return RealizationResult(
            percolated=percolated,
            failure_reason=failure_reason or "empty",
            phi_stop_vol=0.0,
            n_cb=0,
            mean_d_realized_nm=float("nan"),
            std_d_realized_nm=float("nan"),
            min_d_realized_nm=float("nan"),
            d_q05_realized_nm=float("nan"),
            d_q50_realized_nm=float("nan"),
            d_q95_realized_nm=float("nan"),
            max_d_realized_nm=float("nan"),
            elapsed_s=time.time() - t0,
        )

    d = diameters[:n_cb]
    phi_stop = vol_total / l_rve_nm**3
    return RealizationResult(
        percolated=percolated,
        failure_reason=failure_reason,
        phi_stop_vol=float(phi_stop),
        n_cb=int(n_cb),
        mean_d_realized_nm=float(np.mean(d)),
        std_d_realized_nm=float(np.std(d, ddof=1)) if n_cb > 1 else 0.0,
        min_d_realized_nm=float(np.min(d)),
        d_q05_realized_nm=float(np.quantile(d, 0.05)),
        d_q50_realized_nm=float(np.quantile(d, 0.50)),
        d_q95_realized_nm=float(np.quantile(d, 0.95)),
        max_d_realized_nm=float(np.max(d)),
        elapsed_s=time.time() - t0,
    )


def _available_memory_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        return 16.0


def _worker_count(n_max: int, requested: int) -> int:
    if requested <= 1:
        return 1
    bytes_per_worker = n_max * 320
    budget_bytes = _available_memory_gb() * (1024**3) * 0.55
    mem_cap = max(1, int(budget_bytes / max(bytes_per_worker, 1)))
    return max(1, min(requested, mem_cap))


def _ensure_header(path: Path, header: list[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        csv.writer(f).writerow(header)


def _load_done(path: Path) -> set[tuple[int, int]]:
    if not path.exists():
        return set()
    done: set[tuple[int, int]] = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                done.add((int(float(row["mu_d_nm"])), int(float(row["sigma_d_nm"]))))
            except (KeyError, ValueError):
                pass
    return done


def _parse_int_list(text: str, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if not text:
        return fallback
    vals = []
    for chunk in text.split(","):
        item = chunk.strip()
        if item:
            vals.append(int(float(item)))
    if not vals:
        raise ValueError("List argument did not contain any numeric value.")
    return tuple(vals)


def _parse_pairs(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    if not text:
        return pairs
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid pair {item!r}; expected 'mu:sigma', e.g. '148:83'."
            )
        mu_text, sigma_text = item.split(":", 1)
        pairs.append((int(float(mu_text.strip())), int(float(sigma_text.strip()))))
    return pairs


def _write_realizations(
    path: Path,
    mu_d: int,
    sigma_d: int,
    model: DiameterModel,
    seeds: list[int],
    results: list[RealizationResult],
    phi_max_vol: float,
    l_rve_nm: float,
    d_min_nm: float,
    d_max_nm: float,
) -> None:
    cv = sigma_d / mu_d
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        for i, (seed, res) in enumerate(zip(seeds, results), start=1):
            phi_censored = res.phi_stop_vol if res.percolated else phi_max_vol
            writer.writerow(
                [
                    mu_d,
                    sigma_d,
                    f"{cv:.4f}",
                    DIAMETER_MOMENT_MODE,
                    f"{model.parent_mu_d_nm:.8g}",
                    f"{model.parent_sigma_d_nm:.8g}",
                    f"{model.expected_trunc_mean_d_nm:.8g}",
                    f"{model.expected_trunc_std_d_nm:.8g}",
                    i,
                    seed,
                    res.percolated,
                    res.failure_reason,
                    f"{phi_censored:.8g}",
                    f"{phi_to_wt_pct(phi_censored):.6g}",
                    f"{res.phi_stop_vol:.8g}",
                    f"{phi_to_wt_pct(res.phi_stop_vol):.6g}",
                    res.n_cb,
                    f"{res.mean_d_realized_nm:.8g}",
                    f"{res.std_d_realized_nm:.8g}",
                    f"{res.min_d_realized_nm:.8g}",
                    f"{res.d_q05_realized_nm:.8g}",
                    f"{res.d_q50_realized_nm:.8g}",
                    f"{res.d_q95_realized_nm:.8g}",
                    f"{res.max_d_realized_nm:.8g}",
                    l_rve_nm,
                    d_min_nm,
                    d_max_nm,
                    TUNNEL_NM,
                    PHI_MAX_WT,
                    CB_BOUNDARY_MODE,
                    f"{res.elapsed_s:.3f}",
                ]
            )


def _write_summary(
    path: Path,
    mu_d: int,
    sigma_d: int,
    model: DiameterModel,
    results: list[RealizationResult],
    phi_max_vol: float,
    elapsed_s: float,
    l_rve_nm: float,
    d_min_nm: float,
    d_max_nm: float,
) -> None:
    cv = sigma_d / mu_d
    phi_vals = np.asarray(
        [res.phi_stop_vol if res.percolated else phi_max_vol for res in results],
        dtype=float,
    )
    phi_c_vol = float(np.median(phi_vals))
    phi_q25_vol = float(np.quantile(phi_vals, 0.25))
    phi_q75_vol = float(np.quantile(phi_vals, 0.75))
    n_perc = int(sum(res.percolated for res in results))
    n_total = len(results)
    n_censored = n_total - n_perc

    ps_vol = phi_c_ps_mono(mu_d)
    ps_wt = phi_to_wt_pct(ps_vol)
    psp_vol = phi_c_ps_poly(model, d_min_nm, d_max_nm)
    psp_wt = phi_to_wt_pct(psp_vol)
    phi_c_wt = phi_to_wt_pct(phi_c_vol)

    phi_act = np.asarray([res.phi_stop_vol for res in results], dtype=float)
    n_cb = np.asarray([res.n_cb for res in results], dtype=float)
    mean_d = np.asarray([res.mean_d_realized_nm for res in results], dtype=float)
    std_d = np.asarray([res.std_d_realized_nm for res in results], dtype=float)
    q05_d = np.asarray([res.d_q05_realized_nm for res in results], dtype=float)
    q50_d = np.asarray([res.d_q50_realized_nm for res in results], dtype=float)
    q95_d = np.asarray([res.d_q95_realized_nm for res in results], dtype=float)

    with path.open("a", newline="") as f:
        csv.writer(f).writerow(
            [
                mu_d,
                sigma_d,
                f"{cv:.4f}",
                DIAMETER_MOMENT_MODE,
                f"{model.parent_mu_d_nm:.8g}",
                f"{model.parent_sigma_d_nm:.8g}",
                f"{model.expected_trunc_mean_d_nm:.8g}",
                f"{model.expected_trunc_std_d_nm:.8g}",
                f"{model.expected_trunc_e_d3_nm3:.8g}",
                f"{model.trunc_prob_low:.8g}",
                f"{model.trunc_prob_high:.8g}",
                f"{model.trunc_prob_kept:.8g}",
                f"{phi_c_vol:.8g}",
                f"{phi_c_wt:.6g}",
                f"{phi_to_wt_pct(phi_q25_vol):.6g}",
                f"{phi_to_wt_pct(phi_q75_vol):.6g}",
                f"{ps_vol:.8g}",
                f"{ps_wt:.6g}",
                f"{psp_vol:.8g}",
                f"{psp_wt:.6g}",
                f"{phi_c_wt - ps_wt:.6g}",
                n_perc,
                n_total,
                n_censored,
                f"{float(np.mean(phi_act)):.8g}",
                f"{float(np.std(phi_act, ddof=1)) if len(phi_act) > 1 else 0.0:.8g}",
                f"{float(np.mean(n_cb)):.8g}",
                f"{float(np.std(n_cb, ddof=1)) if len(n_cb) > 1 else 0.0:.8g}",
                f"{float(np.mean(mean_d)):.8g}",
                f"{float(np.mean(std_d)):.8g}",
                f"{float(np.mean(q05_d)):.8g}",
                f"{float(np.mean(q50_d)):.8g}",
                f"{float(np.mean(q95_d)):.8g}",
                l_rve_nm,
                d_min_nm,
                d_max_nm,
                TUNNEL_NM,
                PHI_MAX_WT,
                CB_BOUNDARY_MODE,
                f"{elapsed_s:.3f}",
            ]
        )


def _run_tasks(
    tasks: list[tuple],
    workers: int,
    progress_every: int,
) -> list[RealizationResult]:
    results: list[RealizationResult | None] = [None] * len(tasks)
    t0 = time.time()
    n_done = 0

    def record(idx: int, result: RealizationResult) -> None:
        nonlocal n_done
        results[idx - 1] = result
        n_done += 1
        if progress_every > 0 and (n_done % progress_every == 0 or n_done == len(tasks)):
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0.0
            eta = (len(tasks) - n_done) / rate if rate > 0 else float("nan")
            n_ok = sum(1 for r in results if r is not None and r.percolated)
            n_fail = sum(1 for r in results if r is not None and not r.percolated)
            print(
                f"    progress {n_done}/{len(tasks)}  ok={n_ok}  fail={n_fail}  "
                f"elapsed={elapsed:.1f}s  eta={eta:.1f}s",
                flush=True,
            )

    if workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            record(idx, _run_realization(task))
        return [r for r in results if r is not None]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(_run_realization, task): idx for idx, task in enumerate(tasks, start=1)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            record(idx, future.result())

    return [r for r in results if r is not None]


def _run_point(
    mu_d: int,
    sigma_d: int,
    n_real: int,
    requested_workers: int,
    summary_csv: Path,
    real_csv: Path,
    l_rve_nm: float,
    d_min_nm: float,
    d_max_nm: float,
    progress_every: int,
) -> None:
    phi_max_vol = wt_frac_to_phi(PHI_MAX_WT / 100.0)
    model = _solve_parent_for_truncated_moments(
        float(mu_d), float(sigma_d), d_min_nm, d_max_nm
    )
    n_max = _n_max_estimate(model.expected_trunc_e_d3_nm3, phi_max_vol, l_rve_nm)
    n_workers = _worker_count(n_max, requested_workers)
    cv = sigma_d / mu_d

    seeds = [
        SEED_BASE + (int(mu_d) * 1000 + int(sigma_d)) * 1000 + r
        for r in range(n_real)
    ]
    tasks = [
        (model.mu_ln, model.sigma_ln, n_max, phi_max_vol, l_rve_nm, d_min_nm, d_max_nm, seed)
        for seed in seeds
    ]

    print(
        f"CB mu={mu_d:4d} nm  sigma={sigma_d:3d} nm  CV={cv:.3f}  "
        f"parent=({model.parent_mu_d_nm:.2f},{model.parent_sigma_d_nm:.2f}) nm  "
        f"L={l_rve_nm:7.0f} nm  n={n_real:3d}  "
        f"n_max={n_max:7d}  workers={n_workers:2d}",
        flush=True,
    )
    t0 = time.time()
    results = _run_tasks(tasks, n_workers, progress_every)
    elapsed = time.time() - t0

    _write_realizations(
        real_csv, mu_d, sigma_d, model, seeds, results, phi_max_vol, l_rve_nm,
        d_min_nm, d_max_nm
    )
    _write_summary(
        summary_csv, mu_d, sigma_d, model, results, phi_max_vol, elapsed,
        l_rve_nm, d_min_nm, d_max_nm
    )

    phi_vals = [res.phi_stop_vol if res.percolated else phi_max_vol for res in results]
    phi_c_wt = phi_to_wt_pct(float(np.median(phi_vals)))
    n_perc = sum(res.percolated for res in results)
    print(
        f"     -> phi_c={phi_c_wt:.4f} wt%  "
        f"n_perc={n_perc}/{n_real}  elapsed={elapsed:.1f}s",
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    global SUMMARY_CSV, REAL_CSV

    rve_mode = args.rve_mode
    if rve_mode == "ratio":
        rve_label = f"Loverd{args.l_over_d:g}".replace(".", "p")
    else:
        rve_label = f"L{args.l_rve:g}".replace(".", "p")

    if args.tag:
        safe_tag = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in args.tag)
        suffix = f"_{safe_tag}"
    elif args.quick:
        suffix = "_quick"
    else:
        suffix = f"_{PRODUCTION_TAG}"

    SUMMARY_CSV = OUT_DIR / f"cb_phic_sweep_{rve_label}_momentmatched{suffix}.csv"
    REAL_CSV = OUT_DIR / f"cb_phic_sweep_{rve_label}_momentmatched{suffix}_realizations.csv"

    mu_fallback = (148,) if args.quick else MU_D_VALS
    sigma_fallback = (83,) if args.quick else SIGMA_D_VALS
    mu_vals = _parse_int_list(args.mu_list, mu_fallback)
    sigma_vals = _parse_int_list(args.sigma_list, sigma_fallback)
    point_pairs = _parse_pairs(args.pairs)
    n_real = args.n_real if args.n_real is not None else (
        N_REAL_QUICK if args.quick else N_REAL_PRODUCTION
    )
    requested_workers = min(args.workers, os.cpu_count() or args.workers)

    _ensure_header(SUMMARY_CSV, SUMMARY_HEADER)
    _ensure_header(REAL_CSV, REAL_HEADER)
    done = _load_done(SUMMARY_CSV)

    print("=" * 78)
    print("Paper CB phi_c sweep: polydisperse aggregate diameter")
    print(f"Mode: {'quick canonical point' if args.quick else 'production grid'}")
    print(f"RVE mode: {rve_mode}")
    print(f"Diameter moments: {DIAMETER_MOMENT_MODE}")
    print(f"Output suffix: {suffix or '[none]'}")
    if rve_mode == "ratio":
        print(f"L_RVE={args.l_over_d:g} x mu_d  tunnel={TUNNEL_NM:g} nm  boundary={CB_BOUNDARY_MODE}")
    else:
        print(f"L_RVE={args.l_rve:g} nm  tunnel={TUNNEL_NM:g} nm  boundary={CB_BOUNDARY_MODE}")
    print(f"D range=[{args.d_min:g}, {args.d_max:g}] nm")
    print(f"phi_max={PHI_MAX_WT:g} wt%")
    print(f"N_real per point={n_real}  requested workers={requested_workers}")
    print("=" * 78)

    if not point_pairs:
        point_pairs = [(int(mu_d), int(sigma_d)) for mu_d in mu_vals for sigma_d in sigma_vals]

    for mu_d, sigma_d in point_pairs:
        if sigma_d >= mu_d - SIGMA_CONSTRAINT_NM:
            print(
                f"CB mu={mu_d:4d} nm  sigma={sigma_d:3d} nm  skip "
                f"(constraint sigma < mu - {SIGMA_CONSTRAINT_NM:g} nm)"
            )
            continue
        key = (int(mu_d), int(sigma_d))
        if key in done:
            print(f"CB mu={mu_d:4d} nm  sigma={sigma_d:3d} nm  skip (already in CSV)")
            continue
        l_rve_nm = float(args.l_over_d * mu_d) if rve_mode == "ratio" else float(args.l_rve)
        _run_point(
            int(mu_d),
            int(sigma_d),
            int(n_real),
            requested_workers,
            SUMMARY_CSV,
            REAL_CSV,
            l_rve_nm,
            float(args.d_min),
            float(args.d_max),
            args.progress_every,
        )

    print("=" * 78)
    print("Done.")
    print(f"Summary      : {SUMMARY_CSV}")
    print(f"Realizations : {REAL_CSV}")
    print("=" * 78)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the publication polydisperse CB phi_c sweep."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only the canonical (mu=148, sigma=83) point with small N.",
    )
    parser.add_argument(
        "--n-real",
        type=int,
        default=None,
        help="Override realization count per grid point.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=N_WORKERS_DEFAULT,
        help="Requested maximum worker count; script may cap this by memory.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help=(
            "Optional output suffix tag, e.g. smoke or trial1. The production "
            f"default is {PRODUCTION_TAG}."
        ),
    )
    parser.add_argument(
        "--mu-list",
        default="",
        help="Comma-separated mean-diameter values in nm. Defaults to the production grid.",
    )
    parser.add_argument(
        "--sigma-list",
        default="",
        help="Comma-separated diameter-standard-deviation values in nm. Defaults to the production grid.",
    )
    parser.add_argument(
        "--pairs",
        default="",
        help="Comma-separated mu:sigma pairs, e.g. 148:83,148:100. Overrides the grid cross-product.",
    )
    parser.add_argument(
        "--rve-mode",
        choices=("ratio", "fixed"),
        default="ratio",
        help="RVE convention: ratio uses L_RVE=l_over_d*mu_d; fixed uses --l-rve.",
    )
    parser.add_argument(
        "--l-over-d",
        type=float,
        default=L_OVER_D_DEFAULT,
        help="RVE/mean-diameter ratio used when --rve-mode ratio.",
    )
    parser.add_argument(
        "--l-rve",
        type=float,
        default=L_RVE_FIXED_NM,
        help="Fixed RVE edge length in nm used when --rve-mode fixed.",
    )
    parser.add_argument(
        "--d-min",
        type=float,
        default=D_MIN_NM,
        help="Lower diameter support bound in nm.",
    )
    parser.add_argument(
        "--d-max",
        type=float,
        default=D_MAX_NM,
        help="Upper diameter support bound in nm.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress after this many completed realizations; use 0 to disable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
