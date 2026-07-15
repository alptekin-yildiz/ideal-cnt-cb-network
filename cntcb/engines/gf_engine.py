"""Gauge-factor engine (behind ``generate/gauge_factor.py``).

The GF calculation reuses the Simmons conductivity engine.  A single
baseline RVE is sampled for each realization and then affinely deformed
over the strain grid; no strain point is resampled independently.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .conductivity_engine import (
    CBPopulation,
    CNTPopulation,
    sample_cb_population,
    sample_cnt_population,
    solve_step_network,
)
from .percolation_engine import TruncatedLognormal, _run_ensemble_tasks
from .simmons_conductivity_engine import (
    build_simmons_conductance_network,
)


@dataclass(frozen=True)
class SimmonsGFResult:
    """One affine-strain Simmons GF realization."""

    success: bool
    failure_reason: str
    failure_stage: str
    n_strain_ok: int
    gf_r: float
    gf_sigma: float
    gf_geom: float
    gf_r_from_sigma: float
    gf_r_minus_recon: float
    r2_lng_vs_eps: float
    max_resid_lng: float
    r2_lnsigma_vs_eps: float
    max_resid_lnsigma: float
    g_eff_eps0_s: float
    g_eff_epsmax_s: float
    sigma_x_eps0_s_per_m: float
    sigma_x_epsmax_s_per_m: float
    topology_switch_fraction: float
    log10_g_range_max: float
    residual_ratio_max: float
    opening_alpha_ss_nm_per_strain: float
    opening_alpha_sc_nm_per_strain: float
    opening_alpha_cc_nm_per_strain: float
    power_frac_ss_eps0: float
    power_frac_sc_eps0: float
    power_frac_cc_eps0: float
    power_frac_ss_epsmax: float
    power_frac_sc_epsmax: float
    power_frac_cc_epsmax: float
    phi_cb_actual_vol: float
    phi_cnt_actual_vol: float
    n_cb: int
    n_cnt: int
    n_total_segs: int
    mean_d_realized_nm: float
    std_d_realized_nm: float
    mean_L_realized_nm: float
    std_L_realized_nm: float
    mean_n_seg_per_cnt: float
    max_n_seg_per_cnt: int
    strain_grid: tuple[float, ...]
    g_eff_series_s: tuple[float, ...]
    sigma_x_series_s_per_m: tuple[float, ...]
    n_edges_series: tuple[float, ...]
    power_total_series: tuple[float, ...]
    power_frac_ss_series: tuple[float, ...]
    power_frac_sc_series: tuple[float, ...]
    power_frac_cc_series: tuple[float, ...]
    largest_component_fraction_series: tuple[float, ...]
    log10_g_range_series: tuple[float, ...]
    residual_ratio_series: tuple[float, ...]
    failure_by_eps: tuple[str, ...]
    elapsed_s: float


def deform_cb_population(cb: CBPopulation, eps: float, nu: float) -> CBPopulation:
    """Return an affinely deformed CB population; diameters are unchanged."""

    scale = np.array([1.0 + eps, 1.0 - nu * eps, 1.0 - nu * eps], dtype=np.float64)
    return CBPopulation(
        centers=cb.centers * scale[None, :],
        diameters=cb.diameters.copy(),
        phi_actual_vol=cb.phi_actual_vol,
    )


def deform_cnt_population(cnt: CNTPopulation, eps: float, nu: float) -> CNTPopulation:
    """Return an affinely deformed CNT population; radii and metadata are fixed."""

    scale = np.array([1.0 + eps, 1.0 - nu * eps, 1.0 - nu * eps], dtype=np.float64)
    return CNTPopulation(
        seg_starts=cnt.seg_starts * scale[None, :],
        seg_ends=cnt.seg_ends * scale[None, :],
        seg_offsets=cnt.seg_offsets.copy(),
        seg_to_cnt=cnt.seg_to_cnt.copy(),
        lengths=cnt.lengths.copy(),
        n_seg_per_cnt=cnt.n_seg_per_cnt.copy(),
        phi_actual_vol=cnt.phi_actual_vol,
    )


def _linear_fit_log(
    eps_arr: np.ndarray,
    values: np.ndarray,
) -> tuple[float, int, float, float]:
    """Fit ln(values) = a eps + b and return slope, n, R2, max residual."""

    valid = np.isfinite(values) & (values > 0.0)
    n_valid = int(np.count_nonzero(valid))
    if n_valid < 3:
        return float("nan"), n_valid, float("nan"), float("nan")

    x = eps_arr[valid]
    y = np.log(values[valid])
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    residuals = y - pred
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    max_resid = float(np.max(np.abs(residuals)))
    return float(slope), n_valid, r2, max_resid


def _series_value(values: list[float], index: int) -> float:
    if not values:
        return float("nan")
    value = values[index]
    return float(value) if np.isfinite(value) else float("nan")


def _max_finite(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if len(arr) else float("nan")


def _topology_switch_fraction(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2 or float(np.min(arr)) <= 0.0:
        return float("nan")
    return float((np.max(arr) - np.min(arr)) / np.min(arr))


def run_simmons_gf_realization(
    *,
    cb_model: TruncatedLognormal,
    cnt_model: TruncatedLognormal,
    phi_cb_target_vol: float,
    phi_cnt_target_vol: float,
    l_rve_nm: float,
    cnt_diam_nm: float,
    waviness: float,
    tunnel_nm: float,
    seg_len_unit_nm: float,
    d_min_nm: float,
    g_cutoff_s: float,
    strain_grid: tuple[float, ...],
    poisson_nu: float,
    seed: int,
    opening_alpha_ss_nm_per_strain: float = 0.0,
    opening_alpha_sc_nm_per_strain: float = 0.0,
    opening_alpha_cc_nm_per_strain: float = 0.0,
    n_cb_max: int | None = None,
    n_cnt_max: int | None = None,
    direction: int = 0,
) -> SimmonsGFResult:
    """Sample one RVE, deform it over strain, and compute GF diagnostics."""

    t0 = time.time()
    eps_arr = np.asarray(strain_grid, dtype=np.float64)
    if len(eps_arr) < 3:
        raise ValueError("strain_grid must contain at least three points.")
    if np.any(eps_arr < 0.0) or np.any(np.diff(eps_arr) < 0.0):
        raise ValueError("strain_grid must be non-negative and sorted.")
    if direction != 0:
        raise NotImplementedError("GF engine currently supports x-direction loading only.")

    rng = np.random.default_rng(seed)
    try:
        cb0 = sample_cb_population(
            rng=rng,
            cb_model=cb_model,
            phi_target_vol=phi_cb_target_vol,
            l_rve_nm=l_rve_nm,
            n_max=n_cb_max,
        )
        cnt0 = sample_cnt_population(
            rng=rng,
            cnt_model=cnt_model,
            phi_target_vol=phi_cnt_target_vol,
            l_rve_nm=l_rve_nm,
            cnt_diam_nm=cnt_diam_nm,
            waviness=waviness,
            seg_len_unit_nm=seg_len_unit_nm,
            n_max=n_cnt_max,
        )
    except Exception as exc:
        return _failed_result(
            reason=f"sample_exception:{type(exc).__name__}",
            stage="sampling",
            strain_grid=tuple(float(e) for e in eps_arr),
            elapsed_s=time.time() - t0,
        )

    g_eff: list[float] = []
    sigma_x: list[float] = []
    n_edges_series: list[float] = []
    power_total_series: list[float] = []
    power_frac_ss_series: list[float] = []
    power_frac_sc_series: list[float] = []
    power_frac_cc_series: list[float] = []
    lcf_series: list[float] = []
    log10_g_range_series: list[float] = []
    residual_ratio_series: list[float] = []
    failure_by_eps: list[str] = []

    for eps in eps_arr:
        eps_f = float(eps)
        lx = l_rve_nm * (1.0 + eps_f)
        ly = l_rve_nm * (1.0 - poisson_nu * eps_f)
        lz = ly
        if lx <= 0.0 or ly <= 0.0:
            g_eff.append(float("nan"))
            sigma_x.append(float("nan"))
            n_edges_series.append(float("nan"))
            power_total_series.append(float("nan"))
            power_frac_ss_series.append(float("nan"))
            power_frac_sc_series.append(float("nan"))
            power_frac_cc_series.append(float("nan"))
            lcf_series.append(float("nan"))
            log10_g_range_series.append(float("nan"))
            residual_ratio_series.append(float("nan"))
            failure_by_eps.append("invalid_deformed_cell")
            continue

        cb_eps = deform_cb_population(cb0, eps_f, poisson_nu)
        cnt_eps = deform_cnt_population(cnt0, eps_f, poisson_nu)
        try:
            graph, g_stats = build_simmons_conductance_network(
                cb=cb_eps,
                cnt=cnt_eps,
                cb_model=cb_model,
                cnt_diam_nm=cnt_diam_nm,
                tunnel_nm=tunnel_nm,
                seg_len_unit_nm=seg_len_unit_nm,
                d_min_nm=d_min_nm,
                g_cutoff_s=g_cutoff_s,
                distance_offset_ss_nm=opening_alpha_ss_nm_per_strain * eps_f,
                distance_offset_sc_nm=opening_alpha_sc_nm_per_strain * eps_f,
                distance_offset_cc_nm=opening_alpha_cc_nm_per_strain * eps_f,
            )
            sigma_lx, diagnostics = solve_step_network(
                G=graph,
                cb=cb_eps,
                cnt=cnt_eps,
                cnt_diam_nm=cnt_diam_nm,
                l_rve_nm=lx,
                direction=direction,
            )
        except Exception as exc:
            g_eff.append(float("nan"))
            sigma_x.append(float("nan"))
            n_edges_series.append(float("nan"))
            power_total_series.append(float("nan"))
            power_frac_ss_series.append(float("nan"))
            power_frac_sc_series.append(float("nan"))
            power_frac_cc_series.append(float("nan"))
            lcf_series.append(float("nan"))
            log10_g_range_series.append(float("nan"))
            residual_ratio_series.append(float("nan"))
            failure_by_eps.append(f"solve_exception:{type(exc).__name__}")
            continue

        n_edges_series.append(float(diagnostics.get("n_edges", float("nan"))))
        power_total_series.append(
            float(diagnostics.get("power_total", float("nan")))
        )
        power_frac_ss_series.append(
            float(diagnostics.get("power_frac_ss", float("nan")))
        )
        power_frac_sc_series.append(
            float(diagnostics.get("power_frac_sc", float("nan")))
        )
        power_frac_cc_series.append(
            float(diagnostics.get("power_frac_cc", float("nan")))
        )
        lcf_series.append(
            float(diagnostics.get("largest_component_fraction", float("nan")))
        )
        log10_g_range_series.append(float(g_stats.log10_g_range))
        residual_ratio_series.append(
            float(diagnostics.get("residual_ratio", float("nan")))
        )

        if sigma_lx is None or sigma_lx <= 0.0:
            g_eff.append(float("nan"))
            sigma_x.append(float("nan"))
            reason = str(diagnostics.get("failure_reason", "solve_failed"))
            failure_by_eps.append(reason)
            continue

        geff_s = float(sigma_lx * lx)
        g_eff.append(geff_s)
        sigma_x.append(float(geff_s * lx / (ly * lz) * 1.0e9))
        failure_by_eps.append("")

    g_arr = np.asarray(g_eff, dtype=np.float64)
    sigma_arr = np.asarray(sigma_x, dtype=np.float64)
    slope_g, n_g, r2_g, max_resid_g = _linear_fit_log(eps_arr, g_arr)
    slope_sigma, n_sigma, r2_sigma, max_resid_sigma = _linear_fit_log(
        eps_arr,
        sigma_arr,
    )
    geom = np.asarray(
        [
            (l_rve_nm * (1.0 + float(e)))
            / ((l_rve_nm * (1.0 - poisson_nu * float(e))) ** 2)
            for e in eps_arr
        ],
        dtype=np.float64,
    )
    slope_geom, _, _, _ = _linear_fit_log(eps_arr, geom)

    gf_r = -slope_g if np.isfinite(slope_g) else float("nan")
    gf_sigma = -slope_sigma if np.isfinite(slope_sigma) else float("nan")
    gf_geom = slope_geom if np.isfinite(slope_geom) else float("nan")
    gf_r_from_sigma = (
        gf_sigma + gf_geom
        if np.isfinite(gf_sigma) and np.isfinite(gf_geom)
        else float("nan")
    )
    gf_r_minus_recon = (
        gf_r - gf_r_from_sigma
        if np.isfinite(gf_r) and np.isfinite(gf_r_from_sigma)
        else float("nan")
    )
    success = bool(
        np.isfinite(gf_r)
        and np.isfinite(gf_sigma)
        and n_g >= 3
        and n_sigma >= 3
    )
    failure_reason = "" if success else "too_few_valid_strain_points"
    if not success:
        joined = "|".join(reason for reason in failure_by_eps if reason)
        failure_reason = joined or failure_reason

    d = cb0.diameters
    lengths = cnt0.lengths
    n_seg = cnt0.n_seg_per_cnt
    return SimmonsGFResult(
        success=success,
        failure_reason=failure_reason,
        failure_stage="" if success else "gf_regression",
        n_strain_ok=min(n_g, n_sigma),
        gf_r=gf_r,
        gf_sigma=gf_sigma,
        gf_geom=gf_geom,
        gf_r_from_sigma=gf_r_from_sigma,
        gf_r_minus_recon=gf_r_minus_recon,
        r2_lng_vs_eps=r2_g,
        max_resid_lng=max_resid_g,
        r2_lnsigma_vs_eps=r2_sigma,
        max_resid_lnsigma=max_resid_sigma,
        g_eff_eps0_s=_series_value(g_eff, 0),
        g_eff_epsmax_s=_series_value(g_eff, -1),
        sigma_x_eps0_s_per_m=_series_value(sigma_x, 0),
        sigma_x_epsmax_s_per_m=_series_value(sigma_x, -1),
        topology_switch_fraction=_topology_switch_fraction(n_edges_series),
        log10_g_range_max=_max_finite(log10_g_range_series),
        residual_ratio_max=_max_finite(residual_ratio_series),
        opening_alpha_ss_nm_per_strain=float(opening_alpha_ss_nm_per_strain),
        opening_alpha_sc_nm_per_strain=float(opening_alpha_sc_nm_per_strain),
        opening_alpha_cc_nm_per_strain=float(opening_alpha_cc_nm_per_strain),
        power_frac_ss_eps0=_series_value(power_frac_ss_series, 0),
        power_frac_sc_eps0=_series_value(power_frac_sc_series, 0),
        power_frac_cc_eps0=_series_value(power_frac_cc_series, 0),
        power_frac_ss_epsmax=_series_value(power_frac_ss_series, -1),
        power_frac_sc_epsmax=_series_value(power_frac_sc_series, -1),
        power_frac_cc_epsmax=_series_value(power_frac_cc_series, -1),
        phi_cb_actual_vol=cb0.phi_actual_vol,
        phi_cnt_actual_vol=cnt0.phi_actual_vol,
        n_cb=int(len(cb0.diameters)),
        n_cnt=int(len(cnt0.lengths)),
        n_total_segs=int(len(cnt0.seg_starts)),
        mean_d_realized_nm=float(np.mean(d)) if len(d) else float("nan"),
        std_d_realized_nm=float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
        mean_L_realized_nm=float(np.mean(lengths)) if len(lengths) else float("nan"),
        std_L_realized_nm=float(np.std(lengths, ddof=1)) if len(lengths) > 1 else 0.0,
        mean_n_seg_per_cnt=float(np.mean(n_seg)) if len(n_seg) else float("nan"),
        max_n_seg_per_cnt=int(np.max(n_seg)) if len(n_seg) else 0,
        strain_grid=tuple(float(e) for e in eps_arr),
        g_eff_series_s=tuple(float(x) for x in g_eff),
        sigma_x_series_s_per_m=tuple(float(x) for x in sigma_x),
        n_edges_series=tuple(float(x) for x in n_edges_series),
        power_total_series=tuple(float(x) for x in power_total_series),
        power_frac_ss_series=tuple(float(x) for x in power_frac_ss_series),
        power_frac_sc_series=tuple(float(x) for x in power_frac_sc_series),
        power_frac_cc_series=tuple(float(x) for x in power_frac_cc_series),
        largest_component_fraction_series=tuple(float(x) for x in lcf_series),
        log10_g_range_series=tuple(float(x) for x in log10_g_range_series),
        residual_ratio_series=tuple(float(x) for x in residual_ratio_series),
        failure_by_eps=tuple(failure_by_eps),
        elapsed_s=time.time() - t0,
    )


def _failed_result(
    *,
    reason: str,
    stage: str,
    strain_grid: tuple[float, ...],
    elapsed_s: float,
) -> SimmonsGFResult:
    nan_series = tuple(float("nan") for _ in strain_grid)
    return SimmonsGFResult(
        success=False,
        failure_reason=reason,
        failure_stage=stage,
        n_strain_ok=0,
        gf_r=float("nan"),
        gf_sigma=float("nan"),
        gf_geom=float("nan"),
        gf_r_from_sigma=float("nan"),
        gf_r_minus_recon=float("nan"),
        r2_lng_vs_eps=float("nan"),
        max_resid_lng=float("nan"),
        r2_lnsigma_vs_eps=float("nan"),
        max_resid_lnsigma=float("nan"),
        g_eff_eps0_s=float("nan"),
        g_eff_epsmax_s=float("nan"),
        sigma_x_eps0_s_per_m=float("nan"),
        sigma_x_epsmax_s_per_m=float("nan"),
        topology_switch_fraction=float("nan"),
        log10_g_range_max=float("nan"),
        residual_ratio_max=float("nan"),
        opening_alpha_ss_nm_per_strain=0.0,
        opening_alpha_sc_nm_per_strain=0.0,
        opening_alpha_cc_nm_per_strain=0.0,
        power_frac_ss_eps0=float("nan"),
        power_frac_sc_eps0=float("nan"),
        power_frac_cc_eps0=float("nan"),
        power_frac_ss_epsmax=float("nan"),
        power_frac_sc_epsmax=float("nan"),
        power_frac_cc_epsmax=float("nan"),
        phi_cb_actual_vol=0.0,
        phi_cnt_actual_vol=0.0,
        n_cb=0,
        n_cnt=0,
        n_total_segs=0,
        mean_d_realized_nm=float("nan"),
        std_d_realized_nm=float("nan"),
        mean_L_realized_nm=float("nan"),
        std_L_realized_nm=float("nan"),
        mean_n_seg_per_cnt=float("nan"),
        max_n_seg_per_cnt=0,
        strain_grid=strain_grid,
        g_eff_series_s=nan_series,
        sigma_x_series_s_per_m=nan_series,
        n_edges_series=nan_series,
        power_total_series=nan_series,
        power_frac_ss_series=nan_series,
        power_frac_sc_series=nan_series,
        power_frac_cc_series=nan_series,
        largest_component_fraction_series=nan_series,
        log10_g_range_series=nan_series,
        residual_ratio_series=nan_series,
        failure_by_eps=tuple(reason for _ in strain_grid),
        elapsed_s=elapsed_s,
    )


def _simmons_gf_task(kwargs: dict) -> SimmonsGFResult:
    """Module-level worker so process pools can pickle ensemble tasks."""

    return run_simmons_gf_realization(**kwargs)


def run_simmons_gf_ensemble(
    *,
    cb_model: TruncatedLognormal,
    cnt_model: TruncatedLognormal,
    phi_cb_target_vol: float,
    phi_cnt_target_vol: float,
    l_rve_nm: float,
    cnt_diam_nm: float,
    waviness: float,
    tunnel_nm: float,
    seg_len_unit_nm: float,
    d_min_nm: float,
    g_cutoff_s: float,
    strain_grid: tuple[float, ...],
    poisson_nu: float,
    seed_base: int,
    n_realizations: int,
    opening_alpha_ss_nm_per_strain: float = 0.0,
    opening_alpha_sc_nm_per_strain: float = 0.0,
    opening_alpha_cc_nm_per_strain: float = 0.0,
    n_cb_max: int | None = None,
    n_cnt_max: int | None = None,
    direction: int = 0,
    workers: int = 1,
    on_result=None,
) -> list[SimmonsGFResult]:
    """Gauge-factor ensemble with one independent seed per realization.

    Results come back in realization order (``seed_base + r``), so the
    numbers are bit-identical for any ``workers`` value.
    """

    tasks = [
        dict(
            cb_model=cb_model,
            cnt_model=cnt_model,
            phi_cb_target_vol=phi_cb_target_vol,
            phi_cnt_target_vol=phi_cnt_target_vol,
            l_rve_nm=l_rve_nm,
            cnt_diam_nm=cnt_diam_nm,
            waviness=waviness,
            tunnel_nm=tunnel_nm,
            seg_len_unit_nm=seg_len_unit_nm,
            d_min_nm=d_min_nm,
            g_cutoff_s=g_cutoff_s,
            strain_grid=strain_grid,
            poisson_nu=poisson_nu,
            seed=seed_base + r,
            opening_alpha_ss_nm_per_strain=opening_alpha_ss_nm_per_strain,
            opening_alpha_sc_nm_per_strain=opening_alpha_sc_nm_per_strain,
            opening_alpha_cc_nm_per_strain=opening_alpha_cc_nm_per_strain,
            n_cb_max=n_cb_max,
            n_cnt_max=n_cnt_max,
            direction=direction,
        )
        for r in range(n_realizations)
    ]
    return _run_ensemble_tasks(_simmons_gf_task, tasks, workers, on_result)
