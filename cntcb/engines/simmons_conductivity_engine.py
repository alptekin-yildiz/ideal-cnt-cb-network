"""Simmons-weighted conductivity engine (behind ``generate/conductivity.py``).

This module is the physical-weight extension of ``conductivity_engine``.  It
keeps the same sampling conventions and node semantics as the
unit-conductance validation runs:

* one graph node per CB aggregate;
* one graph node per CNT object;
* CNT segments are used only for geometric distance queries;
* edge topology follows the same penetrable tunneling-neighborhood rule.

The difference is that each accepted edge receives a Simmons conductance in SI
units [S], with a local effective distance clamp and a weak-edge cutoff.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from cntcb.kernel.constants import ELECTRON_CHARGE, ELECTRON_MASS, HBAR
from cntcb.kernel.percolation_finder import _SpatialHash, _batch_seg_dist

from .conductivity_engine import (
    CBPopulation,
    CNTPopulation,
    sample_cb_population,
    sample_cnt_population,
    solve_step_network,
    _point_to_segments_dist,
)
from .percolation_engine import TruncatedLognormal, _run_ensemble_tasks


@dataclass(frozen=True)
class JunctionParams:
    """Simmons parameters for one junction class."""

    barrier_eV: float
    area_nm2: float


@dataclass(frozen=True)
class ConductanceStats:
    """Edge-weight diagnostics for one Simmons graph."""

    n_finite_g: int
    g_min: float
    g_max: float
    g_dynamic_range: float
    log10_g_range: float
    log10_g_p0: float
    log10_g_p1: float
    log10_g_p5: float
    log10_g_p50: float
    log10_g_p95: float
    log10_g_p99: float
    log10_g_p100: float
    n_clamped_d_min: int
    frac_clamped_d_min: float
    n_overlap_edges: int
    frac_overlap_edges: float
    n_cutoff_edges: int
    frac_cutoff_edges: float


@dataclass(frozen=True)
class SimmonsConductivityResult:
    """One fixed-composition Simmons-network conductivity realization."""

    success: bool
    failure_reason: str
    failure_stage: str
    sigma_s_per_nm: float
    sigma_s_per_m: float
    effective_conductance_s: float
    phi_cb_actual_vol: float
    phi_cnt_actual_vol: float
    n_cb: int
    n_cnt: int
    n_total_segs: int
    n_nodes: int
    n_edges: int
    n_edges_ss: int
    n_edges_sc: int
    n_edges_cc: int
    power_frac_ss: float
    power_frac_sc: float
    power_frac_cc: float
    n_components: int
    largest_component_size: int
    largest_component_fraction: float
    n_spanning_components: int
    low_nodes: int
    high_nodes: int
    n_unknown: int
    matrix_nnz: int
    residual_ratio: float
    g_min: float
    g_max: float
    g_dynamic_range: float
    log10_g_range: float
    log10_g_p0: float
    log10_g_p1: float
    log10_g_p5: float
    log10_g_p50: float
    log10_g_p95: float
    log10_g_p99: float
    log10_g_p100: float
    n_clamped_d_min: int
    frac_clamped_d_min: float
    n_overlap_edges: int
    frac_overlap_edges: float
    n_cutoff_edges: int
    frac_cutoff_edges: float
    n_finite_g: int
    mean_d_realized_nm: float
    std_d_realized_nm: float
    mean_L_realized_nm: float
    std_L_realized_nm: float
    mean_n_seg_per_cnt: float
    max_n_seg_per_cnt: int
    elapsed_s: float


DEFAULT_BARRIER_EV = 0.35
PARAMS_CNT_CNT = JunctionParams(DEFAULT_BARRIER_EV, 90.0)
PARAMS_CB_CB = JunctionParams(DEFAULT_BARRIER_EV, 145.0)
PARAMS_CB_CNT = JunctionParams(DEFAULT_BARRIER_EV, math.sqrt(90.0 * 145.0))


def simmons_conductance(
    d_surf_nm: float,
    *,
    params: JunctionParams,
    d_min_nm: float,
) -> float:
    """Low-bias Simmons conductance [S] with a local distance clamp.

    This implementation fixes one conventional low-bias prefactor,
    A * e^2 * kappa / (4 * pi^2 * hbar * d); other Simmons variants differ
    by an order-one factor. Absolute conductances are therefore defined up
    to an O(1) prefactor convention, while trends and fitted exponents are
    controlled by the exponential distance dependence exp(-2*kappa*d).
    """

    d_nm = max(float(d_surf_nm), float(d_min_nm))
    phi_b = params.barrier_eV * ELECTRON_CHARGE
    d_m = d_nm * 1e-9
    area_m2 = params.area_nm2 * 1e-18
    kappa = math.sqrt(2.0 * ELECTRON_MASS * phi_b) / HBAR
    prefactor = (
        area_m2
        * ELECTRON_CHARGE**2
        * kappa
        / (4.0 * math.pi**2 * HBAR * d_m)
    )
    return float(prefactor * math.exp(-2.0 * kappa * d_m))


def build_simmons_conductance_network(
    *,
    cb: CBPopulation,
    cnt: CNTPopulation,
    cb_model: TruncatedLognormal,
    cnt_diam_nm: float,
    tunnel_nm: float,
    seg_len_unit_nm: float,
    d_min_nm: float,
    g_cutoff_s: float,
    params_ss: JunctionParams = PARAMS_CB_CB,
    params_cc: JunctionParams = PARAMS_CNT_CNT,
    params_sc: JunctionParams = PARAMS_CB_CNT,
    distance_offset_ss_nm: float = 0.0,
    distance_offset_cc_nm: float = 0.0,
    distance_offset_sc_nm: float = 0.0,
) -> tuple[sparse.csr_matrix, ConductanceStats]:
    """Build the Simmons-weighted CB/CNT conductance matrix."""

    n_cb = len(cb.diameters)
    n_cnt = len(cnt.lengths)
    n_total = n_cb + n_cnt
    if n_total == 0:
        return sparse.csr_matrix((0, 0)), _empty_stats()

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    kept_g: list[float] = []
    n_clamped = 0
    n_overlap = 0
    n_cutoff = 0
    n_candidates = 0
    r_cnt = cnt_diam_nm / 2.0

    def add_edge(
        i: int,
        j: int,
        d_surf_nm: float,
        params: JunctionParams,
        distance_offset_nm: float,
    ) -> None:
        nonlocal n_clamped, n_overlap, n_cutoff, n_candidates
        n_candidates += 1
        d_eff_nm = d_surf_nm + distance_offset_nm
        g = simmons_conductance(d_eff_nm, params=params, d_min_nm=d_min_nm)
        if g < g_cutoff_s:
            n_cutoff += 1
            return
        if d_eff_nm < d_min_nm:
            n_clamped += 1
        if d_surf_nm < 0.0:
            n_overlap += 1
        rows.extend((i, j))
        cols.extend((j, i))
        vals.extend((g, g))
        kept_g.append(g)

    if n_cb > 1:
        cb_hash = _SpatialHash(cell_size=cb_model.upper_nm + tunnel_nm)
        for i, center in enumerate(cb.centers):
            for j_raw in cb_hash.query(center):
                j = int(j_raw)
                center_dist = float(np.linalg.norm(center - cb.centers[j]))
                contact_dist = 0.5 * (cb.diameters[i] + cb.diameters[j])
                if center_dist < contact_dist + tunnel_nm:
                    add_edge(
                        i,
                        j,
                        center_dist - contact_dist,
                        params_ss,
                        distance_offset_ss_nm,
                    )
            cb_hash.insert(i, center)

    if n_cnt > 1:
        counters = _append_cnt_cnt_simmons_edges(
            rows=rows,
            cols=cols,
            vals=vals,
            kept_g=kept_g,
            n_cb=n_cb,
            cnt=cnt,
            r_cnt=r_cnt,
            tunnel_nm=tunnel_nm,
            seg_len_unit_nm=seg_len_unit_nm,
            d_min_nm=d_min_nm,
            g_cutoff_s=g_cutoff_s,
            params=params_cc,
            distance_offset_nm=distance_offset_cc_nm,
        )
        n_candidates += counters[0]
        n_clamped += counters[1]
        n_overlap += counters[2]
        n_cutoff += counters[3]

    if n_cb > 0 and n_cnt > 0:
        counters = _append_cb_cnt_simmons_edges(
            rows=rows,
            cols=cols,
            vals=vals,
            kept_g=kept_g,
            cb=cb,
            cnt=cnt,
            n_cb=n_cb,
            r_cnt=r_cnt,
            tunnel_nm=tunnel_nm,
            cell_size=seg_len_unit_nm + 0.5 * cb_model.upper_nm + r_cnt + tunnel_nm,
            d_min_nm=d_min_nm,
            g_cutoff_s=g_cutoff_s,
            params=params_sc,
            distance_offset_nm=distance_offset_sc_nm,
        )
        n_candidates += counters[0]
        n_clamped += counters[1]
        n_overlap += counters[2]
        n_cutoff += counters[3]

    if not rows:
        return sparse.csr_matrix((n_total, n_total)), _stats_from_edges(
            kept_g,
            n_clamped=n_clamped,
            n_overlap=n_overlap,
            n_cutoff=n_cutoff,
            n_candidates=n_candidates,
        )

    G = sparse.coo_matrix((vals, (rows, cols)), shape=(n_total, n_total)).tocsr()
    return G, _stats_from_edges(
        kept_g,
        n_clamped=n_clamped,
        n_overlap=n_overlap,
        n_cutoff=n_cutoff,
        n_candidates=n_candidates,
    )


def _append_cnt_cnt_simmons_edges(
    *,
    rows: list[int],
    cols: list[int],
    vals: list[float],
    kept_g: list[float],
    n_cb: int,
    cnt: CNTPopulation,
    r_cnt: float,
    tunnel_nm: float,
    seg_len_unit_nm: float,
    d_min_nm: float,
    g_cutoff_s: float,
    params: JunctionParams,
    distance_offset_nm: float,
) -> tuple[int, int, int, int]:
    local = [0, 0, 0, 0]
    mids = 0.5 * (cnt.seg_starts + cnt.seg_ends)
    cell_size = seg_len_unit_nm + 2.0 * r_cnt + tunnel_nm
    seg_hash = _SpatialHash(cell_size=cell_size)
    for flat_id, mid in enumerate(mids):
        seg_hash.insert(flat_id, mid)

    n_cnt = len(cnt.lengths)
    for i in range(n_cnt):
        o0 = int(cnt.seg_offsets[i])
        o1 = int(cnt.seg_offsets[i + 1])
        cand_set: set[int] = set()
        for mid in mids[o0:o1]:
            for flat_id in seg_hash.query(mid):
                j = int(cnt.seg_to_cnt[flat_id])
                if j > i:
                    cand_set.add(j)
        if not cand_set:
            continue

        cand_arr = np.fromiter(sorted(cand_set), dtype=np.intp)
        lengths = cnt.seg_offsets[cand_arr + 1] - cnt.seg_offsets[cand_arr]
        starts = np.empty(len(cand_arr), dtype=np.intp)
        starts[0] = 0
        if len(cand_arr) > 1:
            np.cumsum(lengths[:-1], out=starts[1:])

        idx_parts = [
            np.arange(int(cnt.seg_offsets[j]), int(cnt.seg_offsets[j + 1]), dtype=np.intp)
            for j in cand_arr
        ]
        idx = np.concatenate(idx_parts)
        cand_starts = cnt.seg_starts[idx]
        cand_ends = cnt.seg_ends[idx]
        min_axis = np.full(len(cand_arr), np.inf, dtype=np.float64)

        for s in range(o0, o1):
            dists = _batch_seg_dist(
                cnt.seg_starts[s],
                cnt.seg_ends[s],
                cand_starts,
                cand_ends,
            )
            min_axis = np.minimum(min_axis, np.minimum.reduceat(dists, starts))

        for pos in np.flatnonzero(min_axis < 2.0 * r_cnt + tunnel_nm):
            j = int(cand_arr[pos])
            d_surf = float(min_axis[pos] - 2.0 * r_cnt)
            local[0] += 1
            d_eff = d_surf + distance_offset_nm
            g = simmons_conductance(d_eff, params=params, d_min_nm=d_min_nm)
            if g < g_cutoff_s:
                local[3] += 1
                continue
            if d_eff < d_min_nm:
                local[1] += 1
            if d_surf < 0.0:
                local[2] += 1
            ni = n_cb + i
            nj = n_cb + j
            rows.extend((ni, nj))
            cols.extend((nj, ni))
            vals.extend((g, g))
            kept_g.append(g)

    return tuple(local)


def _append_cb_cnt_simmons_edges(
    *,
    rows: list[int],
    cols: list[int],
    vals: list[float],
    kept_g: list[float],
    cb: CBPopulation,
    cnt: CNTPopulation,
    n_cb: int,
    r_cnt: float,
    tunnel_nm: float,
    cell_size: float,
    d_min_nm: float,
    g_cutoff_s: float,
    params: JunctionParams,
    distance_offset_nm: float,
) -> tuple[int, int, int, int]:
    local = [0, 0, 0, 0]
    mids = 0.5 * (cnt.seg_starts + cnt.seg_ends)
    seg_hash = _SpatialHash(cell_size=cell_size)
    for flat_id, mid in enumerate(mids):
        seg_hash.insert(flat_id, mid)

    for i, center in enumerate(cb.centers):
        cand = sorted(set(int(fid) for fid in seg_hash.query(center)))
        if not cand:
            continue
        cand_arr = np.asarray(cand, dtype=np.intp)
        dists = _point_to_segments_dist(center, cnt.seg_starts[cand_arr], cnt.seg_ends[cand_arr])
        surface = dists - (0.5 * cb.diameters[i] + r_cnt)
        connected = surface < tunnel_nm
        if not np.any(connected):
            continue

        cand_cnt = cnt.seg_to_cnt[cand_arr[connected]]
        cand_surface = surface[connected]
        order = np.argsort(cand_cnt)
        sorted_cnt = cand_cnt[order]
        sorted_surface = cand_surface[order]
        change = np.r_[0, np.flatnonzero(np.diff(sorted_cnt)) + 1]
        per_cnt_min = np.minimum.reduceat(sorted_surface, change)
        per_cnt_ids = sorted_cnt[change]

        for j_raw, d_surf_raw in zip(per_cnt_ids, per_cnt_min):
            d_surf = float(d_surf_raw)
            local[0] += 1
            d_eff = d_surf + distance_offset_nm
            g = simmons_conductance(d_eff, params=params, d_min_nm=d_min_nm)
            if g < g_cutoff_s:
                local[3] += 1
                continue
            if d_eff < d_min_nm:
                local[1] += 1
            if d_surf < 0.0:
                local[2] += 1
            ni = i
            nj = n_cb + int(j_raw)
            rows.extend((ni, nj))
            cols.extend((nj, ni))
            vals.extend((g, g))
            kept_g.append(g)

    return tuple(local)


def _stats_from_edges(
    conductances: list[float],
    *,
    n_clamped: int,
    n_overlap: int,
    n_cutoff: int,
    n_candidates: int,
) -> ConductanceStats:
    if not conductances:
        return ConductanceStats(
            n_finite_g=0,
            g_min=float("nan"),
            g_max=float("nan"),
            g_dynamic_range=float("nan"),
            log10_g_range=float("nan"),
            log10_g_p0=float("nan"),
            log10_g_p1=float("nan"),
            log10_g_p5=float("nan"),
            log10_g_p50=float("nan"),
            log10_g_p95=float("nan"),
            log10_g_p99=float("nan"),
            log10_g_p100=float("nan"),
            n_clamped_d_min=0,
            frac_clamped_d_min=0.0,
            n_overlap_edges=0,
            frac_overlap_edges=0.0,
            n_cutoff_edges=n_cutoff,
            frac_cutoff_edges=n_cutoff / n_candidates if n_candidates else 0.0,
        )

    g = np.asarray(conductances, dtype=np.float64)
    logg = np.log10(g)
    g_min = float(np.min(g))
    g_max = float(np.max(g))
    return ConductanceStats(
        n_finite_g=int(len(g)),
        g_min=g_min,
        g_max=g_max,
        g_dynamic_range=float(g_max / g_min) if g_min > 0.0 else float("inf"),
        log10_g_range=float(np.max(logg) - np.min(logg)),
        log10_g_p0=float(np.percentile(logg, 0)),
        log10_g_p1=float(np.percentile(logg, 1)),
        log10_g_p5=float(np.percentile(logg, 5)),
        log10_g_p50=float(np.percentile(logg, 50)),
        log10_g_p95=float(np.percentile(logg, 95)),
        log10_g_p99=float(np.percentile(logg, 99)),
        log10_g_p100=float(np.percentile(logg, 100)),
        n_clamped_d_min=int(n_clamped),
        frac_clamped_d_min=float(n_clamped / len(g)),
        n_overlap_edges=int(n_overlap),
        frac_overlap_edges=float(n_overlap / len(g)),
        n_cutoff_edges=int(n_cutoff),
        frac_cutoff_edges=n_cutoff / n_candidates if n_candidates else 0.0,
    )


def _empty_stats() -> ConductanceStats:
    return _stats_from_edges([], n_clamped=0, n_overlap=0, n_cutoff=0, n_candidates=0)


def run_simmons_conductivity_realization(
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
    seed: int,
    n_cb_max: int | None = None,
    n_cnt_max: int | None = None,
    direction: int = 0,
    barrier_eV: float | None = None,
) -> SimmonsConductivityResult:
    """Sample one RVE and solve the Simmons-weighted Kirchhoff problem.

    ``barrier_eV`` overrides the tunneling barrier height for all junction
    classes (patch areas unchanged). ``None`` keeps the declared production
    constants (DEFAULT_BARRIER_EV); a value is used only for the barrier
    sensitivity study and never for production figures.
    """

    t0 = time.time()
    rng = np.random.default_rng(seed)
    try:
        cb = sample_cb_population(
            rng=rng,
            cb_model=cb_model,
            phi_target_vol=phi_cb_target_vol,
            l_rve_nm=l_rve_nm,
            n_max=n_cb_max,
        )
        cnt = sample_cnt_population(
            rng=rng,
            cnt_model=cnt_model,
            phi_target_vol=phi_cnt_target_vol,
            l_rve_nm=l_rve_nm,
            cnt_diam_nm=cnt_diam_nm,
            waviness=waviness,
            seg_len_unit_nm=seg_len_unit_nm,
            n_max=n_cnt_max,
        )
        if barrier_eV is None:
            pss, pcc, psc = PARAMS_CB_CB, PARAMS_CNT_CNT, PARAMS_CB_CNT
        else:
            pss = JunctionParams(barrier_eV, PARAMS_CB_CB.area_nm2)
            pcc = JunctionParams(barrier_eV, PARAMS_CNT_CNT.area_nm2)
            psc = JunctionParams(barrier_eV, PARAMS_CB_CNT.area_nm2)
        G, g_stats = build_simmons_conductance_network(
            cb=cb,
            cnt=cnt,
            cb_model=cb_model,
            cnt_diam_nm=cnt_diam_nm,
            tunnel_nm=tunnel_nm,
            seg_len_unit_nm=seg_len_unit_nm,
            d_min_nm=d_min_nm,
            g_cutoff_s=g_cutoff_s,
            params_ss=pss,
            params_cc=pcc,
            params_sc=psc,
        )
        sigma, diagnostics = solve_step_network(
            G=G,
            cb=cb,
            cnt=cnt,
            cnt_diam_nm=cnt_diam_nm,
            l_rve_nm=l_rve_nm,
            direction=direction,
        )
    except Exception as exc:
        return _finalize_result(
            success=False,
            failure_reason=f"exception:{type(exc).__name__}",
            failure_stage="exception",
            sigma=None,
            g_eff=0.0,
            cb=None,
            cnt=None,
            diagnostics={},
            g_stats=_empty_stats(),
            t0=t0,
        )

    success = sigma is not None and sigma > 0.0
    return _finalize_result(
        success=success,
        failure_reason="" if success else str(diagnostics.get("failure_reason", "solve_failed")),
        failure_stage="" if success else str(diagnostics.get("failure_stage", "solve")),
        sigma=sigma,
        g_eff=0.0 if sigma is None else float(sigma * l_rve_nm),
        cb=cb,
        cnt=cnt,
        diagnostics=diagnostics,
        g_stats=g_stats,
        t0=t0,
    )


def _finalize_result(
    *,
    success: bool,
    failure_reason: str,
    failure_stage: str,
    sigma: float | None,
    g_eff: float,
    cb: CBPopulation | None,
    cnt: CNTPopulation | None,
    diagnostics: dict[str, float | int | str],
    g_stats: ConductanceStats,
    t0: float,
) -> SimmonsConductivityResult:
    n_cb = 0 if cb is None else len(cb.diameters)
    n_cnt = 0 if cnt is None else len(cnt.lengths)
    n_total_segs = 0 if cnt is None else len(cnt.seg_starts)
    d = np.empty(0) if cb is None else cb.diameters
    lengths = np.empty(0) if cnt is None else cnt.lengths
    n_seg = np.empty(0, dtype=np.intp) if cnt is None else cnt.n_seg_per_cnt
    sigma_s_per_nm = float(sigma) if success and sigma is not None else 0.0

    return SimmonsConductivityResult(
        success=success,
        failure_reason=failure_reason,
        failure_stage=failure_stage,
        sigma_s_per_nm=sigma_s_per_nm,
        sigma_s_per_m=sigma_s_per_nm * 1.0e9,
        effective_conductance_s=g_eff if success else 0.0,
        phi_cb_actual_vol=0.0 if cb is None else cb.phi_actual_vol,
        phi_cnt_actual_vol=0.0 if cnt is None else cnt.phi_actual_vol,
        n_cb=n_cb,
        n_cnt=n_cnt,
        n_total_segs=n_total_segs,
        n_nodes=int(diagnostics.get("n_nodes", n_cb + n_cnt)),
        n_edges=int(diagnostics.get("n_edges", 0)),
        n_edges_ss=int(diagnostics.get("n_edges_ss", 0)),
        n_edges_sc=int(diagnostics.get("n_edges_sc", 0)),
        n_edges_cc=int(diagnostics.get("n_edges_cc", 0)),
        power_frac_ss=float(diagnostics.get("power_frac_ss", float("nan"))),
        power_frac_sc=float(diagnostics.get("power_frac_sc", float("nan"))),
        power_frac_cc=float(diagnostics.get("power_frac_cc", float("nan"))),
        n_components=int(diagnostics.get("n_components", 0)),
        largest_component_size=int(diagnostics.get("largest_component_size", 0)),
        largest_component_fraction=float(diagnostics.get("largest_component_fraction", 0.0)),
        n_spanning_components=int(diagnostics.get("n_spanning_components", 0)),
        low_nodes=int(diagnostics.get("low_nodes", 0)),
        high_nodes=int(diagnostics.get("high_nodes", 0)),
        n_unknown=int(diagnostics.get("n_unknown", 0)),
        matrix_nnz=int(diagnostics.get("matrix_nnz", 0)),
        residual_ratio=float(diagnostics.get("residual_ratio", float("nan"))),
        g_min=g_stats.g_min,
        g_max=g_stats.g_max,
        g_dynamic_range=g_stats.g_dynamic_range,
        log10_g_range=g_stats.log10_g_range,
        log10_g_p0=g_stats.log10_g_p0,
        log10_g_p1=g_stats.log10_g_p1,
        log10_g_p5=g_stats.log10_g_p5,
        log10_g_p50=g_stats.log10_g_p50,
        log10_g_p95=g_stats.log10_g_p95,
        log10_g_p99=g_stats.log10_g_p99,
        log10_g_p100=g_stats.log10_g_p100,
        n_clamped_d_min=g_stats.n_clamped_d_min,
        frac_clamped_d_min=g_stats.frac_clamped_d_min,
        n_overlap_edges=g_stats.n_overlap_edges,
        frac_overlap_edges=g_stats.frac_overlap_edges,
        n_cutoff_edges=g_stats.n_cutoff_edges,
        frac_cutoff_edges=g_stats.frac_cutoff_edges,
        n_finite_g=g_stats.n_finite_g,
        mean_d_realized_nm=float(np.mean(d)) if n_cb > 0 else float("nan"),
        std_d_realized_nm=float(np.std(d, ddof=1)) if n_cb > 1 else 0.0,
        mean_L_realized_nm=float(np.mean(lengths)) if n_cnt > 0 else float("nan"),
        std_L_realized_nm=float(np.std(lengths, ddof=1)) if n_cnt > 1 else 0.0,
        mean_n_seg_per_cnt=float(np.mean(n_seg)) if n_cnt > 0 else float("nan"),
        max_n_seg_per_cnt=int(np.max(n_seg)) if n_cnt > 0 else 0,
        elapsed_s=time.time() - t0,
    )


def _simmons_conductivity_task(kwargs: dict) -> SimmonsConductivityResult:
    """Module-level worker so process pools can pickle ensemble tasks."""

    return run_simmons_conductivity_realization(**kwargs)


def run_simmons_conductivity_ensemble(
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
    seed_base: int,
    n_realizations: int,
    n_cb_max: int | None = None,
    n_cnt_max: int | None = None,
    direction: int = 0,
    barrier_eV: float | None = None,
    workers: int = 1,
    on_result=None,
) -> list[SimmonsConductivityResult]:
    """Conductivity ensemble with one independent seed per realization.

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
            seed=seed_base + r,
            n_cb_max=n_cb_max,
            n_cnt_max=n_cnt_max,
            direction=direction,
            barrier_eV=barrier_eV,
        )
        for r in range(n_realizations)
    ]
    return _run_ensemble_tasks(_simmons_conductivity_task, tasks, workers, on_result)
