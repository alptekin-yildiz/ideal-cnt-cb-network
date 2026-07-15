"""Step-network (binary-conductance) conductivity engine.

Solves a Kirchhoff network with G_ij = 1 for every connected junction --
the conductance-agnostic limit used for the universality validation runs.
It reuses the sampling conventions of ``percolation_engine``. For the
physical Simmons-weighted conductivity behind ``generate/conductivity.py``,
see ``simmons_conductivity_engine``.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from cntcb.kernel.percolation_finder import _SpatialHash, _batch_seg_dist

from .percolation_engine import (
    TruncatedLognormal,
    _draw_cnt_segments,
    estimate_cb_n_max,
    estimate_cnt_n_max,
    sample_truncated_lognormal,
)


@dataclass(frozen=True)
class StepConductivityResult:
    """One fixed-composition step-network conductivity realization."""

    success: bool
    failure_reason: str
    failure_stage: str
    sigma_uncond: float
    effective_conductance: float
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
    n_components: int
    largest_component_size: int
    largest_component_fraction: float
    n_spanning_components: int
    low_nodes: int
    high_nodes: int
    n_unknown: int
    matrix_nnz: int
    residual_ratio: float
    mean_d_realized_nm: float
    std_d_realized_nm: float
    mean_L_realized_nm: float
    std_L_realized_nm: float
    mean_n_seg_per_cnt: float
    max_n_seg_per_cnt: int
    elapsed_s: float


@dataclass(frozen=True)
class CBPopulation:
    centers: np.ndarray
    diameters: np.ndarray
    phi_actual_vol: float


@dataclass(frozen=True)
class CNTPopulation:
    seg_starts: np.ndarray
    seg_ends: np.ndarray
    seg_offsets: np.ndarray
    seg_to_cnt: np.ndarray
    lengths: np.ndarray
    n_seg_per_cnt: np.ndarray
    phi_actual_vol: float


@dataclass(frozen=True)
class ComponentSolveResult:
    """Numerical diagnostics for one spanning connected component."""

    current: float
    residual_ratio: float
    n_unknown: int
    matrix_nnz: int
    power_total: float
    power_ss: float
    power_sc: float
    power_cc: float


class PopulationCapError(RuntimeError):
    """Raised when the sampling cap is reached before the target volume."""

    def __init__(self, filler: str) -> None:
        self.filler = filler
        super().__init__(f"{filler} population reached n_max before phi_target.")


def sample_cb_population(
    *,
    rng: np.random.Generator,
    cb_model: TruncatedLognormal,
    phi_target_vol: float,
    l_rve_nm: float,
    n_max: int | None = None,
) -> CBPopulation:
    """Draw CB aggregates until the requested volume fraction is reached."""

    if phi_target_vol <= 0.0:
        return CBPopulation(
            centers=np.empty((0, 3), dtype=np.float64),
            diameters=np.empty(0, dtype=np.float64),
            phi_actual_vol=0.0,
        )

    n_cap = n_max or estimate_cb_n_max(cb_model, phi_target_vol, l_rve_nm)
    centers = np.empty((n_cap, 3), dtype=np.float64)
    diameters = np.empty(n_cap, dtype=np.float64)
    vol_total = 0.0
    v_target = phi_target_vol * l_rve_nm**3
    n_cb = 0

    for idx in range(n_cap):
        d_i = sample_truncated_lognormal(rng, cb_model)
        centers[idx] = rng.uniform(0.0, l_rve_nm, size=3)
        diameters[idx] = d_i
        vol_total += (math.pi / 6.0) * d_i**3
        n_cb += 1
        if vol_total >= v_target:
            break
    else:
        raise PopulationCapError("CB")

    return CBPopulation(
        centers=centers[:n_cb].copy(),
        diameters=diameters[:n_cb].copy(),
        phi_actual_vol=float(vol_total / l_rve_nm**3),
    )


def sample_cnt_population(
    *,
    rng: np.random.Generator,
    cnt_model: TruncatedLognormal,
    phi_target_vol: float,
    l_rve_nm: float,
    cnt_diam_nm: float,
    waviness: float,
    seg_len_unit_nm: float,
    n_max: int | None = None,
) -> CNTPopulation:
    """Draw CNTs until the requested volume fraction is reached."""

    if phi_target_vol <= 0.0:
        return CNTPopulation(
            seg_starts=np.empty((0, 3), dtype=np.float64),
            seg_ends=np.empty((0, 3), dtype=np.float64),
            seg_offsets=np.zeros(1, dtype=np.intp),
            seg_to_cnt=np.empty(0, dtype=np.intp),
            lengths=np.empty(0, dtype=np.float64),
            n_seg_per_cnt=np.empty(0, dtype=np.intp),
            phi_actual_vol=0.0,
        )

    n_cap = n_max or estimate_cnt_n_max(
        cnt_model, phi_target_vol, l_rve_nm, cnt_diam_nm
    )
    r_cnt = cnt_diam_nm / 2.0
    v_prefactor = math.pi * r_cnt**2
    v_target = phi_target_vol * l_rve_nm**3

    starts_parts: list[np.ndarray] = []
    ends_parts: list[np.ndarray] = []
    lengths: list[float] = []
    n_seg_per_cnt: list[int] = []
    seg_offsets = [0]
    vol_total = 0.0

    for cnt_id in range(n_cap):
        length_i = sample_truncated_lognormal(rng, cnt_model)
        seg_s, seg_e, seg_lens = _draw_cnt_segments(
            rng=rng,
            length_nm=length_i,
            l_rve_nm=l_rve_nm,
            radius_nm=r_cnt,
            waviness=waviness,
            seg_len_unit_nm=seg_len_unit_nm,
        )
        starts_parts.append(seg_s)
        ends_parts.append(seg_e)
        lengths.append(length_i)
        n_seg_per_cnt.append(int(len(seg_lens)))
        seg_offsets.append(seg_offsets[-1] + int(len(seg_lens)))
        vol_total += v_prefactor * length_i
        if vol_total >= v_target:
            break
    else:
        raise PopulationCapError("CNT")

    starts = np.vstack(starts_parts)
    ends = np.vstack(ends_parts)
    counts = np.asarray(n_seg_per_cnt, dtype=np.intp)
    seg_to_cnt = np.repeat(np.arange(len(counts), dtype=np.intp), counts)

    return CNTPopulation(
        seg_starts=starts,
        seg_ends=ends,
        seg_offsets=np.asarray(seg_offsets, dtype=np.intp),
        seg_to_cnt=seg_to_cnt,
        lengths=np.asarray(lengths, dtype=np.float64),
        n_seg_per_cnt=counts,
        phi_actual_vol=float(vol_total / l_rve_nm**3),
    )


def _point_to_segments_dist(
    point: np.ndarray,
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
) -> np.ndarray:
    """Distances from one point to many segment axes."""

    v = seg_ends - seg_starts
    w = point - seg_starts
    denom = np.einsum("ij,ij->i", v, v) + 1e-30
    t = np.clip(np.einsum("ij,ij->i", w, v) / denom, 0.0, 1.0)
    closest = seg_starts + t[:, None] * v
    return np.linalg.norm(point[None, :] - closest, axis=1)


def build_step_conductance_network(
    *,
    cb: CBPopulation,
    cnt: CNTPopulation,
    cb_model: TruncatedLognormal,
    cnt_diam_nm: float,
    tunnel_nm: float,
    seg_len_unit_nm: float,
) -> sparse.csr_matrix:
    """Build the unit-edge CB/CNT conductance matrix."""

    n_cb = len(cb.diameters)
    n_cnt = len(cnt.lengths)
    n_total = n_cb + n_cnt
    if n_total == 0:
        return sparse.csr_matrix((0, 0))

    rows: list[int] = []
    cols: list[int] = []
    r_cnt = cnt_diam_nm / 2.0

    if n_cb > 1:
        cb_hash = _SpatialHash(cell_size=cb_model.upper_nm + tunnel_nm)
        for i, center in enumerate(cb.centers):
            for j in cb_hash.query(center):
                j = int(j)
                connect_d = 0.5 * (cb.diameters[i] + cb.diameters[j]) + tunnel_nm
                if float(np.linalg.norm(center - cb.centers[j])) < connect_d:
                    rows.extend((i, j))
                    cols.extend((j, i))
            cb_hash.insert(i, center)

    if n_cnt > 1:
        _append_cnt_cnt_edges(
            rows=rows,
            cols=cols,
            n_cb=n_cb,
            cnt=cnt,
            connect_d=2.0 * r_cnt + tunnel_nm,
            cell_size=seg_len_unit_nm + 2.0 * r_cnt + tunnel_nm,
        )

    if n_cb > 0 and n_cnt > 0:
        _append_cb_cnt_edges(
            rows=rows,
            cols=cols,
            cb=cb,
            cnt=cnt,
            n_cb=n_cb,
            r_cnt=r_cnt,
            tunnel_nm=tunnel_nm,
            cell_size=seg_len_unit_nm + 0.5 * cb_model.upper_nm + r_cnt + tunnel_nm,
        )

    if not rows:
        return sparse.csr_matrix((n_total, n_total))

    vals = np.ones(len(rows), dtype=np.float64)
    return sparse.coo_matrix((vals, (rows, cols)), shape=(n_total, n_total)).tocsr()


def _append_cnt_cnt_edges(
    *,
    rows: list[int],
    cols: list[int],
    n_cb: int,
    cnt: CNTPopulation,
    connect_d: float,
    cell_size: float,
) -> None:
    mids = 0.5 * (cnt.seg_starts + cnt.seg_ends)
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
        connected = np.zeros(len(cand_arr), dtype=bool)

        for s in range(o0, o1):
            dists = _batch_seg_dist(cnt.seg_starts[s], cnt.seg_ends[s], cand_starts, cand_ends)
            connected |= np.minimum.reduceat(dists, starts) < connect_d

        for j in cand_arr[np.flatnonzero(connected)]:
            ni = n_cb + i
            nj = n_cb + int(j)
            rows.extend((ni, nj))
            cols.extend((nj, ni))


def _append_cb_cnt_edges(
    *,
    rows: list[int],
    cols: list[int],
    cb: CBPopulation,
    cnt: CNTPopulation,
    n_cb: int,
    r_cnt: float,
    tunnel_nm: float,
    cell_size: float,
) -> None:
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
        threshold = cb.diameters[i] / 2.0 + r_cnt + tunnel_nm
        connected_cnts = sorted(set(int(cnt.seg_to_cnt[fid]) for fid in cand_arr[dists < threshold]))
        for j in connected_cnts:
            ni = i
            nj = n_cb + j
            rows.extend((ni, nj))
            cols.extend((nj, ni))


def solve_step_network(
    *,
    G: sparse.csr_matrix,
    cb: CBPopulation,
    cnt: CNTPopulation,
    cnt_diam_nm: float,
    l_rve_nm: float,
    direction: int = 0,
) -> tuple[float | None, dict[str, float | int | str]]:
    """Solve the unit-edge Kirchhoff problem and return sigma = G_eff/L."""

    n_total = G.shape[0]
    n_cb = len(cb.diameters)
    n_edges_ss, n_edges_sc, n_edges_cc = _edge_type_counts(G, n_cb)
    diagnostics: dict[str, float | int | str] = {
        "n_nodes": int(n_total),
        "n_edges": int(n_edges_ss + n_edges_sc + n_edges_cc),
        "n_edges_ss": int(n_edges_ss),
        "n_edges_sc": int(n_edges_sc),
        "n_edges_cc": int(n_edges_cc),
        "n_components": 0,
        "largest_component_size": 0,
        "largest_component_fraction": 0.0,
        "n_spanning_components": 0,
        "low_nodes": 0,
        "high_nodes": 0,
        "n_unknown": 0,
        "matrix_nnz": 0,
        "residual_ratio": float("nan"),
        "power_total": 0.0,
        "power_ss": 0.0,
        "power_sc": 0.0,
        "power_cc": 0.0,
        "power_frac_ss": float("nan"),
        "power_frac_sc": float("nan"),
        "power_frac_cc": float("nan"),
        "failure_reason": "",
        "failure_stage": "",
    }
    if n_total < 2 or G.nnz == 0:
        diagnostics["failure_reason"] = "empty_graph"
        diagnostics["failure_stage"] = "network_build"
        return None, diagnostics

    low_flags, high_flags = _boundary_flags(
        cb=cb,
        cnt=cnt,
        cnt_diam_nm=cnt_diam_nm,
        l_rve_nm=l_rve_nm,
        direction=direction,
    )
    diagnostics["low_nodes"] = int(np.count_nonzero(low_flags))
    diagnostics["high_nodes"] = int(np.count_nonzero(high_flags))
    if not np.any(low_flags) or not np.any(high_flags):
        diagnostics["failure_reason"] = "missing_electrode_nodes"
        diagnostics["failure_stage"] = "boundary_flags"
        return None, diagnostics
    boundary_overlap = low_flags & high_flags
    diagnostics["boundary_overlap_nodes"] = int(np.count_nonzero(boundary_overlap))
    if np.any(boundary_overlap):
        diagnostics["failure_reason"] = "boundary_overlap_short_circuit"
        diagnostics["failure_stage"] = "boundary_flags"
        return None, diagnostics

    n_comp, labels = connected_components(G, directed=False)
    diagnostics["n_components"] = int(n_comp)
    comp_sizes = np.bincount(labels, minlength=n_comp)
    diagnostics["largest_component_size"] = int(comp_sizes.max())
    diagnostics["largest_component_fraction"] = float(comp_sizes.max() / n_total)

    total_current = 0.0
    n_spanning = 0
    n_unknown_total = 0
    matrix_nnz_total = 0
    power_total = 0.0
    power_ss = 0.0
    power_sc = 0.0
    power_cc = 0.0
    residuals: list[float] = []
    solved_any = False
    for comp_id in range(n_comp):
        nodes = np.flatnonzero(labels == comp_id)
        if not np.any(low_flags[nodes]) or not np.any(high_flags[nodes]):
            continue
        solved = _solve_component_current(
            G[np.ix_(nodes, nodes)].tocsr(),
            low_flags[nodes],
            high_flags[nodes],
            node_ids=nodes,
            n_cb_global=n_cb,
        )
        if solved is None:
            diagnostics["failure_reason"] = "linear_solve"
            diagnostics["failure_stage"] = "linear_solve"
            return None, diagnostics
        total_current += solved.current
        n_spanning += 1
        n_unknown_total += solved.n_unknown
        matrix_nnz_total += solved.matrix_nnz
        power_total += solved.power_total
        power_ss += solved.power_ss
        power_sc += solved.power_sc
        power_cc += solved.power_cc
        residuals.append(solved.residual_ratio)
        solved_any = True

    diagnostics["n_spanning_components"] = int(n_spanning)
    diagnostics["n_unknown"] = int(n_unknown_total)
    diagnostics["matrix_nnz"] = int(matrix_nnz_total)
    diagnostics["residual_ratio"] = float(max(residuals)) if residuals else float("nan")
    diagnostics["power_total"] = float(power_total)
    diagnostics["power_ss"] = float(power_ss)
    diagnostics["power_sc"] = float(power_sc)
    diagnostics["power_cc"] = float(power_cc)
    if power_total > 0.0:
        diagnostics["power_frac_ss"] = float(power_ss / power_total)
        diagnostics["power_frac_sc"] = float(power_sc / power_total)
        diagnostics["power_frac_cc"] = float(power_cc / power_total)

    if not solved_any or total_current <= 1e-30:
        diagnostics["failure_reason"] = "no_spanning_component"
        diagnostics["failure_stage"] = "connectivity"
        return None, diagnostics

    g_eff = float(total_current)
    return g_eff / l_rve_nm, diagnostics


def _edge_type_counts(G: sparse.csr_matrix, n_cb: int) -> tuple[int, int, int]:
    """Return distinct CB-CB, CB-CNT, and CNT-CNT edge counts."""

    upper = sparse.triu(G, k=1).tocoo()
    rows = upper.row
    cols = upper.col
    cb_row = rows < n_cb
    cb_col = cols < n_cb
    n_edges_ss = int(np.count_nonzero(cb_row & cb_col))
    n_edges_sc = int(np.count_nonzero(cb_row ^ cb_col))
    n_edges_cc = int(np.count_nonzero(~cb_row & ~cb_col))
    return n_edges_ss, n_edges_sc, n_edges_cc


def _boundary_flags(
    *,
    cb: CBPopulation,
    cnt: CNTPopulation,
    cnt_diam_nm: float,
    l_rve_nm: float,
    direction: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_cb = len(cb.diameters)
    n_cnt = len(cnt.lengths)
    low = np.zeros(n_cb + n_cnt, dtype=bool)
    high = np.zeros(n_cb + n_cnt, dtype=bool)
    if n_cb > 0:
        radii = cb.diameters / 2.0
        pos = cb.centers[:, direction]
        low[:n_cb] = pos < radii
        high[:n_cb] = pos > l_rve_nm - radii

    r_cnt = cnt_diam_nm / 2.0
    for i in range(n_cnt):
        o0 = int(cnt.seg_offsets[i])
        o1 = int(cnt.seg_offsets[i + 1])
        pts = np.vstack((cnt.seg_starts[o0:o1], cnt.seg_ends[o0:o1]))
        low[n_cb + i] = bool(np.any(pts[:, direction] < r_cnt))
        high[n_cb + i] = bool(np.any(pts[:, direction] > l_rve_nm - r_cnt))
    return low, high


def _solve_component_current(
    G_sub: sparse.csr_matrix,
    low_flags: np.ndarray,
    high_flags: np.ndarray,
    *,
    node_ids: np.ndarray,
    n_cb_global: int,
) -> ComponentSolveResult | None:
    overlap = low_flags & high_flags
    if np.any(overlap):
        return None

    diag = np.asarray(G_sub.sum(axis=1)).ravel()
    lap = sparse.diags(diag) - G_sub
    known = np.flatnonzero(low_flags | high_flags)
    unknown = np.flatnonzero(~(low_flags | high_flags))

    V = np.zeros(G_sub.shape[0], dtype=np.float64)
    V[high_flags] = 1.0

    if len(unknown) > 0:
        L_uu = lap[np.ix_(unknown, unknown)]
        L_uk = lap[np.ix_(unknown, known)]
        rhs = -L_uk.dot(V[known])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                V_u = spsolve(L_uu.tocsc(), rhs)
        except Exception:
            return None
        if not np.all(np.isfinite(V_u)):
            return None
        residual = L_uu.dot(V_u) - rhs
        residual_norm = float(np.linalg.norm(residual))
        rhs_norm = float(np.linalg.norm(rhs))
        residual_ratio = residual_norm / max(rhs_norm, 1e-30)
        V[unknown] = V_u
        matrix_nnz = int(L_uu.nnz)
    else:
        residual_ratio = 0.0
        matrix_nnz = 0

    current = 0.0
    for li in np.flatnonzero(high_flags):
        start = G_sub.indptr[li]
        end = G_sub.indptr[li + 1]
        js = G_sub.indices[start:end]
        gs = G_sub.data[start:end]
        current += float(np.sum(gs * (1.0 - V[js])))

    power_total, power_ss, power_sc, power_cc = _component_edge_power_by_type(
        G_sub=G_sub,
        node_ids=node_ids,
        n_cb_global=n_cb_global,
        potentials=V,
    )
    return ComponentSolveResult(
        current=abs(current),
        residual_ratio=residual_ratio,
        n_unknown=int(len(unknown)),
        matrix_nnz=matrix_nnz,
        power_total=power_total,
        power_ss=power_ss,
        power_sc=power_sc,
        power_cc=power_cc,
    )


def _component_edge_power_by_type(
    *,
    G_sub: sparse.csr_matrix,
    node_ids: np.ndarray,
    n_cb_global: int,
    potentials: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return dissipated edge power by junction type for one solved component."""

    upper = sparse.triu(G_sub, k=1).tocoo()
    if upper.nnz == 0:
        return 0.0, 0.0, 0.0, 0.0

    dv = potentials[upper.row] - potentials[upper.col]
    power = np.asarray(upper.data, dtype=np.float64) * dv * dv
    total = float(np.sum(power))

    row_global = node_ids[upper.row]
    col_global = node_ids[upper.col]
    row_cb = row_global < n_cb_global
    col_cb = col_global < n_cb_global
    ss_mask = row_cb & col_cb
    sc_mask = row_cb ^ col_cb
    cc_mask = ~row_cb & ~col_cb
    return (
        total,
        float(np.sum(power[ss_mask])),
        float(np.sum(power[sc_mask])),
        float(np.sum(power[cc_mask])),
    )


def run_step_conductivity_realization(
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
    seed: int,
    n_cb_max: int | None = None,
    n_cnt_max: int | None = None,
    direction: int = 0,
) -> StepConductivityResult:
    """Sample one fixed-composition RVE and solve the step-network conductivity."""

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
        G = build_step_conductance_network(
            cb=cb,
            cnt=cnt,
            cb_model=cb_model,
            cnt_diam_nm=cnt_diam_nm,
            tunnel_nm=tunnel_nm,
            seg_len_unit_nm=seg_len_unit_nm,
        )
        sigma, diagnostics = solve_step_network(
            G=G,
            cb=cb,
            cnt=cnt,
            cnt_diam_nm=cnt_diam_nm,
            l_rve_nm=l_rve_nm,
            direction=direction,
        )
    except PopulationCapError as exc:
        return _finalize_result(
            success=False,
            failure_reason=f"n_max_{exc.filler.lower()}",
            failure_stage="sampling",
            sigma=None,
            g_eff=0.0,
            cb=None,
            cnt=None,
            diagnostics={},
            t0=t0,
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
            t0=t0,
        )

    return _finalize_result(
        success=sigma is not None and sigma > 0.0,
        failure_reason="" if sigma is not None and sigma > 0.0 else str(diagnostics.get("failure_reason", "solve_failed")),
        failure_stage="" if sigma is not None and sigma > 0.0 else str(diagnostics.get("failure_stage", "solve")),
        sigma=sigma,
        g_eff=0.0 if sigma is None else float(sigma * l_rve_nm),
        cb=cb,
        cnt=cnt,
        diagnostics=diagnostics,
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
    t0: float,
) -> StepConductivityResult:
    n_cb = 0 if cb is None else len(cb.diameters)
    n_cnt = 0 if cnt is None else len(cnt.lengths)
    n_total_segs = 0 if cnt is None else len(cnt.seg_starts)
    d = np.empty(0) if cb is None else cb.diameters
    lengths = np.empty(0) if cnt is None else cnt.lengths
    n_seg = np.empty(0, dtype=np.intp) if cnt is None else cnt.n_seg_per_cnt

    return StepConductivityResult(
        success=success,
        failure_reason=failure_reason,
        failure_stage=failure_stage,
        sigma_uncond=float(sigma) if success and sigma is not None else 0.0,
        effective_conductance=g_eff if success else 0.0,
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
        n_components=int(diagnostics.get("n_components", 0)),
        largest_component_size=int(diagnostics.get("largest_component_size", 0)),
        largest_component_fraction=float(diagnostics.get("largest_component_fraction", 0.0)),
        n_spanning_components=int(diagnostics.get("n_spanning_components", 0)),
        low_nodes=int(diagnostics.get("low_nodes", 0)),
        high_nodes=int(diagnostics.get("high_nodes", 0)),
        n_unknown=int(diagnostics.get("n_unknown", 0)),
        matrix_nnz=int(diagnostics.get("matrix_nnz", 0)),
        residual_ratio=float(diagnostics.get("residual_ratio", float("nan"))),
        mean_d_realized_nm=float(np.mean(d)) if n_cb > 0 else float("nan"),
        std_d_realized_nm=float(np.std(d, ddof=1)) if n_cb > 1 else 0.0,
        mean_L_realized_nm=float(np.mean(lengths)) if n_cnt > 0 else float("nan"),
        std_L_realized_nm=float(np.std(lengths, ddof=1)) if n_cnt > 1 else 0.0,
        mean_n_seg_per_cnt=float(np.mean(n_seg)) if n_cnt > 0 else float("nan"),
        max_n_seg_per_cnt=int(np.max(n_seg)) if n_cnt > 0 else 0,
        elapsed_s=time.time() - t0,
    )
