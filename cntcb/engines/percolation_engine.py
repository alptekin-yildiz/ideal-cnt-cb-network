"""Shared percolation engine: sampling and spanning-detection conventions.

This module does not introduce a new physical model; it centralizes the
sampling and boundary-flagged union-find logic shared by the hybrid
percolation and transport engines and their command-line front ends.

Conventions implemented here:

* penetrable particle ensemble;
* truncated-moment-matched log-normal distributions;
* fixed CNT contour step plus terminal remainder;
* boundary-flagged union-find percolation detection;
* volume-stopped hybrid threshold realizations;
* threshold ensembles with one RNG seed per realization (serial or
  process-parallel; the worker count never changes the numbers).
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from cntcb.kernel.percolation_finder import (
    FACE_X_HIGH,
    FACE_X_LOW,
    FACE_Y_HIGH,
    FACE_Y_LOW,
    FACE_Z_HIGH,
    FACE_Z_LOW,
    _SpatialHash,
    _UnionFind,
    _batch_point_seg_dist,
    _batch_seg_dist,
    _sphere_flags,
)


NORMAL = NormalDist()


@dataclass(frozen=True)
class TruncatedLognormal:
    """Parent distribution whose truncated moments match a target pair."""

    target_mu_nm: float
    target_sigma_nm: float
    lower_nm: float
    upper_nm: float
    parent_mu_nm: float
    parent_sigma_nm: float
    mu_ln: float
    sigma_ln: float
    expected_mean_nm: float
    expected_std_nm: float
    expected_e3_nm3: float
    trunc_prob_low: float
    trunc_prob_high: float
    trunc_prob_kept: float


@dataclass(frozen=True)
class HybridPercolationResult:
    """CNT threshold result in a fixed CB background."""

    percolated: bool
    failure_reason: str
    phi_cnt_stop_vol: float
    phi_cb_stop_vol: float
    n_cb: int
    n_cnt: int
    n_total_segs: int
    mean_d_realized_nm: float
    std_d_realized_nm: float
    mean_L_realized_nm: float
    std_L_realized_nm: float
    mean_n_seg_per_cnt: float
    max_n_seg_per_cnt: int
    elapsed_s: float


@dataclass(frozen=True)
class CBThresholdResult:
    """CB threshold result for one volume-stopped single-filler realization."""

    percolated: bool
    failure_reason: str
    phi_cb_stop_vol: float
    n_cb: int
    mean_d_realized_nm: float
    std_d_realized_nm: float
    elapsed_s: float


def wt_frac_to_phi(wt_frac: float, rho_filler: float, rho_matrix: float) -> float:
    """Convert filler mass fraction to volume fraction."""

    return (wt_frac / rho_filler) / (
        wt_frac / rho_filler + (1.0 - wt_frac) / rho_matrix
    )


def phi_to_wt_pct(phi: float, rho_filler: float, rho_matrix: float) -> float:
    """Convert filler volume fraction to wt.%."""

    wt = phi * rho_filler / (phi * rho_filler + (1.0 - phi) * rho_matrix)
    return 100.0 * wt


def _lognormal_params_from_linear(mu: float, sigma: float) -> tuple[float, float]:
    if mu <= 0.0 or sigma < 0.0:
        raise ValueError("Log-normal linear moments must satisfy mu>0 and sigma>=0.")
    if sigma <= 1e-12:
        return float(math.log(mu)), 0.0
    cv2 = (sigma / mu) ** 2
    sigma_ln = float(math.sqrt(math.log(1.0 + cv2)))
    mu_ln = float(math.log(mu) - 0.5 * sigma_ln**2)
    return mu_ln, sigma_ln


def _normal_cdf(value: float) -> float:
    return float(NORMAL.cdf(value))


def truncated_raw_moment(
    mu_ln: float,
    sigma_ln: float,
    order: int,
    lower_nm: float,
    upper_nm: float,
) -> float:
    """Return E[x^order | lower <= x <= upper] for a log-normal variable."""

    if sigma_ln <= 1e-12:
        value = float(math.exp(mu_ln))
        if not (lower_nm <= value <= upper_nm):
            raise ValueError("Monodisperse value is outside truncation support.")
        return value**order

    z_lo = (math.log(lower_nm) - mu_ln) / sigma_ln
    z_hi = (math.log(upper_nm) - mu_ln) / sigma_ln
    prob_kept = _normal_cdf(z_hi) - _normal_cdf(z_lo)
    if prob_kept <= 0.0:
        raise ValueError("Log-normal truncation interval has zero probability.")

    shifted_lo = (math.log(lower_nm) - mu_ln - order * sigma_ln**2) / sigma_ln
    shifted_hi = (math.log(upper_nm) - mu_ln - order * sigma_ln**2) / sigma_ln
    raw_parent = math.exp(order * mu_ln + 0.5 * order**2 * sigma_ln**2)
    return raw_parent * (_normal_cdf(shifted_hi) - _normal_cdf(shifted_lo)) / prob_kept


def _truncated_stats_from_parent(
    parent_mu_nm: float,
    parent_sigma_nm: float,
    lower_nm: float,
    upper_nm: float,
) -> tuple[float, float, float, float, float, float]:
    """Return truncated mean, std, E[x^3], P(low), P(high), P(kept)."""

    mu_ln, sigma_ln = _lognormal_params_from_linear(parent_mu_nm, parent_sigma_nm)
    if sigma_ln <= 1e-12:
        mean = float(parent_mu_nm)
        if not (lower_nm <= mean <= upper_nm):
            raise ValueError("Monodisperse value is outside truncation support.")
        return mean, 0.0, mean**3, 0.0, 0.0, 1.0

    z_lo = (math.log(lower_nm) - mu_ln) / sigma_ln
    z_hi = (math.log(upper_nm) - mu_ln) / sigma_ln
    prob_low = _normal_cdf(z_lo)
    prob_high = 1.0 - _normal_cdf(z_hi)
    prob_kept = 1.0 - prob_low - prob_high

    e1 = truncated_raw_moment(mu_ln, sigma_ln, 1, lower_nm, upper_nm)
    e2 = truncated_raw_moment(mu_ln, sigma_ln, 2, lower_nm, upper_nm)
    e3 = truncated_raw_moment(mu_ln, sigma_ln, 3, lower_nm, upper_nm)
    std = math.sqrt(max(0.0, e2 - e1**2))
    return e1, std, e3, prob_low, prob_high, prob_kept


def solve_truncated_lognormal(
    target_mu_nm: float,
    target_sigma_nm: float,
    lower_nm: float,
    upper_nm: float,
) -> TruncatedLognormal:
    """Solve parent moments so the truncated distribution matches the target."""

    if lower_nm <= 0.0 or upper_nm <= lower_nm:
        raise ValueError("Truncation support must satisfy 0 < lower < upper.")
    if not (lower_nm <= target_mu_nm <= upper_nm):
        raise ValueError("Target mean must lie inside the truncation support.")
    if target_sigma_nm < 0.0:
        raise ValueError("Target standard deviation must be non-negative.")

    if target_sigma_nm <= 1e-12:
        mu_ln, sigma_ln = _lognormal_params_from_linear(target_mu_nm, 0.0)
        return TruncatedLognormal(
            target_mu_nm=target_mu_nm,
            target_sigma_nm=0.0,
            lower_nm=lower_nm,
            upper_nm=upper_nm,
            parent_mu_nm=target_mu_nm,
            parent_sigma_nm=0.0,
            mu_ln=mu_ln,
            sigma_ln=sigma_ln,
            expected_mean_nm=target_mu_nm,
            expected_std_nm=0.0,
            expected_e3_nm3=target_mu_nm**3,
            trunc_prob_low=0.0,
            trunc_prob_high=0.0,
            trunc_prob_kept=1.0,
        )

    y = math.log(target_mu_nm)
    z = math.log(target_sigma_nm * 1.05)
    for _ in range(100):
        parent_mu = math.exp(y)
        parent_sigma = math.exp(z)
        mean, std, *_ = _truncated_stats_from_parent(
            parent_mu, parent_sigma, lower_nm, upper_nm
        )
        f0 = mean - target_mu_nm
        f1 = std - target_sigma_nm
        if abs(f0) < 1e-8 and abs(f1) < 1e-8:
            break

        h = 1e-5
        mean_y, std_y, *_ = _truncated_stats_from_parent(
            math.exp(y + h), parent_sigma, lower_nm, upper_nm
        )
        mean_z, std_z, *_ = _truncated_stats_from_parent(
            parent_mu, math.exp(z + h), lower_nm, upper_nm
        )
        j00 = (mean_y - mean) / h
        j10 = (std_y - std) / h
        j01 = (mean_z - mean) / h
        j11 = (std_z - std) / h
        det = j00 * j11 - j01 * j10
        if abs(det) < 1e-14:
            raise RuntimeError(
                f"Could not solve parent moments for {target_mu_nm}/{target_sigma_nm} nm."
            )

        dy = (-f0 * j11 + j01 * f1) / det
        dz = (j10 * f0 - j00 * f1) / det
        current_error = f0**2 + f1**2
        damping = 1.0
        for _try in range(24):
            next_y = y + damping * dy
            next_z = z + damping * dz
            next_mean, next_std, *_ = _truncated_stats_from_parent(
                math.exp(next_y), math.exp(next_z), lower_nm, upper_nm
            )
            next_error = (next_mean - target_mu_nm) ** 2 + (
                next_std - target_sigma_nm
            ) ** 2
            if math.isfinite(next_error) and next_error <= current_error:
                y = next_y
                z = next_z
                break
            damping *= 0.5
        else:
            raise RuntimeError(
                f"Moment solver stalled for {target_mu_nm}/{target_sigma_nm} nm."
            )

    parent_mu = math.exp(y)
    parent_sigma = math.exp(z)
    mu_ln, sigma_ln = _lognormal_params_from_linear(parent_mu, parent_sigma)
    mean, std, e3, prob_low, prob_high, prob_kept = _truncated_stats_from_parent(
        parent_mu, parent_sigma, lower_nm, upper_nm
    )
    if abs(mean - target_mu_nm) > 1e-5 or abs(std - target_sigma_nm) > 1e-5:
        raise RuntimeError(
            f"Moment solver did not converge for {target_mu_nm}/{target_sigma_nm} nm."
        )

    return TruncatedLognormal(
        target_mu_nm=target_mu_nm,
        target_sigma_nm=target_sigma_nm,
        lower_nm=lower_nm,
        upper_nm=upper_nm,
        parent_mu_nm=parent_mu,
        parent_sigma_nm=parent_sigma,
        mu_ln=mu_ln,
        sigma_ln=sigma_ln,
        expected_mean_nm=mean,
        expected_std_nm=std,
        expected_e3_nm3=e3,
        trunc_prob_low=prob_low,
        trunc_prob_high=prob_high,
        trunc_prob_kept=prob_kept,
    )


def sample_truncated_lognormal(
    rng: np.random.Generator,
    model: TruncatedLognormal,
) -> float:
    """Draw one value from a moment-matched truncated log-normal model."""

    if model.sigma_ln <= 1e-12:
        return float(math.exp(model.mu_ln))

    z_lo = (math.log(model.lower_nm) - model.mu_ln) / model.sigma_ln
    z_hi = (math.log(model.upper_nm) - model.mu_ln) / model.sigma_ln
    cdf_lo = _normal_cdf(z_lo)
    cdf_hi = _normal_cdf(z_hi)
    u = cdf_lo + float(rng.uniform(0.0, 1.0)) * (cdf_hi - cdf_lo)
    u = max(1e-14, min(u, 1.0 - 1e-14))
    value = math.exp(model.mu_ln + model.sigma_ln * NORMAL.inv_cdf(u))
    return max(model.lower_nm, min(float(value), model.upper_nm))


def fixed_step_segment_lengths(length_nm: float, unit_nm: float) -> np.ndarray:
    """Return fixed contour steps plus a terminal remainder segment."""

    if length_nm <= 0.0:
        raise ValueError(f"CNT length must be positive, got {length_nm}.")
    if unit_nm <= 0.0:
        raise ValueError(f"Segment unit must be positive, got {unit_nm}.")

    n_full = int(math.floor(length_nm / unit_nm))
    remainder = float(length_nm - n_full * unit_nm)
    pieces = [unit_nm] * n_full
    if remainder > 1e-9 or not pieces:
        pieces.append(remainder if remainder > 1e-9 else length_nm)
    return np.asarray(pieces, dtype=np.float64)


def cnt_boundary_flags(
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    radius_nm: float,
    l_rve_nm: float,
) -> int:
    """Six-face boundary flags from all CNT segment endpoints."""

    pts = np.vstack((seg_starts, seg_ends))
    flags = 0
    if np.any(pts[:, 0] < radius_nm):
        flags |= FACE_X_LOW
    if np.any(pts[:, 0] > l_rve_nm - radius_nm):
        flags |= FACE_X_HIGH
    if np.any(pts[:, 1] < radius_nm):
        flags |= FACE_Y_LOW
    if np.any(pts[:, 1] > l_rve_nm - radius_nm):
        flags |= FACE_Y_HIGH
    if np.any(pts[:, 2] < radius_nm):
        flags |= FACE_Z_LOW
    if np.any(pts[:, 2] > l_rve_nm - radius_nm):
        flags |= FACE_Z_HIGH
    return flags


def estimate_cb_n_max(
    cb_model: TruncatedLognormal,
    phi_target_vol: float,
    l_rve_nm: float,
    safety: float = 2.5,
    extra: int = 200,
) -> int:
    """Conservative CB particle-count cap for a volume-stopped run."""

    v_mean = (math.pi / 6.0) * cb_model.expected_e3_nm3
    n_exp = phi_target_vol * l_rve_nm**3 / max(v_mean, 1e-30)
    return max(extra, int(math.ceil(n_exp * safety)) + extra)


def estimate_cnt_n_max(
    cnt_model: TruncatedLognormal,
    phi_target_vol: float,
    l_rve_nm: float,
    cnt_diam_nm: float,
    safety: float = 2.5,
    extra: int = 200,
) -> int:
    """Conservative CNT count cap for a volume-stopped run."""

    r_cnt = cnt_diam_nm / 2.0
    v_mean = math.pi * r_cnt**2 * cnt_model.expected_mean_nm
    n_exp = phi_target_vol * l_rve_nm**3 / max(v_mean, 1e-30)
    return max(extra, int(math.ceil(n_exp * safety)) + extra)


def _draw_cnt_segments(
    *,
    rng: np.random.Generator,
    length_nm: float,
    l_rve_nm: float,
    radius_nm: float,
    waviness: float,
    seg_len_unit_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw one CNT and return segment starts, ends, and segment lengths."""

    seg_lens = fixed_step_segment_lengths(length_nm, seg_len_unit_nm)
    n_seg = int(len(seg_lens))
    starts = np.empty((n_seg, 3), dtype=np.float64)
    ends = np.empty((n_seg, 3), dtype=np.float64)

    pos = rng.uniform(0.0, l_rve_nm, size=3)
    d_vec = rng.standard_normal(3)
    d_vec /= np.linalg.norm(d_vec)
    cos_b = float(waviness)
    sin_b = float(math.sqrt(max(0.0, 1.0 - cos_b**2)))

    for idx, seg_len in enumerate(seg_lens):
        starts[idx] = pos
        ends[idx] = pos + d_vec * float(seg_len)
        pos = ends[idx]
        if idx < n_seg - 1:
            perturb = rng.standard_normal(3)
            perturb -= np.dot(perturb, d_vec) * d_vec
            norm_p = float(np.linalg.norm(perturb))
            if norm_p > 1e-12:
                perturb /= norm_p
            d_vec = cos_b * d_vec + sin_b * perturb
            d_vec /= np.linalg.norm(d_vec)

    return starts, ends, seg_lens


def _connect_new_cnt_to_previous_cnts(
    *,
    uf: _UnionFind,
    new_node: int,
    segs_s: np.ndarray,
    segs_e: np.ndarray,
    mids: np.ndarray,
    seg_hash: _SpatialHash,
    seg_to_node: np.ndarray,
    seg_offsets: np.ndarray,
    act_starts: np.ndarray,
    act_ends: np.ndarray,
    connect_d: float,
) -> bool:
    cand_set: set[int] = set()
    for mid in mids:
        for flat_id in seg_hash.query(mid):
            cand = int(seg_to_node[flat_id])
            if cand != new_node:
                cand_set.add(cand)
    if not cand_set:
        return False

    cand_arr = np.fromiter(sorted(cand_set), dtype=np.intp)
    per_cand_lengths = (seg_offsets[cand_arr + 1] - seg_offsets[cand_arr]).astype(np.intp)
    starts = np.empty(len(cand_arr), dtype=np.intp)
    starts[0] = 0
    if len(cand_arr) > 1:
        np.cumsum(per_cand_lengths[:-1], out=starts[1:])

    total_cand_segs = int(per_cand_lengths.sum())
    idx = np.empty(total_cand_segs, dtype=np.intp)
    write = 0
    for cand, n_s in zip(cand_arr.tolist(), per_cand_lengths.tolist()):
        o0 = int(seg_offsets[cand])
        o1 = o0 + int(n_s)
        idx[write:write + int(n_s)] = np.arange(o0, o1, dtype=np.intp)
        write += int(n_s)

    cand_starts = act_starts[idx]
    cand_ends = act_ends[idx]
    connected = np.zeros(len(cand_arr), dtype=bool)
    for s in range(len(segs_s)):
        dists = _batch_seg_dist(segs_s[s], segs_e[s], cand_starts, cand_ends)
        per_cand_min = np.minimum.reduceat(dists, starts)
        connected |= per_cand_min < connect_d

    percolated = False
    for j_idx in np.flatnonzero(connected):
        if uf.union(new_node, int(cand_arr[j_idx])):
            percolated = True
    return percolated


def run_hybrid_cnt_threshold_realization(
    *,
    cb_model: TruncatedLognormal,
    cnt_model: TruncatedLognormal,
    phi_cb_vol: float,
    phi_cnt_max_vol: float,
    l_rve_nm: float,
    cnt_diam_nm: float,
    waviness: float,
    tunnel_nm: float,
    seg_len_unit_nm: float,
    seed: int,
    n_cb_max: int | None = None,
    n_cnt_max: int | None = None,
) -> HybridPercolationResult:
    """One hybrid realization: place fixed CB background, then add CNTs."""

    t0 = time.time()
    rng = np.random.default_rng(seed)
    r_cnt = cnt_diam_nm / 2.0
    v_rve = l_rve_nm**3

    cb_cap = n_cb_max or estimate_cb_n_max(cb_model, phi_cb_vol, l_rve_nm)
    cnt_cap = n_cnt_max or estimate_cnt_n_max(
        cnt_model, phi_cnt_max_vol, l_rve_nm, cnt_diam_nm
    )
    node_cap = cb_cap + cnt_cap
    max_seg_per_cnt = int(math.ceil(cnt_model.upper_nm / seg_len_unit_nm))
    max_total_segs = cnt_cap * max_seg_per_cnt

    uf = _UnionFind(node_cap)
    cb_centers = np.empty((cb_cap, 3), dtype=np.float64)
    cb_diameters = np.empty(cb_cap, dtype=np.float64)
    cb_hash = _SpatialHash(
        cell_size=max(
            cb_model.upper_nm + tunnel_nm,
            seg_len_unit_nm + cb_model.upper_nm / 2.0 + r_cnt + tunnel_nm,
        )
    )
    seg_hash = _SpatialHash(cell_size=seg_len_unit_nm + 2.0 * r_cnt + tunnel_nm)

    act_starts = np.empty((max_total_segs, 3), dtype=np.float64)
    act_ends = np.empty((max_total_segs, 3), dtype=np.float64)
    seg_to_node = np.empty(max_total_segs, dtype=np.intp)
    seg_offsets = np.zeros(node_cap + 1, dtype=np.intp)
    lengths = np.empty(cnt_cap, dtype=np.float64)
    n_seg_per_cnt = np.empty(cnt_cap, dtype=np.intp)

    cb_vol_total = 0.0
    cb_target_vol = phi_cb_vol * v_rve
    n_cb = 0

    for cb_idx in range(cb_cap):
        if cb_vol_total >= cb_target_vol:
            break
        d_i = sample_truncated_lognormal(rng, cb_model)
        center = rng.uniform(0.0, l_rve_nm, size=3)
        r_i = d_i / 2.0

        uf.set_boundary(cb_idx, _sphere_flags(center, r_i, l_rve_nm))
        cb_percolated = uf.is_root_percolating(cb_idx)
        for j in cb_hash.query(center):
            d_j = cb_diameters[j]
            connect_d = (d_i + d_j) / 2.0 + tunnel_nm
            if float(np.linalg.norm(center - cb_centers[j])) < connect_d:
                if uf.union(cb_idx, int(j)):
                    cb_percolated = True

        cb_centers[cb_idx] = center
        cb_diameters[cb_idx] = d_i
        cb_hash.insert(cb_idx, center)
        n_cb += 1
        cb_vol_total += (math.pi / 6.0) * d_i**3

        if cb_percolated:
            return _finalize_hybrid_result(
                True,
                "cb_percolated",
                0.0,
                cb_vol_total,
                n_cb,
                0,
                0,
                cb_diameters,
                lengths,
                n_seg_per_cnt,
                act_starts,
                act_ends,
                l_rve_nm,
                t0,
            )

    cnt_vol_total = 0.0
    cnt_target_vol = phi_cnt_max_vol * v_rve
    n_total_segs = 0
    v_cnt_prefactor = math.pi * r_cnt**2
    cnt_connect_d = tunnel_nm + 2.0 * r_cnt

    for cnt_local in range(cnt_cap):
        node = n_cb + cnt_local
        length_i = sample_truncated_lognormal(rng, cnt_model)
        segs_s, segs_e, seg_lens = _draw_cnt_segments(
            rng=rng,
            length_nm=length_i,
            l_rve_nm=l_rve_nm,
            radius_nm=r_cnt,
            waviness=waviness,
            seg_len_unit_nm=seg_len_unit_nm,
        )
        n_seg_i = int(len(seg_lens))
        if n_total_segs + n_seg_i > max_total_segs:
            return _finalize_hybrid_result(
                False,
                "segment_capacity",
                cnt_vol_total,
                cb_vol_total,
                n_cb,
                cnt_local,
                n_total_segs,
                cb_diameters,
                lengths,
                n_seg_per_cnt,
                act_starts,
                act_ends,
                l_rve_nm,
                t0,
            )

        seg_offsets[node] = n_total_segs
        seg_offsets[node + 1] = n_total_segs + n_seg_i
        lengths[cnt_local] = length_i
        n_seg_per_cnt[cnt_local] = n_seg_i
        mids = 0.5 * (segs_s + segs_e)

        uf.set_boundary(node, cnt_boundary_flags(segs_s, segs_e, r_cnt, l_rve_nm))
        percolated = uf.is_root_percolating(node)

        percolated = _connect_new_cnt_to_cb(
            uf=uf,
            node=node,
            segs_s=segs_s,
            segs_e=segs_e,
            mids=mids,
            cb_hash=cb_hash,
            cb_centers=cb_centers,
            cb_diameters=cb_diameters,
            n_cb=n_cb,
            r_cnt=r_cnt,
            tunnel_nm=tunnel_nm,
        ) or percolated

        if cnt_local > 0:
            percolated = _connect_new_cnt_to_previous_cnts(
                uf=uf,
                new_node=node,
                segs_s=segs_s,
                segs_e=segs_e,
                mids=mids,
                seg_hash=seg_hash,
                seg_to_node=seg_to_node,
                seg_offsets=seg_offsets,
                act_starts=act_starts,
                act_ends=act_ends,
                connect_d=cnt_connect_d,
            ) or percolated

        base = n_total_segs
        act_starts[base:base + n_seg_i] = segs_s
        act_ends[base:base + n_seg_i] = segs_e
        seg_to_node[base:base + n_seg_i] = node
        for s in range(n_seg_i):
            seg_hash.insert(base + s, mids[s])

        n_total_segs += n_seg_i
        cnt_vol_total += v_cnt_prefactor * length_i

        if percolated:
            return _finalize_hybrid_result(
                True,
                "",
                cnt_vol_total,
                cb_vol_total,
                n_cb,
                cnt_local + 1,
                n_total_segs,
                cb_diameters,
                lengths,
                n_seg_per_cnt,
                act_starts,
                act_ends,
                l_rve_nm,
                t0,
            )
        if cnt_vol_total >= cnt_target_vol:
            return _finalize_hybrid_result(
                False,
                "phi_cnt_max",
                cnt_vol_total,
                cb_vol_total,
                n_cb,
                cnt_local + 1,
                n_total_segs,
                cb_diameters,
                lengths,
                n_seg_per_cnt,
                act_starts,
                act_ends,
                l_rve_nm,
                t0,
            )

    return _finalize_hybrid_result(
        False,
        "n_cnt_max",
        cnt_vol_total,
        cb_vol_total,
        n_cb,
        cnt_cap,
        n_total_segs,
        cb_diameters,
        lengths,
        n_seg_per_cnt,
        act_starts,
        act_ends,
        l_rve_nm,
        t0,
    )


def _connect_new_cnt_to_cb(
    *,
    uf: _UnionFind,
    node: int,
    segs_s: np.ndarray,
    segs_e: np.ndarray,
    mids: np.ndarray,
    cb_hash: _SpatialHash,
    cb_centers: np.ndarray,
    cb_diameters: np.ndarray,
    n_cb: int,
    r_cnt: float,
    tunnel_nm: float,
) -> bool:
    if n_cb <= 0:
        return False

    percolated = False
    for s in range(len(segs_s)):
        cand = [int(j) for j in cb_hash.query(mids[s]) if int(j) < n_cb]
        if not cand:
            continue
        cand_arr = np.asarray(sorted(set(cand)), dtype=np.intp)
        dists = _batch_point_seg_dist(cb_centers[cand_arr], segs_s[s], segs_e[s])
        thresholds = cb_diameters[cand_arr] / 2.0 + r_cnt + tunnel_nm
        for cb_idx in cand_arr[np.flatnonzero(dists < thresholds)]:
            if uf.union(node, int(cb_idx)):
                percolated = True
    return percolated


def _finalize_hybrid_result(
    percolated: bool,
    failure_reason: str,
    cnt_vol_total: float,
    cb_vol_total: float,
    n_cb: int,
    n_cnt: int,
    n_total_segs: int,
    cb_diameters: np.ndarray,
    lengths: np.ndarray,
    n_seg_per_cnt: np.ndarray,
    act_starts: np.ndarray,
    act_ends: np.ndarray,
    l_rve_nm: float,
    t0: float,
) -> HybridPercolationResult:
    cb_d = cb_diameters[:n_cb]
    cnt_lengths = lengths[:n_cnt]
    if n_total_segs > 0:
        seg_lens = np.linalg.norm(
            act_ends[:n_total_segs] - act_starts[:n_total_segs], axis=1
        )
        mean_seg_len = float(np.mean(seg_lens))
    else:
        mean_seg_len = float("nan")

    return HybridPercolationResult(
        percolated=percolated,
        failure_reason=failure_reason,
        phi_cnt_stop_vol=float(cnt_vol_total / l_rve_nm**3),
        phi_cb_stop_vol=float(cb_vol_total / l_rve_nm**3),
        n_cb=int(n_cb),
        n_cnt=int(n_cnt),
        n_total_segs=int(n_total_segs),
        mean_d_realized_nm=float(np.mean(cb_d)) if n_cb > 0 else float("nan"),
        std_d_realized_nm=float(np.std(cb_d, ddof=1)) if n_cb > 1 else 0.0,
        mean_L_realized_nm=float(np.mean(cnt_lengths)) if n_cnt > 0 else float("nan"),
        std_L_realized_nm=float(np.std(cnt_lengths, ddof=1)) if n_cnt > 1 else 0.0,
        mean_n_seg_per_cnt=float(np.mean(n_seg_per_cnt[:n_cnt])) if n_cnt > 0 else float("nan"),
        max_n_seg_per_cnt=int(np.max(n_seg_per_cnt[:n_cnt])) if n_cnt > 0 else 0,
        elapsed_s=time.time() - t0,
    )


def run_cb_threshold_realization(
    *,
    cb_model: TruncatedLognormal,
    phi_cb_max_vol: float,
    l_rve_nm: float,
    tunnel_nm: float,
    seed: int,
    n_cb_max: int | None = None,
) -> CBThresholdResult:
    """One volume-stopped CB realization: add spheres until the box percolates.

    Penetrable spheres with diameters drawn from ``cb_model`` are inserted one
    at a time; the run stops at the first spanning cluster (the stop volume
    fraction is the threshold sample) or, censored, when the filled volume
    reaches ``phi_cb_max_vol`` or the particle cap.
    """

    t0 = time.time()
    rng = np.random.default_rng(seed)
    cap = n_cb_max or estimate_cb_n_max(cb_model, phi_cb_max_vol, l_rve_nm)

    centers = np.empty((cap, 3), dtype=np.float64)
    diameters = np.empty(cap, dtype=np.float64)
    uf = _UnionFind(cap)
    cb_hash = _SpatialHash(cell_size=cb_model.upper_nm + tunnel_nm)

    v_target = phi_cb_max_vol * l_rve_nm**3
    vol_total = 0.0
    n_cb = 0
    percolated = False
    failure_reason = "n_max"

    for step in range(cap):
        d_i = sample_truncated_lognormal(rng, cb_model)
        center = rng.uniform(0.0, l_rve_nm, size=3)

        uf.set_boundary(step, _sphere_flags(center, d_i / 2.0, l_rve_nm))
        step_percolated = uf.is_root_percolating(step)
        for j in cb_hash.query(center):
            connect_d = (d_i + diameters[j]) / 2.0 + tunnel_nm
            dist = float(np.linalg.norm(center - centers[j]))
            if dist < connect_d and uf.union(step, int(j)):
                step_percolated = True

        centers[step] = center
        diameters[step] = d_i
        cb_hash.insert(step, center)
        n_cb += 1
        vol_total += (math.pi / 6.0) * d_i**3

        if step_percolated:
            percolated = True
            failure_reason = ""
            break
        if vol_total >= v_target:
            failure_reason = "phi_max"
            break

    d = diameters[:n_cb]
    return CBThresholdResult(
        percolated=percolated,
        failure_reason=failure_reason,
        phi_cb_stop_vol=float(vol_total / l_rve_nm**3),
        n_cb=int(n_cb),
        mean_d_realized_nm=float(np.mean(d)) if n_cb > 0 else float("nan"),
        std_d_realized_nm=float(np.std(d, ddof=1)) if n_cb > 1 else 0.0,
        elapsed_s=time.time() - t0,
    )


def _cb_threshold_task(kwargs: dict) -> CBThresholdResult:
    """Module-level worker so process pools can pickle ensemble tasks."""

    return run_cb_threshold_realization(**kwargs)


def _hybrid_threshold_task(kwargs: dict) -> HybridPercolationResult:
    """Module-level worker so process pools can pickle ensemble tasks."""

    return run_hybrid_cnt_threshold_realization(**kwargs)


def _run_ensemble_tasks(task_fn, tasks: list[dict], workers: int, on_result=None) -> list:
    """Run realization tasks serially or in a process pool, in task order.

    Every task carries its own RNG seed, so the returned list is bit-identical
    for any ``workers`` value; only the wall-clock time changes.
    """

    results: list = [None] * len(tasks)
    if workers <= 1:
        for idx, task in enumerate(tasks):
            result = task_fn(task)
            results[idx] = result
            if on_result is not None:
                on_result(idx, result, idx + 1, len(tasks))
        return results

    n_done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(task_fn, task): idx for idx, task in enumerate(tasks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            results[idx] = result
            n_done += 1
            if on_result is not None:
                on_result(idx, result, n_done, len(tasks))
    return results


def run_cb_threshold_ensemble(
    *,
    cb_model: TruncatedLognormal,
    phi_cb_max_vol: float,
    l_rve_nm: float,
    tunnel_nm: float,
    seed_base: int,
    n_realizations: int,
    n_cb_max: int | None = None,
    workers: int = 1,
    on_result=None,
) -> list[CBThresholdResult]:
    """CB threshold ensemble with one independent seed per realization."""

    tasks = [
        dict(
            cb_model=cb_model,
            phi_cb_max_vol=phi_cb_max_vol,
            l_rve_nm=l_rve_nm,
            tunnel_nm=tunnel_nm,
            seed=seed_base + r,
            n_cb_max=n_cb_max,
        )
        for r in range(n_realizations)
    ]
    return _run_ensemble_tasks(_cb_threshold_task, tasks, workers, on_result)


def run_cnt_threshold_ensemble(
    *,
    cnt_model: TruncatedLognormal,
    cb_model: TruncatedLognormal,
    phi_cb_vol: float,
    phi_cnt_max_vol: float,
    l_rve_nm: float,
    cnt_diam_nm: float,
    waviness: float,
    tunnel_nm: float,
    seg_len_unit_nm: float,
    seed_base: int,
    n_realizations: int,
    n_cb_max: int | None = None,
    n_cnt_max: int | None = None,
    workers: int = 1,
    on_result=None,
) -> list[HybridPercolationResult]:
    """CNT threshold ensemble: pure CNT for ``phi_cb_vol=0``, else hybrid.

    Each realization places the fixed CB background (skipped entirely when
    ``phi_cb_vol`` is zero) and then adds CNTs until the network percolates,
    with one independent seed per realization.
    """

    tasks = [
        dict(
            cb_model=cb_model,
            cnt_model=cnt_model,
            phi_cb_vol=phi_cb_vol,
            phi_cnt_max_vol=phi_cnt_max_vol,
            l_rve_nm=l_rve_nm,
            cnt_diam_nm=cnt_diam_nm,
            waviness=waviness,
            tunnel_nm=tunnel_nm,
            seg_len_unit_nm=seg_len_unit_nm,
            seed=seed_base + r,
            n_cb_max=n_cb_max,
            n_cnt_max=n_cnt_max,
        )
        for r in range(n_realizations)
    ]
    return _run_ensemble_tasks(_hybrid_threshold_task, tasks, workers, on_result)
