#!/usr/bin/env python3
"""Polydisperse CNT percolation-threshold sweep for the paper.

This paper reproduction wrapper regenerates the manuscript's Fig. 4 CNT
length sweep with the publication CNT segmentation convention:

    fixed 50 nm contour steps + a terminal remainder segment.

A fixed per-CNT segment count (e.g. ``N_SEG=10``) would give 150 nm segments
for a 1500 nm CNT but only 10 nm segments for a 100 nm CNT; the fixed
50 nm contour step instead keeps the local waviness scale the same for
every length in the polydisperse population.

Outputs:
  data/processed/percolation/cnt_phic_sweep_fixedstep_Lmin50.csv
  data/processed/percolation/cnt_phic_sweep_fixedstep_Lmin50_realizations.csv
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

import numpy as np


# Put the repo root on sys.path so a bare checkout runs without pip install;
# the cntcb imports below intentionally follow this bootstrap (hence E402).
ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT
OUT_DIR = PAPER / "data" / "processed" / "percolation"

sys.path.insert(0, str(ROOT))

from cntcb.kernel.percolation_finder import (  # noqa: E402
    FACE_X_HIGH,
    FACE_X_LOW,
    FACE_Y_HIGH,
    FACE_Y_LOW,
    FACE_Z_HIGH,
    FACE_Z_LOW,
    _SpatialHash,
    _UnionFind,
    _batch_seg_dist,
)


# Densities and experimental threshold anchor.
RHO_CNT = 1.75
RHO_PEI = 1.27
PHI_C_EXP_WT = 0.904

# Simulation conventions.
PHI_MAX_WT = 2.0
CNT_DIAM_NM = 10.0
WAVINESS = 0.7
SEG_LEN_UNIT_NM = 50.0
TUNNEL_NM = 10.0
L_RVE = 4000.0
# Publication convention for the CNT length distribution support.
# The lower bound is kept at the fixed local contour step scale so that the
# sampled population does not contain CNT-like objects shorter than one segment.
L_MIN_NM = 50.0
L_MAX_NM = 1500.0
SEED_BASE = 20260527

# Fig. 6 sweep grid.
MU_L_VALS = (700, 600, 500, 400, 300)
SIGMA_L_VALS = (0, 100, 150, 200, 300, 400)
SIGMA_CONSTRAINT_NM = 50

N_REAL_PRODUCTION = 100
N_REAL_QUICK = 3
N_WORKERS_DEFAULT = 10
PRODUCTION_TAG = "Lmin50"

# Monodisperse 500 nm reference used only as a compact analytical guide.
PHI_C_REF_VOL = 0.00803
L_REF_NM = 500.0

SUMMARY_CSV = OUT_DIR / f"cnt_phic_sweep_fixedstep_{PRODUCTION_TAG}.csv"
REAL_CSV = OUT_DIR / f"cnt_phic_sweep_fixedstep_{PRODUCTION_TAG}_realizations.csv"

SEGMENTATION_RULE = "fixed_50nm_plus_terminal_remainder"

SUMMARY_HEADER = [
    "mu_L_nm",
    "sigma_L_nm",
    "cv",
    "phi_c_vol",
    "phi_c_wt_pct",
    "phi_c_q25_wt",
    "phi_c_q75_wt",
    "P_at_exp_wt",
    "P_at_1wt",
    "n_perc",
    "n_total",
    "n_censored",
    "phi_c_analytic_vol",
    "phi_c_analytic_wt_pct",
    "phi_act_mean",
    "phi_act_std",
    "n_cnt_mean",
    "n_cnt_std",
    "n_total_segs_mean",
    "mean_n_seg_per_cnt",
    "max_n_seg_per_cnt",
    "mean_segment_len_nm",
    "max_segment_len_nm",
    "cnt_segmentation_rule",
    "seg_len_unit_nm",
    "terminal_remainder",
    "L_RVE_nm",
    "L_min_nm",
    "L_max_nm",
    "CNT_diam_nm",
    "waviness",
    "tunnel_nm",
    "phi_max_wt_pct",
    "phi_exp_wt_pct",
    "elapsed_s",
]

REAL_HEADER = [
    "mu_L_nm",
    "sigma_L_nm",
    "cv",
    "realization",
    "seed",
    "percolated",
    "failure_reason",
    "phi_perc_vol_censored",
    "phi_perc_wt_pct_censored",
    "phi_stop_vol",
    "phi_stop_wt_pct",
    "n_cnt",
    "n_total_segs",
    "mean_n_seg_per_cnt",
    "max_n_seg_per_cnt",
    "mean_segment_len_nm",
    "max_segment_len_nm",
    "mean_L_realized_nm",
    "std_L_realized_nm",
    "cnt_segmentation_rule",
    "seg_len_unit_nm",
    "terminal_remainder",
    "L_RVE_nm",
    "L_min_nm",
    "L_max_nm",
    "CNT_diam_nm",
    "waviness",
    "tunnel_nm",
    "phi_max_wt_pct",
    "elapsed_s",
]


@dataclass(frozen=True)
class RealizationResult:
    percolated: bool
    failure_reason: str
    phi_stop_vol: float
    n_cnt: int
    n_total_segs: int
    mean_n_seg_per_cnt: float
    max_n_seg_per_cnt: int
    mean_segment_len_nm: float
    max_segment_len_nm: float
    mean_L_realized_nm: float
    std_L_realized_nm: float
    elapsed_s: float


def wt_frac_to_phi(wt_frac: float) -> float:
    """Convert filler mass fraction to volume fraction."""
    return (wt_frac / RHO_CNT) / (wt_frac / RHO_CNT + (1.0 - wt_frac) / RHO_PEI)


def phi_to_wt_pct(phi: float) -> float:
    """Convert filler volume fraction to wt.%."""
    wt = phi * RHO_CNT / (phi * RHO_CNT + (1.0 - phi) * RHO_PEI)
    return 100.0 * wt


def phi_c_analytic(mu_L: float, cv: float) -> float:
    """Mean-field guide for a polydisperse rod population, in volume fraction."""
    return PHI_C_REF_VOL * L_REF_NM / (mu_L * (1.0 + cv**2))


def _solve_lognormal_truncated(
    mu_target: float,
    sigma_target: float,
    l_min: float,
    l_max: float,
) -> tuple[float, float]:
    """Solve log-normal parameters whose truncated moments match the targets."""
    from scipy.optimize import fsolve
    from scipy.special import ndtr

    if sigma_target <= 0.0:
        return float(np.log(mu_target)), 0.0

    def _trunc_moments(mu_ln: float, sigma_ln: float) -> tuple[float, float]:
        a = (np.log(l_min) - mu_ln) / sigma_ln
        b = (np.log(l_max) - mu_ln) / sigma_ln
        p = ndtr(b) - ndtr(a)
        if p < 1e-14:
            return np.inf, np.inf

        m1 = (
            np.exp(mu_ln + 0.5 * sigma_ln**2)
            * (ndtr(b - sigma_ln) - ndtr(a - sigma_ln))
            / p
        )
        m2 = (
            np.exp(2.0 * mu_ln + 2.0 * sigma_ln**2)
            * (ndtr(b - 2.0 * sigma_ln) - ndtr(a - 2.0 * sigma_ln))
            / p
        )
        return float(m1), float(np.sqrt(max(0.0, m2 - m1**2)))

    def _residuals(params: np.ndarray) -> list[float]:
        mu_ln, log_s = params
        sigma_ln = float(np.exp(log_s))
        m, s = _trunc_moments(float(mu_ln), sigma_ln)
        return [
            (m - mu_target) / mu_target,
            (s - sigma_target) / sigma_target,
        ]

    cv0 = sigma_target / mu_target
    s0 = float(np.sqrt(np.log(1.0 + cv0**2)))
    m0 = float(np.log(mu_target) - 0.5 * s0**2)

    sol = fsolve(_residuals, [m0, np.log(s0)], full_output=True)
    mu_ln = float(sol[0][0])
    sigma_ln = float(np.exp(sol[0][1]))
    return mu_ln, sigma_ln


def _segment_lengths_fixed_step(length_nm: float) -> np.ndarray:
    """Return 50 nm contour steps plus a terminal remainder segment."""
    if length_nm <= 0.0:
        raise ValueError(f"CNT length must be positive, got {length_nm}")

    n_full = int(math.floor(length_nm / SEG_LEN_UNIT_NM))
    remainder = float(length_nm - n_full * SEG_LEN_UNIT_NM)

    pieces = [SEG_LEN_UNIT_NM] * n_full
    if remainder > 1e-9 or not pieces:
        pieces.append(remainder if remainder > 1e-9 else length_nm)
    return np.asarray(pieces, dtype=np.float64)


def _cnt_flags(
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    radius_nm: float,
    rve_nm: float,
) -> int:
    """Six-face boundary flags from all segment endpoints of one CNT."""
    pts = np.vstack((seg_starts, seg_ends))
    flags = 0
    if np.any(pts[:, 0] < radius_nm):
        flags |= FACE_X_LOW
    if np.any(pts[:, 0] > rve_nm - radius_nm):
        flags |= FACE_X_HIGH
    if np.any(pts[:, 1] < radius_nm):
        flags |= FACE_Y_LOW
    if np.any(pts[:, 1] > rve_nm - radius_nm):
        flags |= FACE_Y_HIGH
    if np.any(pts[:, 2] < radius_nm):
        flags |= FACE_Z_LOW
    if np.any(pts[:, 2] > rve_nm - radius_nm):
        flags |= FACE_Z_HIGH
    return flags


def _sample_length(
    rng: np.random.Generator,
    mu_ln: float,
    sigma_ln: float,
    cdf_lo: float,
    cdf_rng: float,
    ndtri_func,
) -> float:
    if sigma_ln <= 0.0:
        return float(np.exp(mu_ln))

    u = float(rng.uniform(0.0, 1.0))
    u_scaled = cdf_lo + u * cdf_rng
    u_scaled = max(1e-14, min(u_scaled, 1.0 - 1e-14))
    length = float(np.exp(mu_ln + sigma_ln * float(ndtri_func(u_scaled))))
    return max(L_MIN_NM, min(length, L_MAX_NM))


def _run_realization(args: tuple[float, float, int, float, int]) -> RealizationResult:
    """One volume-stopped CNT realization with variable segment counts."""
    from scipy.special import ndtr, ndtri

    mu_ln, sigma_ln, n_max, phi_target, seed = args

    t0 = time.time()
    rng = np.random.default_rng(seed)

    r_cnt = CNT_DIAM_NM / 2.0
    connect_d = TUNNEL_NM + 2.0 * r_cnt
    max_seg_per_cnt = int(math.ceil(L_MAX_NM / SEG_LEN_UNIT_NM))
    max_total_segs = n_max * max_seg_per_cnt
    cell_size = SEG_LEN_UNIT_NM + connect_d

    v_rve = L_RVE**3
    v_target = phi_target * v_rve
    v_prefactor = math.pi * r_cnt**2

    if sigma_ln > 0.0:
        cdf_lo = float(ndtr((np.log(L_MIN_NM) - mu_ln) / sigma_ln))
        cdf_hi = float(ndtr((np.log(L_MAX_NM) - mu_ln) / sigma_ln))
        cdf_rng = cdf_hi - cdf_lo
    else:
        cdf_lo = 0.0
        cdf_rng = 0.0

    cos_b = WAVINESS
    sin_b = float(np.sqrt(max(0.0, 1.0 - cos_b**2)))

    uf = _UnionFind(n_max)
    seg_hash = _SpatialHash(cell_size=cell_size)

    act_starts = np.empty((max_total_segs, 3), dtype=np.float64)
    act_ends = np.empty((max_total_segs, 3), dtype=np.float64)
    seg_to_cnt = np.empty(max_total_segs, dtype=np.intp)
    seg_offsets = np.zeros(n_max + 1, dtype=np.intp)

    lengths = np.empty(n_max, dtype=np.float64)
    n_seg_per_cnt = np.empty(n_max, dtype=np.intp)

    n_total_segs = 0
    vol_total = 0.0

    for cnt_id in range(n_max):
        length_i = _sample_length(rng, mu_ln, sigma_ln, cdf_lo, cdf_rng, ndtri)
        seg_lens = _segment_lengths_fixed_step(length_i)
        n_seg_i = int(len(seg_lens))

        if n_total_segs + n_seg_i > max_total_segs:
            return _finalize_result(
                False,
                "segment_capacity",
                vol_total,
                cnt_id,
                n_total_segs,
                lengths,
                n_seg_per_cnt,
                act_starts,
                act_ends,
                t0,
            )

        segs_s = np.empty((n_seg_i, 3), dtype=np.float64)
        segs_e = np.empty((n_seg_i, 3), dtype=np.float64)

        pos = rng.uniform(0.0, L_RVE, size=3)
        d_vec = rng.standard_normal(3)
        d_vec /= np.linalg.norm(d_vec)

        for s, seg_len in enumerate(seg_lens):
            segs_s[s] = pos
            segs_e[s] = pos + d_vec * float(seg_len)
            pos = segs_e[s]
            if s < n_seg_i - 1:
                perturb = rng.standard_normal(3)
                perturb -= np.dot(perturb, d_vec) * d_vec
                norm_p = float(np.linalg.norm(perturb))
                if norm_p > 1e-12:
                    perturb /= norm_p
                d_vec = cos_b * d_vec + sin_b * perturb
                d_vec /= np.linalg.norm(d_vec)

        seg_offsets[cnt_id] = n_total_segs
        seg_offsets[cnt_id + 1] = n_total_segs + n_seg_i
        lengths[cnt_id] = length_i
        n_seg_per_cnt[cnt_id] = n_seg_i

        mids = 0.5 * (segs_s + segs_e)
        flags = _cnt_flags(segs_s, segs_e, r_cnt, L_RVE)
        uf.set_boundary(cnt_id, flags)
        percolated = uf.is_root_percolating(cnt_id)

        if cnt_id > 0:
            cand_set: set[int] = set()
            for s in range(n_seg_i):
                for flat_id in seg_hash.query(mids[s]):
                    cand = int(seg_to_cnt[flat_id])
                    if cand != cnt_id:
                        cand_set.add(cand)

            if cand_set:
                cand_arr = np.fromiter(sorted(cand_set), dtype=np.intp)
                per_cand_lengths = (
                    seg_offsets[cand_arr + 1] - seg_offsets[cand_arr]
                ).astype(np.intp)
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

                for s in range(n_seg_i):
                    dists = _batch_seg_dist(segs_s[s], segs_e[s], cand_starts, cand_ends)
                    per_cand_min = np.minimum.reduceat(dists, starts)
                    connected |= per_cand_min < connect_d

                for j_idx in np.flatnonzero(connected):
                    if uf.union(cnt_id, int(cand_arr[j_idx])):
                        percolated = True

        base = n_total_segs
        act_starts[base:base + n_seg_i] = segs_s
        act_ends[base:base + n_seg_i] = segs_e
        seg_to_cnt[base:base + n_seg_i] = cnt_id
        for s in range(n_seg_i):
            seg_hash.insert(base + s, mids[s])

        n_total_segs += n_seg_i
        vol_total += v_prefactor * length_i

        if percolated:
            return _finalize_result(
                True,
                "",
                vol_total,
                cnt_id + 1,
                n_total_segs,
                lengths,
                n_seg_per_cnt,
                act_starts,
                act_ends,
                t0,
            )

        if vol_total >= v_target:
            return _finalize_result(
                False,
                "phi_max",
                vol_total,
                cnt_id + 1,
                n_total_segs,
                lengths,
                n_seg_per_cnt,
                act_starts,
                act_ends,
                t0,
            )

    return _finalize_result(
        False,
        "n_max",
        vol_total,
        n_max,
        n_total_segs,
        lengths,
        n_seg_per_cnt,
        act_starts,
        act_ends,
        t0,
    )


def _finalize_result(
    percolated: bool,
    failure_reason: str,
    vol_total: float,
    n_cnt: int,
    n_total_segs: int,
    lengths: np.ndarray,
    n_seg_per_cnt: np.ndarray,
    act_starts: np.ndarray,
    act_ends: np.ndarray,
    t0: float,
) -> RealizationResult:
    if n_cnt <= 0 or n_total_segs <= 0:
        return RealizationResult(
            percolated=percolated,
            failure_reason=failure_reason or "empty",
            phi_stop_vol=0.0,
            n_cnt=0,
            n_total_segs=0,
            mean_n_seg_per_cnt=float("nan"),
            max_n_seg_per_cnt=0,
            mean_segment_len_nm=float("nan"),
            max_segment_len_nm=float("nan"),
            mean_L_realized_nm=float("nan"),
            std_L_realized_nm=float("nan"),
            elapsed_s=time.time() - t0,
        )

    seg_lens = np.linalg.norm(act_ends[:n_total_segs] - act_starts[:n_total_segs], axis=1)
    phi_stop = vol_total / L_RVE**3
    return RealizationResult(
        percolated=percolated,
        failure_reason=failure_reason,
        phi_stop_vol=float(phi_stop),
        n_cnt=int(n_cnt),
        n_total_segs=int(n_total_segs),
        mean_n_seg_per_cnt=float(np.mean(n_seg_per_cnt[:n_cnt])),
        max_n_seg_per_cnt=int(np.max(n_seg_per_cnt[:n_cnt])),
        mean_segment_len_nm=float(np.mean(seg_lens)),
        max_segment_len_nm=float(np.max(seg_lens)),
        mean_L_realized_nm=float(np.mean(lengths[:n_cnt])),
        std_L_realized_nm=float(np.std(lengths[:n_cnt], ddof=1)) if n_cnt > 1 else 0.0,
        elapsed_s=time.time() - t0,
    )


def _n_max_estimate(mu_L: float, phi_target: float) -> int:
    r = CNT_DIAM_NM / 2.0
    v_particle = math.pi * r**2 * mu_L
    return max(200, int(math.ceil(phi_target * L_RVE**3 / v_particle * 2.5)))


def _available_memory_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        return 16.0


def _worker_count(n_max: int, requested: int) -> int:
    if requested <= 1:
        return 1

    max_seg_per_cnt = int(math.ceil(L_MAX_NM / SEG_LEN_UNIT_NM))
    max_total_segs = n_max * max_seg_per_cnt
    bytes_per_worker = max_total_segs * 64 + n_max * 128
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
                done.add((int(float(row["mu_L_nm"])), int(float(row["sigma_L_nm"]))))
            except (KeyError, ValueError):
                pass
    return done


def _write_realizations(
    path: Path,
    mu_L: int,
    sigma_L: int,
    seeds: list[int],
    results: list[RealizationResult],
    phi_max_vol: float,
) -> None:
    cv = sigma_L / mu_L
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        for i, (seed, res) in enumerate(zip(seeds, results), start=1):
            phi_censored = res.phi_stop_vol if res.percolated else phi_max_vol
            writer.writerow(
                [
                    mu_L,
                    sigma_L,
                    f"{cv:.4f}",
                    i,
                    seed,
                    res.percolated,
                    res.failure_reason,
                    f"{phi_censored:.8g}",
                    f"{phi_to_wt_pct(phi_censored):.6g}",
                    f"{res.phi_stop_vol:.8g}",
                    f"{phi_to_wt_pct(res.phi_stop_vol):.6g}",
                    res.n_cnt,
                    res.n_total_segs,
                    f"{res.mean_n_seg_per_cnt:.8g}",
                    res.max_n_seg_per_cnt,
                    f"{res.mean_segment_len_nm:.8g}",
                    f"{res.max_segment_len_nm:.8g}",
                    f"{res.mean_L_realized_nm:.8g}",
                    f"{res.std_L_realized_nm:.8g}",
                    SEGMENTATION_RULE,
                    SEG_LEN_UNIT_NM,
                    True,
                    L_RVE,
                    L_MIN_NM,
                    L_MAX_NM,
                    CNT_DIAM_NM,
                    WAVINESS,
                    TUNNEL_NM,
                    PHI_MAX_WT,
                    f"{res.elapsed_s:.3f}",
                ]
            )


def _write_summary(
    path: Path,
    mu_L: int,
    sigma_L: int,
    results: list[RealizationResult],
    phi_max_vol: float,
    phi_exp_vol: float,
    phi_1wt_vol: float,
    elapsed_s: float,
) -> None:
    cv = sigma_L / mu_L
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
    phi_an = phi_c_analytic(mu_L, cv)

    phi_act = np.asarray([res.phi_stop_vol for res in results], dtype=float)
    n_cnt = np.asarray([res.n_cnt for res in results], dtype=float)
    n_total_segs = np.asarray([res.n_total_segs for res in results], dtype=float)
    mean_n_seg = np.asarray([res.mean_n_seg_per_cnt for res in results], dtype=float)
    max_n_seg = np.asarray([res.max_n_seg_per_cnt for res in results], dtype=float)
    mean_seg_len = np.asarray([res.mean_segment_len_nm for res in results], dtype=float)
    max_seg_len = np.asarray([res.max_segment_len_nm for res in results], dtype=float)

    with path.open("a", newline="") as f:
        csv.writer(f).writerow(
            [
                mu_L,
                sigma_L,
                f"{cv:.4f}",
                f"{phi_c_vol:.8g}",
                f"{phi_to_wt_pct(phi_c_vol):.6g}",
                f"{phi_to_wt_pct(phi_q25_vol):.6g}",
                f"{phi_to_wt_pct(phi_q75_vol):.6g}",
                f"{float(np.mean(phi_vals <= phi_exp_vol)):.4f}",
                f"{float(np.mean(phi_vals <= phi_1wt_vol)):.4f}",
                n_perc,
                n_total,
                n_censored,
                f"{phi_an:.8g}",
                f"{phi_to_wt_pct(phi_an):.6g}",
                f"{float(np.mean(phi_act)):.8g}",
                f"{float(np.std(phi_act, ddof=1)) if len(phi_act) > 1 else 0.0:.8g}",
                f"{float(np.mean(n_cnt)):.8g}",
                f"{float(np.std(n_cnt, ddof=1)) if len(n_cnt) > 1 else 0.0:.8g}",
                f"{float(np.mean(n_total_segs)):.8g}",
                f"{float(np.mean(mean_n_seg)):.8g}",
                f"{int(np.max(max_n_seg))}",
                f"{float(np.mean(mean_seg_len)):.8g}",
                f"{float(np.max(max_seg_len)):.8g}",
                SEGMENTATION_RULE,
                SEG_LEN_UNIT_NM,
                True,
                L_RVE,
                L_MIN_NM,
                L_MAX_NM,
                CNT_DIAM_NM,
                WAVINESS,
                TUNNEL_NM,
                PHI_MAX_WT,
                PHI_C_EXP_WT,
                f"{elapsed_s:.3f}",
            ]
        )


def _init_worker(l_rve_nm: float) -> None:
    global L_RVE
    L_RVE = float(l_rve_nm)


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

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(L_RVE,),
    ) as pool:
        future_to_idx = {
            pool.submit(_run_realization, task): idx for idx, task in enumerate(tasks, start=1)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            record(idx, future.result())

    return [r for r in results if r is not None]


def _run_point(
    mu_L: int,
    sigma_L: int,
    n_real: int,
    requested_workers: int,
    summary_csv: Path,
    real_csv: Path,
    progress_every: int,
) -> None:
    phi_max_vol = wt_frac_to_phi(PHI_MAX_WT / 100.0)
    phi_exp_vol = wt_frac_to_phi(PHI_C_EXP_WT / 100.0)
    phi_1wt_vol = wt_frac_to_phi(0.01)
    cv = sigma_L / mu_L

    mu_ln, sigma_ln = _solve_lognormal_truncated(mu_L, sigma_L, L_MIN_NM, L_MAX_NM)
    n_max = _n_max_estimate(mu_L, phi_max_vol)
    n_workers = _worker_count(n_max, requested_workers)

    seeds = [
        SEED_BASE + (int(mu_L) * 1000 + int(sigma_L)) * 1000 + r
        for r in range(n_real)
    ]
    tasks = [(mu_ln, sigma_ln, n_max, phi_max_vol, seed) for seed in seeds]

    print(
        f"CNT mu={mu_L:4d} nm  sigma={sigma_L:3d} nm  CV={cv:.3f}  "
        f"n={n_real:3d}  n_max={n_max:7d}  workers={n_workers:2d}",
        flush=True,
    )
    t0 = time.time()
    results = _run_tasks(tasks, n_workers, progress_every)
    elapsed = time.time() - t0

    _write_realizations(real_csv, mu_L, sigma_L, seeds, results, phi_max_vol)
    _write_summary(
        summary_csv,
        mu_L,
        sigma_L,
        results,
        phi_max_vol,
        phi_exp_vol,
        phi_1wt_vol,
        elapsed,
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
    global SUMMARY_CSV, REAL_CSV, L_RVE

    if args.l_rve_nm is not None:
        if args.l_rve_nm <= 0.0:
            sys.exit(f"error: --l-rve-nm must be > 0 (got {args.l_rve_nm:g}).")
        L_RVE = float(args.l_rve_nm)

    single_point = args.only_mu is not None or args.only_sigma is not None
    if single_point and (args.only_mu is None or args.only_sigma is None):
        sys.exit("error: --only-mu and --only-sigma must be given together.")

    if args.tag:
        safe_tag = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in args.tag)
        suffix = f"_{safe_tag}"
    elif args.quick:
        suffix = "_quick"
    else:
        suffix = f"_{PRODUCTION_TAG}"

    SUMMARY_CSV = OUT_DIR / f"cnt_phic_sweep_fixedstep{suffix}.csv"
    REAL_CSV = OUT_DIR / f"cnt_phic_sweep_fixedstep{suffix}_realizations.csv"

    if single_point:
        mu_vals = (int(args.only_mu),)
        sigma_vals = (int(args.only_sigma),)
    else:
        mu_vals = (500,) if args.quick else MU_L_VALS
        sigma_vals = (200,) if args.quick else SIGMA_L_VALS
    n_real = args.n_real if args.n_real is not None else (
        N_REAL_QUICK if args.quick else N_REAL_PRODUCTION
    )
    requested_workers = min(args.workers, os.cpu_count() or args.workers)

    _ensure_header(SUMMARY_CSV, SUMMARY_HEADER)
    _ensure_header(REAL_CSV, REAL_HEADER)
    done = _load_done(SUMMARY_CSV)

    print("=" * 78)
    print("Paper CNT phi_c sweep: polydisperse length, fixed-step segmentation")
    print(f"Mode: {'quick canonical point' if args.quick else 'production grid'}")
    print(f"Output suffix: {suffix or '[none]'}")
    print(f"RVE={L_RVE:g} nm  d_CNT={CNT_DIAM_NM:g} nm  w={WAVINESS:g}  tunnel={TUNNEL_NM:g} nm")
    print(f"L range=[{L_MIN_NM:g}, {L_MAX_NM:g}] nm")
    print(f"Segmentation: {SEGMENTATION_RULE}, unit={SEG_LEN_UNIT_NM:g} nm")
    print(f"phi_max={PHI_MAX_WT:g} wt%  target line={PHI_C_EXP_WT:g} wt%")
    print(f"N_real per point={n_real}  requested workers={requested_workers}")
    print("=" * 78)

    for mu_L in mu_vals:
        for sigma_L in sigma_vals:
            if not single_point and sigma_L > mu_L - SIGMA_CONSTRAINT_NM:
                continue
            key = (int(mu_L), int(sigma_L))
            if key in done:
                print(f"CNT mu={mu_L:4d} nm  sigma={sigma_L:3d} nm  skip (already in CSV)")
                continue
            _run_point(
                int(mu_L),
                int(sigma_L),
                int(n_real),
                requested_workers,
                SUMMARY_CSV,
                REAL_CSV,
                args.progress_every,
            )

    print("=" * 78)
    print("Done.")
    print(f"Summary      : {SUMMARY_CSV}")
    print(f"Realizations : {REAL_CSV}")
    print("=" * 78)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the publication polydisperse CNT phi_c sweep."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only the canonical (mu=500, sigma=200) point with small N.",
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
        "--progress-every",
        type=int,
        default=10,
        help="Print progress after this many completed realizations; use 0 to disable.",
    )
    parser.add_argument(
        "--l-rve-nm",
        type=float,
        default=None,
        help=(
            "Override the cubic RVE edge length in nm (production default "
            f"{L_RVE:g}). Used for the targeted finite-size robustness check; the "
            "numeric kernel, seeds, and distribution are otherwise unchanged."
        ),
    )
    parser.add_argument(
        "--only-mu",
        type=int,
        default=None,
        help="Run a single grid point at this mu_L (nm) instead of the full sweep.",
    )
    parser.add_argument(
        "--only-sigma",
        type=int,
        default=None,
        help="Run a single grid point at this sigma_L (nm); requires --only-mu.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
