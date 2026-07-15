#!/usr/bin/env python3
"""Hybrid CNT-threshold boundary sweep for the paper.

For each fixed CB loading, this wrapper places a penetrable CB background and
then adds CNTs until the hybrid network percolates.  It uses the shared
percolation engine, so the CNT/CB conventions match the Fig. 6 and
Fig. 7 production wrappers:

* CNT length: target 500/300 nm, truncated over [50, 1500] nm;
* CNT segmentation: fixed 50 nm contour steps plus terminal remainder;
* CB diameter: target 148/83 nm, truncated over [20, 500] nm;
* penetrable particle ensemble and boundary-flagged union-find detection.

The default sweep compares four distribution modes:

* MM: CB monodisperse, CNT monodisperse;
* PM: CB polydisperse, CNT monodisperse;
* MP: CB monodisperse, CNT polydisperse;
* PP: CB polydisperse, CNT polydisperse.

Outputs:
  data/processed/percolation/hybrid_phic_boundary_<tag>.csv
  data/processed/percolation/hybrid_phic_boundary_<tag>_realizations.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# Put the repo root on sys.path so a bare checkout runs without pip install;
# the cntcb imports below intentionally follow this bootstrap (hence E402).
ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT
OUT_DIR = PAPER / "data" / "processed" / "percolation"

sys.path.insert(0, str(ROOT))

from cntcb.engines.percolation_engine import (  # noqa: E402
    HybridPercolationResult,
    estimate_cb_n_max,
    estimate_cnt_n_max,
    run_hybrid_cnt_threshold_realization,
    solve_truncated_lognormal,
)


RHO_CNT = 1.75
RHO_CB = 1.80
RHO_PEI = 1.27

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
PHI_CNT_MAX_WT = 2.0
CB_WT_VALS = (1.0, 3.0, 5.0, 10.0, 20.0, 30.0)

MODE_CONFIGS = {
    "MM": ("CB mono / CNT mono", 0.0, 0.0),
    "PM": ("CB poly / CNT mono", CB_SIGMA_D_NM, 0.0),
    "MP": ("CB mono / CNT poly", 0.0, CNT_SIGMA_L_NM),
    "PP": ("CB poly / CNT poly", CB_SIGMA_D_NM, CNT_SIGMA_L_NM),
}
DEFAULT_MODES = ("MM", "PM", "MP", "PP")

N_REAL_PILOT = 20
N_WORKERS_DEFAULT = 10
SEED_BASE = 20260608
SAFETY = 2.5

SUMMARY_HEADER = [
    "mode",
    "mode_label",
    "w_cb_wt_pct",
    "phi_cb_vol",
    "phi_cnt_c_median_vol",
    "phi_cnt_c_median_wt_pct",
    "phi_cnt_c_mean_vol",
    "phi_cnt_c_mean_wt_pct",
    "phi_cnt_c_std_wt_pct",
    "phi_cnt_c_q25_wt_pct",
    "phi_cnt_c_q75_wt_pct",
    "phi_cnt_c_q05_wt_pct",
    "phi_cnt_c_q95_wt_pct",
    "n_ok",
    "n_fail",
    "n_cb_percolated",
    "n_censored",
    "n_total",
    "n_cb_mean",
    "n_cnt_mean",
    "n_total_segs_mean",
    "mean_d_realized_nm",
    "std_d_realized_nm",
    "mean_L_realized_nm",
    "std_L_realized_nm",
    "mean_n_seg_per_cnt",
    "max_n_seg_per_cnt",
    "cb_parent_mu_d_nm",
    "cb_parent_sigma_d_nm",
    "cnt_parent_mu_L_nm",
    "cnt_parent_sigma_L_nm",
    "L_RVE_nm",
    "tunnel_nm",
    "phi_cnt_max_wt_pct",
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
    "elapsed_s",
]

REAL_HEADER = [
    "mode",
    "mode_label",
    "w_cb_wt_pct",
    "phi_cb_vol",
    "realization",
    "seed",
    "percolated",
    "failure_reason",
    "phi_cnt_stop_vol",
    "phi_cnt_stop_wt_pct",
    "phi_cb_stop_vol",
    "n_cb",
    "n_cnt",
    "n_total_segs",
    "mean_d_realized_nm",
    "std_d_realized_nm",
    "mean_L_realized_nm",
    "std_L_realized_nm",
    "mean_n_seg_per_cnt",
    "max_n_seg_per_cnt",
    "CB_sigma_d_nm",
    "CNT_sigma_L_nm",
    "elapsed_s",
]


def wt_cb_to_phi_binary(w_cb_wt_pct: float) -> float:
    """Convert CB wt.% in a binary CB/PEI background to volume fraction."""

    w = w_cb_wt_pct / 100.0
    return (w / RHO_CB) / (w / RHO_CB + (1.0 - w) / RHO_PEI)


def phi_cnt_to_wt_pct(phi_cnt: float, phi_cb: float) -> float:
    """Convert CNT volume fraction to wt.% in a ternary CNT/CB/PEI system."""

    phi_matrix = 1.0 - phi_cnt - phi_cb
    mass_cnt = phi_cnt * RHO_CNT
    mass_total = mass_cnt + phi_cb * RHO_CB + phi_matrix * RHO_PEI
    return 100.0 * mass_cnt / mass_total


def wt_cnt_to_phi_at_fixed_cb(w_cnt_wt_pct: float, phi_cb: float) -> float:
    """CNT wt.% cap converted to CNT volume fraction at fixed CB volume."""

    w = w_cnt_wt_pct / 100.0
    constant_mass = RHO_PEI + phi_cb * (RHO_CB - RHO_PEI)
    denom = RHO_CNT - w * (RHO_CNT - RHO_PEI)
    return w * constant_mass / denom


def _parse_float_list(text: str) -> tuple[float, ...]:
    vals = []
    for chunk in text.split(","):
        item = chunk.strip()
        if item:
            vals.append(float(item))
    if not vals:
        raise ValueError("List argument did not contain any numeric value.")
    return tuple(vals)


def _parse_modes(text: str) -> tuple[str, ...]:
    modes = []
    for chunk in text.split(","):
        mode = chunk.strip().upper()
        if not mode:
            continue
        if mode not in MODE_CONFIGS:
            raise ValueError(
                f"Unknown mode {mode!r}; expected one of {', '.join(MODE_CONFIGS)}."
            )
        modes.append(mode)
    if not modes:
        raise ValueError("Mode list is empty.")
    return tuple(modes)


def _safe_tag(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    if not safe:
        raise ValueError("--tag must contain at least one safe character.")
    return safe


def _nanmean_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return float("nan")
    return float(np.mean(finite))


def _output_paths(out_dir: Path, tag: str) -> tuple[Path, Path]:
    base = f"hybrid_phic_boundary_{_safe_tag(tag)}"
    return out_dir / f"{base}.csv", out_dir / f"{base}_realizations.csv"


def _run_one(
    args: tuple[
        object,
        object,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        int,
        int,
        int,
    ],
) -> HybridPercolationResult:
    (
        cb_model,
        cnt_model,
        phi_cb,
        phi_cnt_max,
        l_rve_nm,
        cnt_diam_nm,
        waviness,
        tunnel_nm,
        seg_len_unit_nm,
        n_cb_max,
        n_cnt_max,
        seed,
    ) = args
    return run_hybrid_cnt_threshold_realization(
        cb_model=cb_model,
        cnt_model=cnt_model,
        phi_cb_vol=phi_cb,
        phi_cnt_max_vol=phi_cnt_max,
        l_rve_nm=l_rve_nm,
        cnt_diam_nm=cnt_diam_nm,
        waviness=waviness,
        tunnel_nm=tunnel_nm,
        seg_len_unit_nm=seg_len_unit_nm,
        seed=seed,
        n_cb_max=n_cb_max,
        n_cnt_max=n_cnt_max,
    )


def _run_tasks(
    tasks: list[tuple],
    workers: int,
    progress_every: int,
) -> list[HybridPercolationResult]:
    results: list[HybridPercolationResult | None] = [None] * len(tasks)
    t0 = time.time()
    n_done = 0

    def record(idx: int, result: HybridPercolationResult) -> None:
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
            record(idx, _run_one(task))
        return [r for r in results if r is not None]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(_run_one, task): idx for idx, task in enumerate(tasks, start=1)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            record(idx, future.result())

    return [r for r in results if r is not None]


def _write_realizations(
    path: Path,
    mode: str,
    mode_label: str,
    w_cb: float,
    phi_cb: float,
    cb_sigma_d_nm: float,
    cnt_sigma_l_nm: float,
    seeds: list[int],
    results: list[HybridPercolationResult],
) -> None:
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        for idx, (seed, res) in enumerate(zip(seeds, results), start=1):
            writer.writerow(
                [
                    mode,
                    mode_label,
                    f"{w_cb:.6g}",
                    f"{phi_cb:.8g}",
                    idx,
                    seed,
                    res.percolated,
                    res.failure_reason,
                    f"{res.phi_cnt_stop_vol:.8g}",
                    f"{phi_cnt_to_wt_pct(res.phi_cnt_stop_vol, phi_cb):.8g}",
                    f"{res.phi_cb_stop_vol:.8g}",
                    res.n_cb,
                    res.n_cnt,
                    res.n_total_segs,
                    f"{res.mean_d_realized_nm:.8g}",
                    f"{res.std_d_realized_nm:.8g}",
                    f"{res.mean_L_realized_nm:.8g}",
                    f"{res.std_L_realized_nm:.8g}",
                    f"{res.mean_n_seg_per_cnt:.8g}",
                    res.max_n_seg_per_cnt,
                    f"{cb_sigma_d_nm:.8g}",
                    f"{cnt_sigma_l_nm:.8g}",
                    f"{res.elapsed_s:.3f}",
                ]
            )


def _write_summary_row(
    path: Path,
    mode: str,
    mode_label: str,
    w_cb: float,
    phi_cb: float,
    results: list[HybridPercolationResult],
    cb_model,
    cnt_model,
    cb_sigma_d_nm: float,
    cnt_sigma_l_nm: float,
    phi_cnt_max_wt: float,
    elapsed_s: float,
) -> None:
    ok = [res for res in results if res.percolated]
    failed = [res for res in results if not res.percolated]
    cnt_vals = np.asarray([res.phi_cnt_stop_vol for res in ok], dtype=float)
    cnt_wt_vals = np.asarray(
        [phi_cnt_to_wt_pct(res.phi_cnt_stop_vol, phi_cb) for res in ok],
        dtype=float,
    )

    if len(ok) > 0:
        median_vol = float(np.median(cnt_vals))
        mean_vol = float(np.mean(cnt_vals))
        median_wt = float(np.median(cnt_wt_vals))
        mean_wt = float(np.mean(cnt_wt_vals))
        std_wt = float(np.std(cnt_wt_vals, ddof=1)) if len(ok) > 1 else 0.0
        q05, q25, q75, q95 = np.quantile(cnt_wt_vals, [0.05, 0.25, 0.75, 0.95])
    else:
        median_vol = mean_vol = median_wt = mean_wt = std_wt = float("nan")
        q05 = q25 = q75 = q95 = float("nan")

    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                mode,
                mode_label,
                f"{w_cb:.6g}",
                f"{phi_cb:.8g}",
                f"{median_vol:.8g}",
                f"{median_wt:.8g}",
                f"{mean_vol:.8g}",
                f"{mean_wt:.8g}",
                f"{std_wt:.8g}",
                f"{q25:.8g}",
                f"{q75:.8g}",
                f"{q05:.8g}",
                f"{q95:.8g}",
                len(ok),
                len(failed),
                sum(res.failure_reason == "cb_percolated" for res in ok),
                sum(res.failure_reason in {"phi_cnt_max", "n_cnt_max"} for res in failed),
                len(results),
                f"{np.mean([res.n_cb for res in results]):.8g}",
                f"{np.mean([res.n_cnt for res in results]):.8g}",
                f"{np.mean([res.n_total_segs for res in results]):.8g}",
                f"{_nanmean_or_nan([res.mean_d_realized_nm for res in results]):.8g}",
                f"{_nanmean_or_nan([res.std_d_realized_nm for res in results]):.8g}",
                f"{_nanmean_or_nan([res.mean_L_realized_nm for res in results]):.8g}",
                f"{_nanmean_or_nan([res.std_L_realized_nm for res in results]):.8g}",
                f"{_nanmean_or_nan([res.mean_n_seg_per_cnt for res in results]):.8g}",
                max(res.max_n_seg_per_cnt for res in results),
                f"{cb_model.parent_mu_nm:.8g}",
                f"{cb_model.parent_sigma_nm:.8g}",
                f"{cnt_model.parent_mu_nm:.8g}",
                f"{cnt_model.parent_sigma_nm:.8g}",
                L_RVE_NM,
                TUNNEL_NM,
                phi_cnt_max_wt,
                CB_MU_D_NM,
                cb_sigma_d_nm,
                CB_D_MIN_NM,
                CB_D_MAX_NM,
                CNT_MU_L_NM,
                cnt_sigma_l_nm,
                CNT_L_MIN_NM,
                CNT_L_MAX_NM,
                CNT_DIAM_NM,
                CNT_WAVINESS,
                SEG_LEN_UNIT_NM,
                f"{elapsed_s:.3f}",
            ]
        )


def _prepare_output(path: Path, header: list[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Use --overwrite or choose a different --tag."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        csv.writer(f).writerow(header)


def run(args: argparse.Namespace) -> None:
    cb_wt_vals = _parse_float_list(args.cb_wt)
    modes = _parse_modes(args.modes)
    out_dir = Path(args.out_dir).expanduser().resolve()
    summary_csv, real_csv = _output_paths(out_dir, args.tag)

    _prepare_output(summary_csv, SUMMARY_HEADER, args.overwrite)
    _prepare_output(real_csv, REAL_HEADER, args.overwrite)

    print("=" * 78)
    print("Hybrid CNT-threshold boundary sweep")
    print(f"  tag={args.tag}  N={args.n_real}  workers={args.workers}")
    print(f"  modes={','.join(modes)}")
    print(f"  CB : mu={CB_MU_D_NM:.0f} nm over [{CB_D_MIN_NM:.0f}, {CB_D_MAX_NM:.0f}]")
    print(f"  CNT: mu={CNT_MU_L_NM:.0f} nm over [{CNT_L_MIN_NM:.0f}, {CNT_L_MAX_NM:.0f}]")
    print(f"  L_RVE={L_RVE_NM:.0f} nm  tunnel={TUNNEL_NM:.0f} nm  phi_CNT,max={args.phi_cnt_max_wt:.2f} wt%")
    print(f"  summary: {summary_csv}")
    print(f"  per-realization: {real_csv}")
    print("=" * 78)

    for mode_idx, mode in enumerate(modes):
        mode_label, cb_sigma_d_nm, cnt_sigma_l_nm = MODE_CONFIGS[mode]
        cb_model = solve_truncated_lognormal(
            CB_MU_D_NM, cb_sigma_d_nm, CB_D_MIN_NM, CB_D_MAX_NM
        )
        cnt_model = solve_truncated_lognormal(
            CNT_MU_L_NM, cnt_sigma_l_nm, CNT_L_MIN_NM, CNT_L_MAX_NM
        )
        print(
            f"\n--- mode {mode}: {mode_label} "
            f"(sigma_CB={cb_sigma_d_nm:.0f} nm, sigma_CNT={cnt_sigma_l_nm:.0f} nm) ---"
        )

        for w_idx, w_cb in enumerate(cb_wt_vals):
            phi_cb = wt_cb_to_phi_binary(w_cb)
            phi_cnt_max = wt_cnt_to_phi_at_fixed_cb(args.phi_cnt_max_wt, phi_cb)
            n_cb_max = estimate_cb_n_max(cb_model, phi_cb, L_RVE_NM, safety=SAFETY)
            n_cnt_max = estimate_cnt_n_max(
                cnt_model, phi_cnt_max, L_RVE_NM, CNT_DIAM_NM, safety=SAFETY
            )
            seeds = [
                args.seed_base + mode_idx * 1_000_000 + w_idx * 100_000 + r
                for r in range(args.n_real)
            ]
            tasks = [
                (
                    cb_model,
                    cnt_model,
                    phi_cb,
                    phi_cnt_max,
                    L_RVE_NM,
                    CNT_DIAM_NM,
                    CNT_WAVINESS,
                    TUNNEL_NM,
                    SEG_LEN_UNIT_NM,
                    n_cb_max,
                    n_cnt_max,
                    seed,
                )
                for seed in seeds
            ]

            print(
                f"w_CB={w_cb:g} wt%  phi_CB={phi_cb:.5f}  "
                f"n_cb_max={n_cb_max}  n_cnt_max={n_cnt_max}"
            )
            t0 = time.time()
            results = _run_tasks(tasks, args.workers, args.progress_every)
            elapsed = time.time() - t0

            ok = [res for res in results if res.percolated]
            if ok:
                vals = np.asarray(
                    [phi_cnt_to_wt_pct(res.phi_cnt_stop_vol, phi_cb) for res in ok],
                    dtype=float,
                )
                print(
                    f"  n_ok={len(ok)}/{len(results)}  "
                    f"median phi_c,CNT={np.median(vals):.4f} wt%  "
                    f"IQR=[{np.quantile(vals, 0.25):.4f}, {np.quantile(vals, 0.75):.4f}]  "
                    f"elapsed={elapsed:.1f}s"
                )
            else:
                print(f"  n_ok=0/{len(results)}  elapsed={elapsed:.1f}s")

            _write_realizations(
                real_csv,
                mode,
                mode_label,
                w_cb,
                phi_cb,
                cb_sigma_d_nm,
                cnt_sigma_l_nm,
                seeds,
                results,
            )
            _write_summary_row(
                summary_csv,
                mode,
                mode_label,
                w_cb,
                phi_cb,
                results,
                cb_model,
                cnt_model,
                cb_sigma_d_nm,
                cnt_sigma_l_nm,
                args.phi_cnt_max_wt,
                elapsed,
            )

    print("\nDone.")
    print(f"Summary CSV: {summary_csv}")
    print(f"Realization CSV: {real_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="pilot", help="Output tag.")
    parser.add_argument("--n-real", type=int, default=N_REAL_PILOT)
    parser.add_argument("--workers", type=int, default=N_WORKERS_DEFAULT)
    parser.add_argument("--cb-wt", default=",".join(f"{v:g}" for v in CB_WT_VALS))
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--phi-cnt-max-wt", type=float, default=PHI_CNT_MAX_WT)
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
