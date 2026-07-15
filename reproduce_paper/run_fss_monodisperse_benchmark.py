#!/usr/bin/env python3
"""Monodisperse percolation/FSS benchmark for the paper Results section.

This paper reproduction wrapper runs the S1/S2-style finite-size benchmark
with the geometries chosen for the manuscript:

  CB  : monodisperse effective aggregate diameter d = 150 nm
  CNT : L = 500 nm, d = 10 nm, w = {1.0, 0.7, 0.3}

The CNT contour is represented with the publication segmentation convention:
fixed 50 nm contour steps. For L = 500 nm this gives exactly 10 segments and
no terminal remainder.

Outputs:
  data/processed/percolation/fss_monodisperse_cb_d150.csv
  data/processed/percolation/fss_monodisperse_cb_d150_realizations.csv
  data/processed/percolation/fss_monodisperse_cnt_L500_w_sweep.csv
  data/processed/percolation/fss_monodisperse_cnt_L500_w_sweep_realizations.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# Put the repo root on sys.path so a bare checkout runs without pip install;
# the cntcb imports below intentionally follow this bootstrap (hence E402).
ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT
OUT_DIR = PAPER / "data" / "processed" / "percolation"

sys.path.insert(0, str(ROOT))

from cntcb.kernel.materials import carbon_black, carbon_nanotube  # noqa: E402
from cntcb.kernel.percolation_finder import _phi_c_theory, find_threshold_cb, find_threshold_cnt  # noqa: E402


TUNNEL_NM = 10.0
CB_DIAM_NM = 150.0
CNT_LEN_NM = 500.0
CNT_DIAM_NM = 10.0
SEG_LEN_UNIT_NM = 50.0
CNT_W_VALUES = (1.0, 0.7, 0.3)
L_NM_LIST = (2500, 4000, 5500, 7000, 8500, 10000)
SAFETY = 2.5
SEED_BASE = 20260525

N_REAL_SCHEDULE = {
    2500: 480,
    4000: 480,
    5500: 180,
    7000: 120,
    8500: 120,
    10000: 120,
}

N_REAL_QUICK = {
    2500: 30,
    4000: 20,
    5500: 12,
    7000: 8,
    8500: 8,
    10000: 6,
}

CB_CSV = OUT_DIR / "fss_monodisperse_cb_d150.csv"
CNT_CSV = OUT_DIR / "fss_monodisperse_cnt_L500_w_sweep.csv"
CB_REAL_CSV = OUT_DIR / "fss_monodisperse_cb_d150_realizations.csv"
CNT_REAL_CSV = OUT_DIR / "fss_monodisperse_cnt_L500_w_sweep_realizations.csv"

CB_HEADER = [
    "system",
    "filler_type",
    "L_nm",
    "L_over_d",
    "phi_c_mean",
    "phi_c_std",
    "phi_c_q25",
    "phi_c_q75",
    "phi_c_theory",
    "n_real_requested",
    "n_ok",
    "n_fail",
    "n_max",
    "elapsed_s",
    "tunnel_nm",
    "d_cb_nm",
]

CNT_HEADER = [
    "system",
    "filler_type",
    "w",
    "L_nm",
    "L_over_Lcnt",
    "phi_c_mean",
    "phi_c_std",
    "phi_c_q25",
    "phi_c_q75",
    "phi_c_theory",
    "n_real_requested",
    "n_ok",
    "n_fail",
    "n_max",
    "elapsed_s",
    "tunnel_nm",
    "cnt_length_nm",
    "cnt_diam_nm",
    "cnt_segmentation_rule",
    "seg_len_unit_nm",
    "n_segments",
    "terminal_remainder_nm",
    "mean_segment_length_nm",
    "max_segment_length_nm",
]

CB_REAL_HEADER = [
    "system",
    "filler_type",
    "L_nm",
    "L_over_d",
    "realization",
    "seed",
    "phi_c",
    "success",
    "n_max",
    "tunnel_nm",
    "d_cb_nm",
]

CNT_REAL_HEADER = [
    "system",
    "filler_type",
    "w",
    "L_nm",
    "L_over_Lcnt",
    "realization",
    "seed",
    "phi_c",
    "success",
    "n_max",
    "tunnel_nm",
    "cnt_length_nm",
    "cnt_diam_nm",
    "cnt_segmentation_rule",
    "seg_len_unit_nm",
    "n_segments",
    "terminal_remainder_nm",
    "mean_segment_length_nm",
    "max_segment_length_nm",
]


def _fixed_step_cnt_segments(length_nm: float, unit_nm: float) -> tuple[int, float]:
    n_full = int(np.floor(length_nm / unit_nm))
    remainder = float(length_nm - n_full * unit_nm)
    if remainder < 1e-9:
        remainder = 0.0
    if remainder > 0.0:
        raise ValueError(
            "This monodisperse benchmark wrapper only supports CNT lengths "
            "that are exact multiples of the fixed segment unit. Use a "
            "publication wrapper with terminal-remainder support for other lengths."
        )
    return max(1, n_full), remainder


CNT_N_SEG, CNT_REMAINDER_NM = _fixed_step_cnt_segments(CNT_LEN_NM, SEG_LEN_UNIT_NM)


def _available_memory_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        return 16.0


def _worker_count(kind: str, n_max: int, n_seg: int, requested: int) -> int:
    """Cap workers by a conservative memory estimate."""
    if requested <= 1:
        return 1

    if kind == "cnt":
        # act_starts + act_ends dominate: 2 * n_max * n_seg * 3 float64.
        # Extra factor covers union-find, hash buckets, temporary arrays, imports.
        bytes_per_worker = n_max * n_seg * 96 + n_max * 96
    else:
        # CB stores centers plus union-find/spatial-hash overhead.
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


def _load_done_cb(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done: set[int] = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            done.add(int(float(row["L_nm"])))
    return done


def _load_done_cnt(path: Path) -> set[tuple[float, int]]:
    if not path.exists():
        return set()
    done: set[tuple[float, int]] = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            done.add((float(row["w"]), int(float(row["L_nm"]))))
    return done


def _n_max_cb(L_nm: float) -> tuple[int, float]:
    filler = carbon_black(diameter_nm=CB_DIAM_NM)
    phi_theory = _phi_c_theory(filler, TUNNEL_NM)
    v_particle = (4.0 / 3.0) * np.pi * (CB_DIAM_NM / 2.0) ** 3
    n_max = int(np.ceil(phi_theory * L_nm**3 / v_particle * SAFETY))
    return max(100, n_max), float(phi_theory)


def _n_max_cnt(L_nm: float, w: float) -> tuple[int, float]:
    filler = carbon_nanotube(
        diameter_nm=CNT_DIAM_NM,
        length_um=CNT_LEN_NM / 1000.0,
        waviness=w,
        n_segments=CNT_N_SEG,
    )
    phi_theory = _phi_c_theory(filler, TUNNEL_NM)
    v_particle = np.pi * (CNT_DIAM_NM / 2.0) ** 2 * CNT_LEN_NM
    n_max = int(np.ceil(phi_theory * L_nm**3 / v_particle * SAFETY))
    return max(100, n_max), float(phi_theory)


def _worker_cb(args: tuple[float, int, int]) -> float:
    L_nm, n_max, seed = args
    filler = carbon_black(diameter_nm=CB_DIAM_NM)
    res = find_threshold_cb(
        filler,
        rve_size=L_nm,
        tunnel_cutoff_nm=TUNNEL_NM,
        n_max=n_max,
        n_realizations=1,
        seed=seed,
        verbose=False,
    )
    return float(res.phi_c_mean)


def _worker_cnt(args: tuple[float, int, float, int]) -> float:
    L_nm, n_max, w, seed = args
    filler = carbon_nanotube(
        diameter_nm=CNT_DIAM_NM,
        length_um=CNT_LEN_NM / 1000.0,
        waviness=w,
        n_segments=CNT_N_SEG,
    )
    res = find_threshold_cnt(
        filler,
        rve_size=L_nm,
        tunnel_cutoff_nm=TUNNEL_NM,
        n_max=n_max,
        n_realizations=1,
        seed=seed,
        verbose=False,
    )
    return float(res.phi_c_mean)


def _summarize(values: list[float]) -> tuple[float, float, float, float, int, int]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n_ok = int(arr.size)
    n_fail = len(values) - n_ok
    if n_ok == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"), 0, n_fail)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n_ok > 1 else 0.0
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    return mean, std, q25, q75, n_ok, n_fail


def _run_values(
    worker: Callable[[tuple], float],
    tasks: list[tuple],
    workers: int,
    progress_every: int,
) -> list[float]:
    values: list[float | None] = [None] * len(tasks)
    t0 = time.time()
    n_done = 0

    def record(idx: int, value: float) -> None:
        nonlocal n_done
        values[idx - 1] = value
        n_done += 1
        if progress_every > 0 and (n_done % progress_every == 0 or n_done == len(tasks)):
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0.0
            eta = (len(tasks) - n_done) / rate if rate > 0 else float("nan")
            n_ok = sum(1 for v in values if v is not None and np.isfinite(v))
            n_fail = sum(1 for v in values if v is not None and not np.isfinite(v))
            print(
                f"    progress {n_done}/{len(tasks)}  ok={n_ok}  fail={n_fail}  "
                f"elapsed={elapsed:.1f}s  eta={eta:.1f}s",
                flush=True,
            )

    if workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            record(idx, worker(task))
        return [v for v in values if v is not None]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(worker, task): idx for idx, task in enumerate(tasks, start=1)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            record(idx, future.result())

    return [v for v in values if v is not None]


def _append_cb_row(
    L_nm: int,
    phi_mean: float,
    phi_std: float,
    phi_q25: float,
    phi_q75: float,
    phi_theory: float,
    n_req: int,
    n_ok: int,
    n_fail: int,
    n_max: int,
    elapsed_s: float,
) -> None:
    with CB_CSV.open("a", newline="") as f:
        csv.writer(f).writerow(
            [
                "CB d=150 nm",
                "CB",
                L_nm,
                f"{L_nm / CB_DIAM_NM:.6g}",
                f"{phi_mean:.8g}",
                f"{phi_std:.8g}",
                f"{phi_q25:.8g}",
                f"{phi_q75:.8g}",
                f"{phi_theory:.8g}",
                n_req,
                n_ok,
                n_fail,
                n_max,
                f"{elapsed_s:.3f}",
                TUNNEL_NM,
                CB_DIAM_NM,
            ]
        )


def _append_cb_realizations(
    L_nm: int,
    n_max: int,
    tasks: list[tuple[float, int, int]],
    values: list[float],
) -> None:
    with CB_REAL_CSV.open("a", newline="") as f:
        writer = csv.writer(f)
        for realization, (task, phi_c) in enumerate(zip(tasks, values), start=1):
            _, _, seed = task
            success = bool(np.isfinite(phi_c))
            writer.writerow(
                [
                    "CB d=150 nm",
                    "CB",
                    L_nm,
                    f"{L_nm / CB_DIAM_NM:.6g}",
                    realization,
                    seed,
                    f"{phi_c:.8g}" if success else "",
                    success,
                    n_max,
                    TUNNEL_NM,
                    CB_DIAM_NM,
                ]
            )


def _append_cnt_row(
    w: float,
    L_nm: int,
    phi_mean: float,
    phi_std: float,
    phi_q25: float,
    phi_q75: float,
    phi_theory: float,
    n_req: int,
    n_ok: int,
    n_fail: int,
    n_max: int,
    elapsed_s: float,
) -> None:
    with CNT_CSV.open("a", newline="") as f:
        csv.writer(f).writerow(
            [
                f"CNT L=500 nm w={w:.1f}",
                "CNT",
                f"{w:.6g}",
                L_nm,
                f"{L_nm / CNT_LEN_NM:.6g}",
                f"{phi_mean:.8g}",
                f"{phi_std:.8g}",
                f"{phi_q25:.8g}",
                f"{phi_q75:.8g}",
                f"{phi_theory:.8g}",
                n_req,
                n_ok,
                n_fail,
                n_max,
                f"{elapsed_s:.3f}",
                TUNNEL_NM,
                CNT_LEN_NM,
                CNT_DIAM_NM,
                "fixed_50nm_contour_step",
                SEG_LEN_UNIT_NM,
                CNT_N_SEG,
                CNT_REMAINDER_NM,
                f"{CNT_LEN_NM / CNT_N_SEG:.8g}",
                SEG_LEN_UNIT_NM,
            ]
        )


def _append_cnt_realizations(
    w: float,
    L_nm: int,
    n_max: int,
    tasks: list[tuple[float, int, float, int]],
    values: list[float],
) -> None:
    with CNT_REAL_CSV.open("a", newline="") as f:
        writer = csv.writer(f)
        for realization, (task, phi_c) in enumerate(zip(tasks, values), start=1):
            _, _, _, seed = task
            success = bool(np.isfinite(phi_c))
            writer.writerow(
                [
                    f"CNT L=500 nm w={w:.1f}",
                    "CNT",
                    f"{w:.6g}",
                    L_nm,
                    f"{L_nm / CNT_LEN_NM:.6g}",
                    realization,
                    seed,
                    f"{phi_c:.8g}" if success else "",
                    success,
                    n_max,
                    TUNNEL_NM,
                    CNT_LEN_NM,
                    CNT_DIAM_NM,
                    "fixed_50nm_contour_step",
                    SEG_LEN_UNIT_NM,
                    CNT_N_SEG,
                    CNT_REMAINDER_NM,
                    f"{CNT_LEN_NM / CNT_N_SEG:.8g}",
                    SEG_LEN_UNIT_NM,
                ]
            )


def _run_cb_point(
    L_nm: int,
    n_real: int,
    requested_workers: int,
    progress_every: int,
) -> None:
    n_max, phi_theory = _n_max_cb(float(L_nm))
    n_workers = _worker_count("cb", n_max, 1, requested_workers)
    tasks = [(float(L_nm), n_max, SEED_BASE + L_nm * 10 + i) for i in range(n_real)]

    print(
        f"CB   L={L_nm:5d} nm  n={n_real:4d}  n_max={n_max:8d}  "
        f"workers={n_workers:2d}  phi_ref={phi_theory:.5f}",
        flush=True,
    )
    t0 = time.time()
    values = _run_values(_worker_cb, tasks, n_workers, progress_every)
    elapsed = time.time() - t0

    phi_mean, phi_std, phi_q25, phi_q75, n_ok, n_fail = _summarize(values)
    _append_cb_realizations(L_nm, n_max, tasks, values)
    _append_cb_row(
        L_nm,
        phi_mean,
        phi_std,
        phi_q25,
        phi_q75,
        phi_theory,
        n_real,
        n_ok,
        n_fail,
        n_max,
        elapsed,
    )
    print(
        f"     -> phi_c={phi_mean:.5f} +/- {phi_std:.5f}  "
        f"ok={n_ok}/{n_real}  elapsed={elapsed:.1f}s",
        flush=True,
    )


def _run_cnt_point(
    w: float,
    L_nm: int,
    n_real: int,
    requested_workers: int,
    progress_every: int,
) -> None:
    n_max, phi_theory = _n_max_cnt(float(L_nm), w)
    n_workers = _worker_count("cnt", n_max, CNT_N_SEG, requested_workers)
    seed_offset = int(round(w * 1000)) * 1_000_000 + L_nm * 10
    tasks = [
        (float(L_nm), n_max, float(w), SEED_BASE + seed_offset + i)
        for i in range(n_real)
    ]

    print(
        f"CNT  w={w:.1f}  L={L_nm:5d} nm  n={n_real:4d}  n_max={n_max:8d}  "
        f"segments={CNT_N_SEG}x50nm  workers={n_workers:2d}  "
        f"phi_ref={phi_theory:.5f}",
        flush=True,
    )
    t0 = time.time()
    values = _run_values(_worker_cnt, tasks, n_workers, progress_every)
    elapsed = time.time() - t0

    phi_mean, phi_std, phi_q25, phi_q75, n_ok, n_fail = _summarize(values)
    _append_cnt_realizations(w, L_nm, n_max, tasks, values)
    _append_cnt_row(
        w,
        L_nm,
        phi_mean,
        phi_std,
        phi_q25,
        phi_q75,
        phi_theory,
        n_real,
        n_ok,
        n_fail,
        n_max,
        elapsed,
    )
    print(
        f"     -> phi_c={phi_mean:.5f} +/- {phi_std:.5f}  "
        f"ok={n_ok}/{n_real}  elapsed={elapsed:.1f}s",
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    global CB_CSV, CNT_CSV, CB_REAL_CSV, CNT_REAL_CSV

    if args.tag:
        suffix = "_" + "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in args.tag)
    elif args.quick:
        suffix = "_quick"
    else:
        suffix = ""

    CB_CSV = OUT_DIR / f"fss_monodisperse_cb_d150{suffix}.csv"
    CNT_CSV = OUT_DIR / f"fss_monodisperse_cnt_L500_w_sweep{suffix}.csv"
    CB_REAL_CSV = OUT_DIR / f"fss_monodisperse_cb_d150{suffix}_realizations.csv"
    CNT_REAL_CSV = OUT_DIR / f"fss_monodisperse_cnt_L500_w_sweep{suffix}_realizations.csv"

    schedule = N_REAL_QUICK if args.quick else N_REAL_SCHEDULE
    requested_workers = min(args.workers, os.cpu_count() or args.workers)

    _ensure_header(CB_CSV, CB_HEADER)
    _ensure_header(CNT_CSV, CNT_HEADER)
    _ensure_header(CB_REAL_CSV, CB_REAL_HEADER)
    _ensure_header(CNT_REAL_CSV, CNT_REAL_HEADER)

    done_cb = _load_done_cb(CB_CSV)
    done_cnt = _load_done_cnt(CNT_CSV)

    print("=" * 72)
    print("Paper FSS benchmark: monodisperse CB + CNT waviness family")
    print(f"L grid: {list(L_NM_LIST)} nm")
    print(f"CB: d={CB_DIAM_NM:g} nm")
    print(
        f"CNT: L={CNT_LEN_NM:g} nm, d={CNT_DIAM_NM:g} nm, "
        f"segments={CNT_N_SEG} x {SEG_LEN_UNIT_NM:g} nm, w={CNT_W_VALUES}"
    )
    print(f"Mode: {'quick' if args.quick else 'production'}")
    if suffix:
        print(f"Output suffix: {suffix}")
    print(f"Requested workers: {requested_workers}")
    print("=" * 72)

    if args.only in ("all", "cb"):
        for L_nm in L_NM_LIST:
            if L_nm in done_cb:
                print(f"CB   L={L_nm:5d} nm  skip (already in CSV)")
                continue
            _run_cb_point(L_nm, schedule[L_nm], requested_workers, args.progress_every)

    if args.only in ("all", "cnt"):
        for w in CNT_W_VALUES:
            for L_nm in L_NM_LIST:
                key = (float(w), int(L_nm))
                if key in done_cnt:
                    print(f"CNT  w={w:.1f}  L={L_nm:5d} nm  skip (already in CSV)")
                    continue
                _run_cnt_point(
                    float(w),
                    L_nm,
                    schedule[L_nm],
                    requested_workers,
                    args.progress_every,
                )

    print("=" * 72)
    print("Done.")
    print(f"CB CSV : {CB_CSV}")
    print(f"CB realizations : {CB_REAL_CSV}")
    print(f"CNT CSV: {CNT_CSV}")
    print(f"CNT realizations: {CNT_REAL_CSV}")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the publication monodisperse percolation/FSS benchmark."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small realization schedule for smoke testing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Requested maximum worker count; script may cap this by memory.",
    )
    parser.add_argument(
        "--only",
        choices=("all", "cb", "cnt"),
        default="all",
        help="Run only CB, only CNT, or all systems.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional output suffix tag, e.g. smoke or trial1.",
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
