#!/usr/bin/env python3
"""Fit conductivity-defined thresholds from sigma(x) data.

The fitted model is

    sigma = sigma0 * (x - x_c)^t

where ``x_c`` is the conductivity-defined threshold and ``t`` is the
power-law exponent. For each candidate threshold, ``t`` and ``sigma0`` are
obtained by a log-log least-squares line fit; the reported threshold is the
candidate with the highest R2. This mirrors the Pan & Li case-study fit used
by the manuscript while keeping the interface general for user CSV files.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FitResult:
    group: str
    x_c: float
    t: float
    sigma0: float
    r2: float
    n_fit: int
    x_min_fit: float
    x_max_fit: float


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(line for line in f if not line.lstrip().startswith("#")))


def _finite_float(value: str, *, col: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column {col!r} contains a non-numeric value: {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"Column {col!r} contains a non-finite value: {value!r}")
    return out


def _optional_finite_float(value: str) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _include_flag(value: str, *, col: str, row_idx: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "include"}:
        return True
    if normalized in {"", "0", "false", "no", "n", "exclude"}:
        return False
    raise ValueError(
        f"Row {row_idx}: {col!r} must be a boolean include flag "
        f"(got {value!r})."
    )


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return float(slope), float(intercept), r2


def fit_powerlaw(
    x_values: Iterable[float],
    sigma_values: Iterable[float],
    *,
    group: str = "all",
    threshold_min: float = 0.0,
    threshold_step: float = 0.005,
    threshold_margin: float = 0.001,
) -> FitResult:
    x = np.asarray(list(x_values), dtype=float)
    sigma = np.asarray(list(sigma_values), dtype=float)
    if len(x) != len(sigma):
        raise ValueError("x and sigma arrays must have the same length.")
    ok = np.isfinite(x) & np.isfinite(sigma) & (sigma > 0.0)
    x = x[ok]
    sigma = sigma[ok]
    if len(x) < 3:
        raise ValueError(f"{group}: at least three positive sigma points are required.")
    if threshold_step <= 0.0:
        raise ValueError("--threshold-step must be > 0.")
    if threshold_margin < 0.0:
        raise ValueError("--threshold-margin must be >= 0.")

    upper = float(np.min(x)) - threshold_margin
    if upper <= threshold_min:
        raise ValueError(
            f"{group}: no threshold candidates below the first fit point "
            f"({np.min(x):.8g})."
        )

    best: tuple[float, float, float, float] | None = None
    for x_c in np.arange(threshold_min, upper, threshold_step):
        delta = x - x_c
        if np.any(delta <= 0.0):
            continue
        log_delta = np.log(delta)
        log_sigma = np.log(sigma)
        t, log_sigma0, r2 = _linear_fit(log_delta, log_sigma)
        pred = t * log_delta + log_sigma0
        ss_res = float(np.sum((log_sigma - pred) ** 2))
        if best is None or r2 > best[0] or (r2 == best[0] and ss_res < best[1]):
            best = (r2, ss_res, float(x_c), t)

    if best is None:
        raise ValueError(f"{group}: could not fit a conductivity threshold.")

    _r2, _ss_res, x_c, _t = best
    log_delta = np.log(x - x_c)
    t, log_sigma0, r2 = _linear_fit(log_delta, np.log(sigma))
    return FitResult(
        group=group,
        x_c=float(x_c),
        t=t,
        sigma0=float(np.exp(log_sigma0)),
        r2=r2,
        n_fit=len(x),
        x_min_fit=float(np.min(x)),
        x_max_fit=float(np.max(x)),
    )


def _grouped_fit_rows(
    rows: list[dict[str, str]],
    *,
    group_col: str | None,
    x_col: str,
    sigma_col: str,
    n_ok_col: str | None,
    n_total_col: str | None,
    min_success_fraction: float | None,
    fit_include_col: str | None,
    threshold_min: float,
    threshold_step: float,
    threshold_margin: float,
) -> list[FitResult]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row_idx, row in enumerate(rows, start=2):
        if fit_include_col is not None:
            if fit_include_col not in row:
                raise ValueError(f"Missing fit-include column: {fit_include_col!r}.")
            if not _include_flag(row[fit_include_col], col=fit_include_col, row_idx=row_idx):
                continue
        group = row[group_col] if group_col else "all"
        x = _optional_finite_float(row.get(x_col, ""))
        sigma = _optional_finite_float(row.get(sigma_col, ""))
        if x is None or sigma is None or sigma <= 0.0:
            continue
        if min_success_fraction is not None:
            if not n_ok_col or not n_total_col:
                raise ValueError(
                    "--min-success-fraction requires --n-ok-col and --n-total-col."
                )
            n_ok = _finite_float(row.get(n_ok_col, ""), col=n_ok_col)
            n_total = _finite_float(row.get(n_total_col, ""), col=n_total_col)
            if n_total <= 0.0:
                raise ValueError(f"Row {row_idx}: {n_total_col!r} must be > 0.")
            if n_ok / n_total < min_success_fraction:
                continue
        grouped.setdefault(group, []).append((x, sigma))

    if not grouped:
        raise ValueError("No usable positive-conductivity rows were found.")

    results: list[FitResult] = []
    for group in sorted(grouped):
        pairs = grouped[group]
        results.append(
            fit_powerlaw(
                [x for x, _sigma in pairs],
                [sigma for _x, sigma in pairs],
                group=group,
                threshold_min=threshold_min,
                threshold_step=threshold_step,
                threshold_margin=threshold_margin,
            )
        )
    return results


def _fmt(value: float) -> str:
    return f"{value:.12g}"


def _write_csv(
    path: Path,
    results: list[FitResult],
    *,
    source: str,
    x_col: str,
    sigma_col: str,
    min_success_fraction: float | None,
    fit_include_col: str | None,
    threshold_step: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "source",
        "x_c",
        "t",
        "sigma0",
        "r2",
        "n_fit",
        "x_min_fit",
        "x_max_fit",
        "x_col",
        "sigma_col",
        "min_success_fraction",
        "fit_include_col",
        "threshold_step",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(
                {
                    "group": res.group,
                    "source": source,
                    "x_c": _fmt(res.x_c),
                    "t": _fmt(res.t),
                    "sigma0": _fmt(res.sigma0),
                    "r2": _fmt(res.r2),
                    "n_fit": str(res.n_fit),
                    "x_min_fit": _fmt(res.x_min_fit),
                    "x_max_fit": _fmt(res.x_max_fit),
                    "x_col": x_col,
                    "sigma_col": sigma_col,
                    "min_success_fraction": (
                        "" if min_success_fraction is None else _fmt(min_success_fraction)
                    ),
                    "fit_include_col": "" if fit_include_col is None else fit_include_col,
                    "threshold_step": _fmt(threshold_step),
                }
            )


def run(args: argparse.Namespace) -> list[FitResult]:
    if args.min_success_fraction is not None and not (0.0 <= args.min_success_fraction <= 1.0):
        raise SystemExit("error: --min-success-fraction must be between 0 and 1.")
    rows = _read_rows(Path(args.input_csv))
    results = _grouped_fit_rows(
        rows,
        group_col=args.group_col,
        x_col=args.x_col,
        sigma_col=args.sigma_col,
        n_ok_col=args.n_ok_col,
        n_total_col=args.n_total_col,
        min_success_fraction=args.min_success_fraction,
        fit_include_col=args.fit_include_col,
        threshold_min=args.threshold_min,
        threshold_step=args.threshold_step,
        threshold_margin=args.threshold_margin,
    )

    print("group  n_fit  x_c  t  sigma0  R2")
    for res in results:
        print(
            f"{res.group}  {res.n_fit}  {_fmt(res.x_c)}  {_fmt(res.t)}  "
            f"{_fmt(res.sigma0)}  {_fmt(res.r2)}"
        )
    if args.out:
        out_path = Path(args.out)
        _write_csv(
            out_path,
            results,
            source=args.source,
            x_col=args.x_col,
            sigma_col=args.sigma_col,
            min_success_fraction=args.min_success_fraction,
            fit_include_col=args.fit_include_col,
            threshold_step=args.threshold_step,
        )
        print(f"wrote: {out_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit sigma = sigma0 * (x - x_c)^t from a conductivity CSV. "
            "Use --group-col for one fit per material/system."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_csv", help="CSV containing loading and conductivity columns.")
    parser.add_argument("--x-col", default="wt_pct", help="Loading column used as x.")
    parser.add_argument(
        "--sigma-col",
        default="sigma_s_per_m_median",
        help="Positive conductivity column used for the fit.",
    )
    parser.add_argument("--group-col", default=None, help="Optional column for grouped fits.")
    parser.add_argument("--n-ok-col", default="n_ok", help="Successful-realization count column.")
    parser.add_argument("--n-total-col", default="n_total", help="Total-realization count column.")
    parser.add_argument(
        "--min-success-fraction",
        type=float,
        default=None,
        help="Optional n_ok/n_total cutoff before fitting.",
    )
    parser.add_argument(
        "--fit-include-col",
        default=None,
        help=(
            "Optional boolean column selecting the rows used in the fit. "
            "Accepted true values are 1/true/yes/y/include; false values are "
            "0/false/no/n/exclude/blank."
        ),
    )
    parser.add_argument("--threshold-min", type=float, default=0.0)
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.005,
        help="Grid step for the threshold scan, in the same units as x.",
    )
    parser.add_argument(
        "--threshold-margin",
        type=float,
        default=0.001,
        help="Largest candidate is below min(x) by this margin.",
    )
    parser.add_argument("--source", default="model", help="Source label written to --out CSV.")
    parser.add_argument("--out", default=None, help="Optional output CSV path.")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
