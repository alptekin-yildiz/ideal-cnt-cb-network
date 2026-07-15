#!/usr/bin/env python3
"""Ideal Simmons-weighted conductivity generator -- ideal-cnt-cb-network.

Enter your own filler distributions and volume-fraction targets and read back
the ideal-network conductivity as an ENSEMBLE: sigma = mean +/- std over N
stochastic realizations [S/m].
The summary also reports the median share of Joule power carried by SS
(CB-CB), SC (CB-CNT), and CC (CNT-CNT) junctions in the conducting network.

You set the CB and CNT volume fractions DIRECTLY (--phi-cb-vol / --phi-cnt-vol);
at least one must be positive. This generator does not need a precomputed
percolation table -- it is self-contained. Set --cb-sigma-d-nm 0 (or
--cnt-sigma-l-nm 0) for a monodisperse population.

Note: conductivity realizations are heavier than percolation. Keep --l-rve-nm
larger than the longest CNT (L_max); the quick demo below stays safe in its
small 1000 nm box by using a monodisperse 500 nm CNT population
(--cnt-sigma-l-nm 0).

Examples
--------
Quick CNT-only demo (seconds):
    python generate/conductivity.py --phi-cnt-vol 0.02 --cnt-sigma-l-nm 0 \\
        --l-rve-nm 1000 --n-real 2

Your polydisperse CNTs at phi = 0.012 (vol), 10 realizations:
    python generate/conductivity.py --phi-cnt-vol 0.012 \\
        --cnt-mu-l-nm 500 --cnt-sigma-l-nm 300 --l-rve-nm 4000 --n-real 10
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

# Put the repo root on sys.path so a bare checkout runs without pip install;
# the cntcb imports below intentionally follow this bootstrap (hence E402).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cntcb.engines.percolation_engine import (  # noqa: E402
    estimate_cb_n_max,
    estimate_cnt_n_max,
    solve_truncated_lognormal,
)
from cntcb.engines.simmons_conductivity_engine import (  # noqa: E402
    run_simmons_conductivity_ensemble,
)

SAFETY = 2.5


def _resolve(args) -> tuple[float, int]:
    l_rve = args.l_rve_nm if args.l_rve_nm is not None else (1000.0 if args.quick else 4000.0)
    n_real = args.n_real if args.n_real is not None else (2 if args.quick else 10)
    return l_rve, n_real


def _make_progress_callback(progress_every: int, n_total: int):
    if progress_every <= 0 or n_total < progress_every:
        return None

    t0 = time.time()
    n_ok = 0
    n_fail = 0

    def on_result(_idx, result, n_done: int, total: int) -> None:
        nonlocal n_ok, n_fail
        if result.success and math.isfinite(result.sigma_s_per_m):
            n_ok += 1
        else:
            n_fail += 1
        if n_done % progress_every == 0 or n_done == total:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0.0
            eta = (total - n_done) / rate if rate > 0 else float("nan")
            print(
                f"    progress {n_done}/{total}  ok={n_ok}  fail={n_fail}  "
                f"elapsed={elapsed:.1f}s  eta={eta:.1f}s",
                flush=True,
            )

    return on_result


def run(args: argparse.Namespace) -> None:
    if args.phi_cb_vol <= 0.0 and args.phi_cnt_vol <= 0.0:
        raise SystemExit("Set at least one of --phi-cb-vol / --phi-cnt-vol > 0.")
    if args.workers < 1:
        raise SystemExit("error: --workers must be >= 1.")
    if args.progress_every < 0:
        raise SystemExit("error: --progress-every must be >= 0.")

    l_rve, n_real = _resolve(args)
    cb_model = solve_truncated_lognormal(
        args.cb_mu_d_nm, args.cb_sigma_d_nm, args.cb_d_min_nm, args.cb_d_max_nm
    )
    cnt_model = solve_truncated_lognormal(
        args.cnt_mu_l_nm, args.cnt_sigma_l_nm, args.cnt_l_min_nm, args.cnt_l_max_nm
    )
    n_cb_max = estimate_cb_n_max(cb_model, max(args.phi_cb_vol, 1e-12), l_rve, safety=SAFETY)
    n_cnt_max = estimate_cnt_n_max(
        cnt_model, max(args.phi_cnt_vol, 1e-12), l_rve, args.cnt_diam_nm, safety=SAFETY
    )

    results = run_simmons_conductivity_ensemble(
        cb_model=cb_model,
        cnt_model=cnt_model,
        phi_cb_target_vol=args.phi_cb_vol,
        phi_cnt_target_vol=args.phi_cnt_vol,
        l_rve_nm=l_rve,
        cnt_diam_nm=args.cnt_diam_nm,
        waviness=args.cnt_waviness,
        tunnel_nm=args.tunnel_nm,
        seg_len_unit_nm=args.seg_len_nm,
        d_min_nm=args.d_min_nm,
        g_cutoff_s=args.g_cutoff_s,
        seed_base=args.seed_base,
        n_realizations=n_real,
        n_cb_max=n_cb_max,
        n_cnt_max=n_cnt_max,
        workers=args.workers,
        on_result=_make_progress_callback(args.progress_every, n_real),
    )
    # Realization order is preserved, so the collected list (and every
    # order-sensitive float reduction on it) is identical for any --workers.
    sigmas = [
        res.sigma_s_per_m
        for res in results
        if res.success and math.isfinite(res.sigma_s_per_m)
    ]
    channel_power = {
        "ss": [],
        "sc": [],
        "cc": [],
    }
    for res in results:
        if res.success and math.isfinite(res.sigma_s_per_m):
            channel_power["ss"].append(res.power_frac_ss)
            channel_power["sc"].append(res.power_frac_sc)
            channel_power["cc"].append(res.power_frac_cc)

    _report(args, l_rve, n_real, n_cb_max, n_cnt_max, sigmas, channel_power)
    if args.out:
        _write_csv(Path(args.out), args, l_rve, n_real, sigmas, channel_power)


def _stats(vals: list[float]) -> dict[str, float]:
    a = np.asarray(vals, dtype=float)
    n = len(a)
    return {
        "mean": float(np.mean(a)) if n else float("nan"),
        "std": float(np.std(a, ddof=1)) if n > 1 else 0.0,
        "median": float(np.median(a)) if n else float("nan"),
        "p25": float(np.percentile(a, 25)) if n else float("nan"),
        "p75": float(np.percentile(a, 75)) if n else float("nan"),
    }


def _median(vals: list[float]) -> float:
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else float("nan")


def _channel_medians(channel_power: dict[str, list[float]]) -> dict[str, float]:
    return {key: _median(vals) for key, vals in channel_power.items()}


def _fmt_frac(value: float) -> str:
    return f"{value:.4g}" if math.isfinite(value) else "nan"


def _format_channel_line(ch: dict[str, float]) -> str:
    return (
        f"ss={_fmt_frac(ch['ss'])} "
        f"sc={_fmt_frac(ch['sc'])} "
        f"cc={_fmt_frac(ch['cc'])} [median fraction]"
    )


def _report(args, l_rve, n_real, n_cb_max, n_cnt_max, sigmas, channel_power) -> None:
    s = _stats(sigmas)
    ch = _channel_medians(channel_power)
    bar = "=" * 70
    print(bar)
    print(" ideal-cnt-cb-network | Simmons-weighted conductivity generator")
    print(" Monte Carlo ENSEMBLE -- reported value is mean +/- std over N runs")
    print(bar)
    print(f" phi target    : CB={args.phi_cb_vol:g}  CNT={args.phi_cnt_vol:g}  [vol fraction]")
    print(f" CB dist       : mu_d={args.cb_mu_d_nm:g} sigma_d={args.cb_sigma_d_nm:g} nm  [{args.cb_d_min_nm:g},{args.cb_d_max_nm:g}]")
    print(f" CNT dist      : mu_L={args.cnt_mu_l_nm:g} sigma_L={args.cnt_sigma_l_nm:g} nm  [{args.cnt_l_min_nm:g},{args.cnt_l_max_nm:g}]  d={args.cnt_diam_nm:g} w={args.cnt_waviness:g}")
    print(f" RVE           : {l_rve:g} nm   tunnel={args.tunnel_nm:g} nm  d_min={args.d_min_nm:g} nm  G_cut={args.g_cutoff_s:g} S")
    print(f" realizations  : {len(sigmas)} ok / {n_real}   (seed base {args.seed_base}, workers {args.workers})   n_cb_max={n_cb_max} n_cnt_max={n_cnt_max}")
    print("-" * 70)
    if not sigmas:
        print(" sigma (MC)    : NaN -- no realization produced a spanning network")
        print(" hint          : raise phi above the percolation threshold, or enlarge --l-rve-nm")
    else:
        print(f" sigma (MC)    : {s['mean']:.4g} +/- {s['std']:.2g}  [S/m]")
        print(f" sigma median  : {s['median']:.4g}  [p25 {s['p25']:.4g}, p75 {s['p75']:.4g}]  [S/m]")
        print(" channel power : " + _format_channel_line(ch))
        print(" note          : sigma can span orders of magnitude near threshold; median/IQR is the robust locator")
    print(bar)


def _write_csv(path: Path, args, l_rve, n_real, sigmas, channel_power) -> None:
    s = _stats(sigmas)
    ch = _channel_medians(channel_power)
    if path.exists():
        print(f"[warn] overwriting existing {path}", file=sys.stderr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["phi_cb_vol", "phi_cnt_vol", "sigma_mean_s_per_m", "sigma_std_s_per_m",
             "sigma_median_s_per_m", "sigma_p25_s_per_m", "sigma_p75_s_per_m",
             "power_frac_ss_median", "power_frac_sc_median", "power_frac_cc_median",
             "n_ok", "n_real", "l_rve_nm", "tunnel_nm", "d_min_nm", "g_cutoff_s",
             "workers", "seed_base"]
        )
        w.writerow(
            [args.phi_cb_vol, args.phi_cnt_vol, s["mean"], s["std"], s["median"],
             s["p25"], s["p75"], ch["ss"], ch["sc"], ch["cc"], len(sigmas),
             n_real, l_rve, args.tunnel_nm, args.d_min_nm, args.g_cutoff_s,
             args.workers, args.seed_base]
        )
    print(f" wrote: {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ideal Simmons-weighted CNT/CB conductivity as a Monte Carlo ensemble (mean +/- std).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Volume-fraction targets (the primary inputs)
    p.add_argument("--phi-cb-vol", type=float, default=0.0, help="CB volume fraction target.")
    p.add_argument("--phi-cnt-vol", type=float, default=0.0, help="CNT volume fraction target.")
    # CB distribution (sigma=0 => monodisperse)
    p.add_argument("--cb-mu-d-nm", type=float, default=148.0)
    p.add_argument("--cb-sigma-d-nm", type=float, default=83.0)
    p.add_argument("--cb-d-min-nm", type=float, default=20.0)
    p.add_argument("--cb-d-max-nm", type=float, default=500.0)
    # CNT distribution (sigma=0 => monodisperse)
    p.add_argument("--cnt-mu-l-nm", type=float, default=500.0)
    p.add_argument("--cnt-sigma-l-nm", type=float, default=300.0)
    p.add_argument("--cnt-l-min-nm", type=float, default=50.0)
    p.add_argument("--cnt-l-max-nm", type=float, default=1500.0)
    p.add_argument("--cnt-diam-nm", "--cnt-diameter-nm", type=float, default=10.0)
    p.add_argument("--cnt-waviness", type=float, default=0.7)
    # Transport / RVE knobs
    p.add_argument("--l-rve-nm", "--rve-nm", type=float, default=None, help="RVE edge length [nm]; should exceed the longest CNT (recommended).")
    p.add_argument("--tunnel-nm", type=float, default=10.0)
    p.add_argument("--d-min-nm", type=float, default=0.34, help="Minimum tunneling gap [nm].")
    p.add_argument("--g-cutoff-s", type=float, default=1.0e-15, help="Edge conductance cutoff [S].")
    p.add_argument("--seg-len-nm", type=float, default=50.0, help="CNT segment unit length [nm].")
    # Stochastic
    p.add_argument("--n-real", type=int, default=None, help="Number of realizations to average.")
    p.add_argument("--workers", type=int, default=1,
                   help="Worker processes for the realization ensemble; every "
                        "realization has its own RNG seed, so any count gives "
                        "bit-identical numbers (only the wall time changes).")
    p.add_argument("--progress-every", type=int, default=10,
                   help="Print progress after this many realizations; use 0 to disable.")
    p.add_argument("--seed-base", "--seed", type=int, default=20260610)
    p.add_argument("--quick", action="store_true", help="Fast low-statistics demo (small RVE, N=2).")
    p.add_argument("--out", default=None, help="Optional CSV path for the summary row.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
