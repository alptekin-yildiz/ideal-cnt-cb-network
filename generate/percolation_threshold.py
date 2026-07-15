#!/usr/bin/env python3
"""Ideal percolation-threshold generator (CNT, CB, or CNT in a CB background).

Enter your own filler geometry and read back the Monte Carlo percolation
threshold as an ENSEMBLE over N stochastic realizations. That is the
stochastic contract of the whole toolkit -- a single run is noisy, so the
reported number is a statistic over many, which is why --n-real matters.

Two run paths, chosen by the flags you pass:

* MONODISPERSE path (the default): one length (--cnt-length-um) or one
  diameter (--cb-diameter-nm) per run -- the original exact behavior of this
  tool, on a single sequential RNG stream (--workers does not apply).
* DISTRIBUTION path: any truncated-lognormal flag (--cnt-mu-l-nm,
  --cnt-sigma-l-nm, --cb-mu-d-nm, --cb-sigma-d-nm, their min/max
  companions) or --phi-cb-vol switches to polydisperse sampling with one
  RNG seed per realization, parallelizable with --workers. Setting a sigma
  flag to 0 there gives the monodisperse limit of the same path.

Examples
--------
Quick monodisperse demo (seconds):
    python generate/percolation_threshold.py --filler cnt --quick

Polydisperse CNT threshold (lengths 500 +/- 300 nm, truncated to [50, 1500]):
    python generate/percolation_threshold.py --filler cnt \\
        --cnt-mu-l-nm 500 --cnt-sigma-l-nm 300 --quick

Polydisperse CB threshold (aggregate diameters 148 +/- 83 nm):
    python generate/percolation_threshold.py --filler cb \\
        --cb-mu-d-nm 148 --cb-sigma-d-nm 83 --quick

Hybrid: CNT threshold inside a fixed CB background (2 vol.% CB), 4 workers:
    python generate/percolation_threshold.py --filler cnt \\
        --phi-cb-vol 0.02 --quick --workers 4
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

# Put the repo root on sys.path so a bare checkout runs without pip install;
# the cntcb imports below intentionally follow this bootstrap (hence E402).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cntcb.engines.percolation_engine import (  # noqa: E402
    run_cb_threshold_ensemble,
    run_cnt_threshold_ensemble,
    solve_truncated_lognormal,
)
from cntcb.kernel.materials import carbon_black, carbon_nanotube  # noqa: E402
from cntcb.kernel.percolation_finder import find_threshold_cb, find_threshold_cnt  # noqa: E402


# Presets used when a knob is left unset. --quick trades statistics for speed;
# any explicit flag (e.g. --n-real 50) always overrides the preset.
FULL = {
    "cnt": {"rve_nm": 3000.0, "n_max": 900, "n_real": 20},
    "cb": {"rve_nm": 1500.0, "n_max": 6000, "n_real": 30},
}
QUICK = {
    "cnt": {"rve_nm": 1200.0, "n_max": 450, "n_real": 3},
    "cb": {"rve_nm": 1000.0, "n_max": 1500, "n_real": 3},
}

# Distribution-path defaults, applied per flag left unset: the paper's
# truncated-lognormal conventions (CNT length 500 +/- 300 nm in [50, 1500];
# CB effective aggregate diameter 148 +/- 83 nm in [20, 500]).
DIST_CNT_DEFAULTS = {
    "cnt_mu_l_nm": 500.0,
    "cnt_sigma_l_nm": 300.0,
    "cnt_l_min_nm": 50.0,
    "cnt_l_max_nm": 1500.0,
}
DIST_CB_DEFAULTS = {
    "cb_mu_d_nm": 148.0,
    "cb_sigma_d_nm": 83.0,
    "cb_d_min_nm": 20.0,
    "cb_d_max_nm": 500.0,
}
# Volume-fraction stop cap per swept filler on the distribution path;
# realizations that reach it without percolating are censored.
DIST_PHI_MAX_DEFAULT = {"cnt": 0.05, "cb": 0.45}

_CNT_DIST_ARGS = tuple(DIST_CNT_DEFAULTS)
_CB_DIST_ARGS = tuple(DIST_CB_DEFAULTS)


def _resolve(explicit, filler_type, key, quick):
    if explicit is not None:
        return explicit
    return (QUICK if quick else FULL)[filler_type][key]


def _distribution_requested(args: argparse.Namespace) -> bool:
    flags = _CNT_DIST_ARGS + _CB_DIST_ARGS
    return any(getattr(args, name) is not None for name in flags) or args.phi_cb_vol > 0.0


def _validate_numeric(args: argparse.Namespace) -> None:
    # Runs before path dispatch: a negative --phi-cb-vol would otherwise fail
    # the > 0 test in _distribution_requested and be silently ignored on the
    # monodisperse path, and a non-positive --phi-max-vol would produce an
    # all-censored NaN ensemble instead of an error.
    if args.phi_cb_vol < 0.0:
        sys.exit(f"error: --phi-cb-vol must be >= 0 (got {args.phi_cb_vol:g}).")
    if args.phi_cb_vol >= 1.0:
        sys.exit(
            f"error: --phi-cb-vol must be < 1 (a volume fraction; got {args.phi_cb_vol:g})."
        )
    if args.phi_max_vol is not None and not 0.0 < args.phi_max_vol < 1.0:
        sys.exit(
            "error: --phi-max-vol must be > 0 and < 1 "
            f"(a volume fraction; got {args.phi_max_vol:g})."
        )
    if args.n_real is not None and args.n_real < 1:
        sys.exit(f"error: --n-real must be >= 1 (got {args.n_real}).")
    if args.n_max is not None and args.n_max < 1:
        sys.exit(f"error: --n-max must be >= 1 (got {args.n_max}).")
    if args.rve_nm is not None and args.rve_nm <= 0.0:
        sys.exit(f"error: --rve-nm must be > 0 (got {args.rve_nm:g}).")
    if args.tunnel_nm < 0.0:
        sys.exit(f"error: --tunnel-nm must be >= 0 (got {args.tunnel_nm:g}).")
    if args.seg_len_nm <= 0.0:
        sys.exit(f"error: --seg-len-nm must be > 0 (got {args.seg_len_nm:g}).")
    if args.progress_every < 0:
        sys.exit(f"error: --progress-every must be >= 0 (got {args.progress_every}).")


def _make_progress_callback(progress_every: int, n_total: int, is_ok):
    if progress_every <= 0 or n_total < progress_every:
        return None

    t0 = time.time()
    n_ok = 0
    n_fail = 0

    def on_result(_idx, result, n_done: int, total: int) -> None:
        nonlocal n_ok, n_fail
        if is_ok(result):
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
    _validate_numeric(args)
    if _distribution_requested(args):
        _validate_distribution(args)
        run_distribution(args)
    else:
        if args.workers > 1:
            print(
                "warning: --workers is ignored on the monodisperse path (it keeps "
                "the original single sequential RNG stream); pass a distribution "
                "flag such as --cnt-sigma-l-nm for the parallel path.",
                file=sys.stderr,
            )
        run_mono(args)


# --------------------------------------------------------------------------
# Monodisperse path: the original exact behavior of this tool.
# --------------------------------------------------------------------------

def run_mono(args: argparse.Namespace) -> None:
    ft = args.filler
    rve_nm = _resolve(args.rve_nm, ft, "rve_nm", args.quick)
    n_max = _resolve(args.n_max, ft, "n_max", args.quick)
    n_real = _resolve(args.n_real, ft, "n_real", args.quick)

    if ft == "cnt":
        filler = carbon_nanotube(
            diameter_nm=args.cnt_diameter_nm,
            length_um=args.cnt_length_um,
            waviness=args.cnt_waviness,
        )
        geom = (
            f"CNT (d={args.cnt_diameter_nm:g} nm, "
            f"L={args.cnt_length_um:g} um, waviness={args.cnt_waviness:g})"
        )
        res = find_threshold_cnt(
            filler,
            rve_size=rve_nm,
            tunnel_cutoff_nm=args.tunnel_nm,
            n_max=n_max,
            n_realizations=n_real,
            seed=args.seed,
            verbose=False,
            progress_every=args.progress_every,
        )
    else:
        filler = carbon_black(diameter_nm=args.cb_diameter_nm)
        geom = f"CB (d={args.cb_diameter_nm:g} nm, spherical)"
        res = find_threshold_cb(
            filler,
            rve_size=rve_nm,
            tunnel_cutoff_nm=args.tunnel_nm,
            n_max=n_max,
            n_realizations=n_real,
            seed=args.seed,
            verbose=False,
            progress_every=args.progress_every,
        )

    _report(geom, rve_nm, n_max, args, res)
    if args.out:
        _write_csv(Path(args.out), ft, geom, rve_nm, n_max, args, res)


def _report(geom, rve_nm, n_max, args, res) -> None:
    bar = "=" * 70
    print(bar)
    print(" ideal-cnt-cb-network | percolation threshold generator")
    print(" Monte Carlo ENSEMBLE -- reported value is mean +/- std over N runs")
    print(bar)
    print(f" filler        : {geom}")
    print(f" RVE           : {rve_nm:g} nm     tunnel cutoff : {args.tunnel_nm:g} nm")
    print(f" realizations  : {res.n_realizations} ok (seed base {args.seed})   n_max={n_max}")
    print("-" * 70)
    if res.n_realizations == 0:
        print(" phi_c (MC)    : NaN -- every realization failed to percolate")
        print(f" phi_c (theory): {res.phi_c_analytical:.6g}  [vol fraction]")
        print(" hint          : raise --n-max (too few fillers to span the RVE)")
    else:
        dev = (res.phi_c_mean - res.phi_c_analytical) / res.phi_c_analytical * 100.0
        print(f" phi_c (MC)    : {res.phi_c_mean:.6g} +/- {res.phi_c_std:.2g}  [vol fraction]")
        print(f" phi_c (theory): {res.phi_c_analytical:.6g}")
        print(f" deviation     : {dev:+.1f} %  (finite-size / finite-N; shrinks with larger RVE and N)")
        print(f" failed runs   : {res.n_failed} / {res.n_realizations + res.n_failed}")
    print(bar)


def _write_csv(path: Path, filler_type, geom, rve_nm, n_max, args, res) -> None:
    if path.exists():
        print(f"[warn] overwriting existing {path}", file=sys.stderr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "filler_type",
                "geometry",
                "phi_c_mean_vol",
                "phi_c_std_vol",
                "phi_c_analytical_vol",
                "n_realizations",
                "n_failed",
                "rve_nm",
                "tunnel_cutoff_nm",
                "n_max",
                "seed",
            ]
        )
        w.writerow(
            [
                filler_type,
                geom,
                res.phi_c_mean,
                res.phi_c_std,
                res.phi_c_analytical,
                res.n_realizations,
                res.n_failed,
                rve_nm,
                args.tunnel_nm,
                n_max,
                args.seed,
            ]
        )
    print(f" wrote: {path}")


# --------------------------------------------------------------------------
# Distribution path: truncated-lognormal sizes, per-realization seeds.
# --------------------------------------------------------------------------

def _validate_distribution(args: argparse.Namespace) -> None:
    if args.workers < 1:
        sys.exit("error: --workers must be >= 1.")
    if args.filler == "cb":
        if args.phi_cb_vol > 0.0:
            sys.exit(
                "error: --phi-cb-vol sets the fixed CB *background* of the hybrid "
                "question, but --filler cb thresholds the CB network itself. For "
                "the CNT-in-CB hybrid threshold use --filler cnt --phi-cb-vol X."
            )
        if any(getattr(args, name) is not None for name in _CNT_DIST_ARGS):
            sys.exit(
                "error: --filler cb thresholds a CB-only network, so the CNT "
                "distribution flags do not apply; a CB threshold inside a CNT "
                "background is not implemented. Use --filler cnt for the CNT "
                "and hybrid questions."
            )
    else:
        if (
            any(getattr(args, name) is not None for name in _CB_DIST_ARGS)
            and args.phi_cb_vol <= 0.0
        ):
            sys.exit(
                "error: with --filler cnt the CB distribution flags describe the "
                "fixed CB background of the hybrid question; add --phi-cb-vol > 0 "
                "(or drop the CB flags for a pure CNT threshold)."
            )


def _dist_value(args: argparse.Namespace, name: str) -> float:
    explicit = getattr(args, name)
    if explicit is not None:
        return explicit
    return {**DIST_CNT_DEFAULTS, **DIST_CB_DEFAULTS}[name]


def _solve_model(mu: float, sigma: float, lower: float, upper: float):
    try:
        return solve_truncated_lognormal(mu, sigma, lower, upper)
    except (ValueError, RuntimeError) as exc:
        sys.exit(f"error: invalid truncated-lognormal input: {exc}")


def run_distribution(args: argparse.Namespace) -> None:
    ft = args.filler
    rve_nm = _resolve(args.rve_nm, ft, "rve_nm", args.quick)
    n_real = _resolve(args.n_real, ft, "n_real", args.quick)
    phi_max = (
        args.phi_max_vol if args.phi_max_vol is not None else DIST_PHI_MAX_DEFAULT[ft]
    )

    cnt_mu = _dist_value(args, "cnt_mu_l_nm")
    cnt_sigma = _dist_value(args, "cnt_sigma_l_nm")
    cnt_lmin = _dist_value(args, "cnt_l_min_nm")
    cnt_lmax = _dist_value(args, "cnt_l_max_nm")
    cb_mu = _dist_value(args, "cb_mu_d_nm")
    cb_sigma = _dist_value(args, "cb_sigma_d_nm")
    cb_dmin = _dist_value(args, "cb_d_min_nm")
    cb_dmax = _dist_value(args, "cb_d_max_nm")

    cb_model = _solve_model(cb_mu, cb_sigma, cb_dmin, cb_dmax)
    progress = _make_progress_callback(
        args.progress_every,
        n_real,
        lambda result: bool(result.percolated),
    )

    if ft == "cnt":
        cnt_model = _solve_model(cnt_mu, cnt_sigma, cnt_lmin, cnt_lmax)
        cnt_tag = "poly" if cnt_sigma > 0 else "mono (sigma=0)"
        geom = (
            f"CNT {cnt_tag}: L = {cnt_mu:g} +/- {cnt_sigma:g} nm in "
            f"[{cnt_lmin:g}, {cnt_lmax:g}], d={args.cnt_diameter_nm:g} nm, "
            f"waviness={args.cnt_waviness:g}"
        )
        if args.phi_cb_vol > 0.0:
            cb_tag = "poly" if cb_sigma > 0 else "mono (sigma=0)"
            background = (
                f"CB {cb_tag}: d = {cb_mu:g} +/- {cb_sigma:g} nm in "
                f"[{cb_dmin:g}, {cb_dmax:g}] at phi_CB = {args.phi_cb_vol:g}"
            )
        else:
            background = None
        results = run_cnt_threshold_ensemble(
            cnt_model=cnt_model,
            cb_model=cb_model,
            phi_cb_vol=args.phi_cb_vol,
            phi_cnt_max_vol=phi_max,
            l_rve_nm=rve_nm,
            cnt_diam_nm=args.cnt_diameter_nm,
            waviness=args.cnt_waviness,
            tunnel_nm=args.tunnel_nm,
            seg_len_unit_nm=args.seg_len_nm,
            seed_base=args.seed,
            n_realizations=n_real,
            n_cnt_max=args.n_max,
            workers=args.workers,
            on_result=progress,
        )
        phi_vals = [r.phi_cnt_stop_vol for r in results if r.percolated]
        n_cb_percolated = sum(r.failure_reason == "cb_percolated" for r in results)
    else:
        cb_tag = "poly" if cb_sigma > 0 else "mono (sigma=0)"
        geom = (
            f"CB {cb_tag}: d = {cb_mu:g} +/- {cb_sigma:g} nm in "
            f"[{cb_dmin:g}, {cb_dmax:g}]"
        )
        background = None
        results = run_cb_threshold_ensemble(
            cb_model=cb_model,
            phi_cb_max_vol=phi_max,
            l_rve_nm=rve_nm,
            tunnel_nm=args.tunnel_nm,
            seed_base=args.seed,
            n_realizations=n_real,
            n_cb_max=args.n_max,
            workers=args.workers,
            on_result=progress,
        )
        phi_vals = [r.phi_cb_stop_vol for r in results if r.percolated]
        n_cb_percolated = 0

    stats = _ensemble_stats(phi_vals)
    _report_distribution(
        geom, background, rve_nm, phi_max, args, len(results), stats, n_cb_percolated
    )
    if args.out:
        _write_csv_distribution(
            Path(args.out),
            args,
            ft,
            rve_nm,
            phi_max,
            len(results),
            stats,
            n_cb_percolated,
            (cnt_mu, cnt_sigma, cnt_lmin, cnt_lmax) if ft == "cnt" else None,
            (cb_mu, cb_sigma, cb_dmin, cb_dmax),
        )


def _ensemble_stats(phi_vals: list[float]) -> dict:
    n_ok = len(phi_vals)
    if n_ok == 0:
        keys = ("median", "q25", "q75", "mean", "std")
        return {"n_ok": 0, **{k: float("nan") for k in keys}}
    arr = np.asarray(phi_vals, dtype=float)
    return {
        "n_ok": n_ok,
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if n_ok > 1 else 0.0,
    }


def _report_distribution(
    geom, background, rve_nm, phi_max, args, n_total, stats, n_cb_percolated
) -> None:
    bar = "=" * 70
    n_censored = n_total - stats["n_ok"]
    print(bar)
    print(" ideal-cnt-cb-network | percolation threshold generator")
    print(" Monte Carlo ENSEMBLE -- statistics over N independently seeded runs")
    print(bar)
    print(" path          : DISTRIBUTION (truncated lognormal, one seed per realization)")
    print(f" filler        : {geom}")
    if background:
        print(f" background    : {background}")
    print(f" RVE           : {rve_nm:g} nm     tunnel cutoff : {args.tunnel_nm:g} nm")
    print(
        f" realizations  : {stats['n_ok']} percolated / {n_total}   "
        f"(seed base {args.seed}, workers {args.workers})"
    )
    n_max_note = args.n_max if args.n_max is not None else "auto"
    print(f" stop caps     : phi_max={phi_max:g}   n_max={n_max_note}")
    print("-" * 70)
    if stats["n_ok"] == 0:
        print(" phi_c (MC)    : NaN -- every realization hit its cap without percolating")
        print(" hint          : raise --phi-max-vol / --n-max, grow --rve-nm, or")
        print("                 check that the geometry can span the box at all")
    else:
        print(
            f" phi_c (MC)    : median {stats['median']:.6g}   "
            f"IQR [{stats['q25']:.6g}, {stats['q75']:.6g}]  [vol fraction]"
        )
        print(f"                 mean {stats['mean']:.6g} +/- {stats['std']:.2g}")
        if n_cb_percolated:
            print(
                f" note          : {n_cb_percolated} realization(s) percolated through "
                "the CB background alone (phi_c,CNT = 0 there)"
            )
        print(f" censored      : {n_censored} / {n_total} hit the cap without percolating")
    print(bar)


def _write_csv_distribution(
    path: Path,
    args,
    filler_type,
    rve_nm,
    phi_max,
    n_total,
    stats,
    n_cb_percolated,
    cnt_dist,
    cb_dist,
) -> None:
    cb_mu, cb_sigma, cb_dmin, cb_dmax = cb_dist
    if cnt_dist is not None:
        cnt_mu, cnt_sigma, cnt_lmin, cnt_lmax = cnt_dist
        cnt_cols = [cnt_mu, cnt_sigma, cnt_lmin, cnt_lmax,
                    args.cnt_diameter_nm, args.cnt_waviness]
    else:
        cnt_cols = ["", "", "", "", "", ""]
    cb_used = filler_type == "cb" or args.phi_cb_vol > 0.0
    cb_cols = [cb_mu, cb_sigma, cb_dmin, cb_dmax] if cb_used else ["", "", "", ""]

    if path.exists():
        print(f"[warn] overwriting existing {path}", file=sys.stderr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "path",
                "filler_type",
                "phi_cb_vol",
                "phi_c_median_vol",
                "phi_c_q25_vol",
                "phi_c_q75_vol",
                "phi_c_mean_vol",
                "phi_c_std_vol",
                "n_ok",
                "n_censored",
                "n_cb_percolated",
                "n_total",
                "cnt_mu_l_nm",
                "cnt_sigma_l_nm",
                "cnt_l_min_nm",
                "cnt_l_max_nm",
                "cnt_diam_nm",
                "cnt_waviness",
                "cb_mu_d_nm",
                "cb_sigma_d_nm",
                "cb_d_min_nm",
                "cb_d_max_nm",
                "rve_nm",
                "tunnel_cutoff_nm",
                "seg_len_nm",
                "phi_max_vol",
                "n_real",
                "workers",
                "seed_base",
            ]
        )
        w.writerow(
            [
                "distribution",
                filler_type,
                args.phi_cb_vol if filler_type == "cnt" else "",
                stats["median"],
                stats["q25"],
                stats["q75"],
                stats["mean"],
                stats["std"],
                stats["n_ok"],
                n_total - stats["n_ok"],
                n_cb_percolated,
                n_total,
                *cnt_cols,
                *cb_cols,
                rve_nm,
                args.tunnel_nm,
                args.seg_len_nm if filler_type == "cnt" else "",
                phi_max,
                n_total,
                args.workers,
                args.seed,
            ]
        )
    print(f" wrote: {path}")


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Ideal CNT/CB percolation threshold as a Monte Carlo ensemble.\n"
            "\n"
            "Mono- or polydisperse? The flags choose the path:\n"
            "  MONODISPERSE (default)  one length/diameter per run via --cnt-length-um\n"
            "                          or --cb-diameter-nm; original exact behavior,\n"
            "                          single RNG stream, --workers not applicable.\n"
            "  POLYDISPERSE            any distribution flag below (--cnt-mu-l-nm,\n"
            "                          --cnt-sigma-l-nm, --cb-mu-d-nm, ...) or\n"
            "                          --phi-cb-vol switches to truncated-lognormal\n"
            "                          sampling, one RNG seed per realization,\n"
            "                          parallel with --workers. Sigma 0 there is the\n"
            "                          monodisperse limit of the same path.\n"
            "  HYBRID                  --filler cnt --phi-cb-vol X thresholds the CNT\n"
            "                          network inside a fixed CB background at phi_CB=X."
        ),
        formatter_class=_HelpFormatter,
    )
    g_run = p.add_argument_group("run selection")
    g_run.add_argument("--filler", choices=("cnt", "cb"), required=True,
                       help="Which filler network to threshold.")
    g_run.add_argument("--quick", action="store_true", help="Fast low-statistics demo run.")
    g_run.add_argument("--workers", type=int, default=1,
                       help="Worker processes on the polydisperse/hybrid path; any "
                            "count gives bit-identical numbers (per-realization "
                            "seeds). Ignored with a warning on the monodisperse path.")

    g_mono = p.add_argument_group(
        "monodisperse geometry (default path: one size per run)"
    )
    g_mono.add_argument("--cnt-length-um", type=float, default=0.5,
                        help="Single CNT length [um].")
    g_mono.add_argument("--cb-diameter-nm", type=float, default=148.0,
                        help="Single effective CB particle diameter [nm].")

    g_both = p.add_argument_group("geometry used on both paths")
    g_both.add_argument("--cnt-diameter-nm", "--cnt-diam-nm", type=float, default=10.0,
                        help="CNT diameter [nm].")
    g_both.add_argument("--cnt-waviness", type=float, default=0.7,
                        help="CNT waviness (cosine of the bend angle).")

    g_dist = p.add_argument_group(
        "polydisperse / hybrid geometry (passing any flag here selects the "
        "distribution path; unset flags fall back to the paper defaults "
        "500/300 [50,1500] nm CNT, 148/83 [20,500] nm CB)"
    )
    g_dist.add_argument("--cnt-mu-l-nm", type=float, default=None,
                        help="Target mean CNT length [nm] of the truncated "
                             "sampled population.")
    g_dist.add_argument("--cnt-sigma-l-nm", type=float, default=None,
                        help="Target std of CNT length [nm]; 0 = monodisperse limit.")
    g_dist.add_argument("--cnt-l-min-nm", type=float, default=None,
                        help="Lower CNT length truncation bound [nm].")
    g_dist.add_argument("--cnt-l-max-nm", type=float, default=None,
                        help="Upper CNT length truncation bound [nm].")
    g_dist.add_argument("--cb-mu-d-nm", type=float, default=None,
                        help="Target mean effective CB diameter [nm] of the "
                             "truncated sampled population.")
    g_dist.add_argument("--cb-sigma-d-nm", type=float, default=None,
                        help="Target std of effective CB diameter [nm]; "
                             "0 = monodisperse limit.")
    g_dist.add_argument("--cb-d-min-nm", type=float, default=None,
                        help="Lower CB diameter truncation bound [nm].")
    g_dist.add_argument("--cb-d-max-nm", type=float, default=None,
                        help="Upper CB diameter truncation bound [nm].")
    g_dist.add_argument("--phi-cb-vol", type=float, default=0.0,
                        help="Fixed CB background volume fraction for the hybrid "
                             "question (--filler cnt only): CNT threshold in a CB "
                             "background.")
    g_dist.add_argument("--phi-max-vol", type=float, default=None,
                        help="Volume-fraction stop cap for the swept filler "
                             "(default: 0.05 for cnt, 0.45 for cb); realizations "
                             "reaching it without percolating are censored.")
    g_dist.add_argument("--seg-len-nm", type=float, default=50.0,
                        help="Fixed CNT contour step [nm] on the distribution path.")

    g_ens = p.add_argument_group("ensemble, RVE, and output (both paths)")
    g_ens.add_argument("--rve-nm", "--l-rve-nm", type=float, default=None,
                       help="RVE edge length [nm].")
    g_ens.add_argument("--n-max", type=int, default=None,
                       help="Max fillers per realization (monodisperse path: preset; "
                            "distribution path: estimated from the stop cap unless set).")
    g_ens.add_argument("--n-real", type=int, default=None,
                       help="Number of stochastic realizations to average.")
    g_ens.add_argument("--tunnel-nm", type=float, default=10.0,
                       help="Tunneling connection cutoff [nm].")
    g_ens.add_argument("--seed", "--seed-base", type=int, default=42,
                       help="Base RNG seed (reproducible).")
    g_ens.add_argument("--progress-every", type=int, default=10,
                       help="Print progress after this many realizations; "
                            "use 0 to disable.")
    g_ens.add_argument("--out", default=None, help="Optional CSV path for the summary row.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
