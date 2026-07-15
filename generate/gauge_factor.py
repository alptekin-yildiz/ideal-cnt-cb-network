#!/usr/bin/env python3
"""Ideal piezoresistive gauge-factor generator -- ideal-cnt-cb-network.

Enter your own filler distributions and volume-fraction targets and read back
the ideal-network gauge factor as an ENSEMBLE: GF = mean +/- std over N
stochastic realizations. GF is the resistance-based response gf_r =
d ln(R) / d eps fitted over the strain grid; the sigma- and geometry-only
components (gf_sigma, gf_geom) are reported alongside as the decomposition.
The summary also reports the median share of Joule power carried by SS
(CB-CB), SC (CB-CNT), and CC (CNT-CNT) junctions at the first and last strain
points; these channel fractions help interpret why a GF is small or large.

By default the junction-opening coefficients are zero, i.e. a purely AFFINE
(ideal) network. This reports an ideal-affine reference value, not a universal
constraint; measured GF above that reference is interpreted here as evidence
for non-affine junction opening, which you can explore with --opening-alpha-*.

You set the CB and CNT volume fractions DIRECTLY (--phi-cb-vol / --phi-cnt-vol);
at least one must be positive. Self-contained -- no precomputed tables needed.

Examples
--------
Quick CNT-only demo (seconds):
    python generate/gauge_factor.py --phi-cnt-vol 0.02 --cnt-sigma-l-nm 0 \\
        --l-rve-nm 1000 --n-real 2

Hybrid at your fractions, non-affine CC opening 5 nm/strain, 10 realizations:
    python generate/gauge_factor.py --phi-cb-vol 0.02 --phi-cnt-vol 0.01 \\
        --opening-alpha-cc 5.0 --l-rve-nm 4000 --n-real 10
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

from cntcb.engines.gf_engine import run_simmons_gf_ensemble  # noqa: E402
from cntcb.engines.percolation_engine import (  # noqa: E402
    estimate_cb_n_max,
    estimate_cnt_n_max,
    solve_truncated_lognormal,
)

SAFETY = 2.5


def _parse_strain_grid(text: str) -> tuple[float, ...]:
    vals = tuple(float(x) for x in text.split(",") if x.strip())
    if len(vals) < 3:
        raise SystemExit("--strain-grid needs at least three comma-separated values (e.g. 0,0.005,0.01).")
    return vals


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
        ok = result.success and result.n_strain_ok >= 2 and math.isfinite(result.gf_r)
        if ok:
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

    strain_grid = _parse_strain_grid(args.strain_grid)
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

    results = run_simmons_gf_ensemble(
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
        strain_grid=strain_grid,
        poisson_nu=args.poisson_nu,
        seed_base=args.seed_base,
        n_realizations=n_real,
        opening_alpha_ss_nm_per_strain=args.opening_alpha_ss,
        opening_alpha_sc_nm_per_strain=args.opening_alpha_sc,
        opening_alpha_cc_nm_per_strain=args.opening_alpha_cc,
        n_cb_max=n_cb_max,
        n_cnt_max=n_cnt_max,
        workers=args.workers,
        on_result=_make_progress_callback(args.progress_every, n_real),
    )
    # Realization order is preserved, so the collected lists (and every
    # order-sensitive float reduction on them) are identical for any --workers.
    gf_r: list[float] = []
    gf_sigma: list[float] = []
    gf_geom: list[float] = []
    channel_power = {
        "ss_eps0": [],
        "sc_eps0": [],
        "cc_eps0": [],
        "ss_epsmax": [],
        "sc_epsmax": [],
        "cc_epsmax": [],
    }
    for res in results:
        if res.success and res.n_strain_ok >= 2 and math.isfinite(res.gf_r):
            gf_r.append(res.gf_r)
            gf_sigma.append(res.gf_sigma)
            gf_geom.append(res.gf_geom)
            channel_power["ss_eps0"].append(res.power_frac_ss_eps0)
            channel_power["sc_eps0"].append(res.power_frac_sc_eps0)
            channel_power["cc_eps0"].append(res.power_frac_cc_eps0)
            channel_power["ss_epsmax"].append(res.power_frac_ss_epsmax)
            channel_power["sc_epsmax"].append(res.power_frac_sc_epsmax)
            channel_power["cc_epsmax"].append(res.power_frac_cc_epsmax)

    _report(
        args,
        l_rve,
        n_real,
        strain_grid,
        n_cb_max,
        n_cnt_max,
        gf_r,
        gf_sigma,
        gf_geom,
        channel_power,
    )
    if args.out:
        _write_csv(
            Path(args.out),
            args,
            l_rve,
            n_real,
            strain_grid,
            gf_r,
            gf_sigma,
            gf_geom,
            channel_power,
        )


def _ms(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    n = len(a)
    mean = float(np.mean(a)) if n else float("nan")
    std = float(np.std(a, ddof=1)) if n > 1 else 0.0
    return mean, std


def _median(vals: list[float]) -> float:
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else float("nan")


def _channel_medians(channel_power: dict[str, list[float]]) -> dict[str, float]:
    return {key: _median(vals) for key, vals in channel_power.items()}


def _fmt_frac(value: float) -> str:
    return f"{value:.4g}" if math.isfinite(value) else "nan"


def _report(
    args,
    l_rve,
    n_real,
    strain_grid,
    n_cb_max,
    n_cnt_max,
    gf_r,
    gf_sigma,
    gf_geom,
    channel_power,
) -> None:
    m_r, s_r = _ms(gf_r)
    m_sig, _ = _ms(gf_sigma)
    m_geo, _ = _ms(gf_geom)
    ch = _channel_medians(channel_power)
    bar = "=" * 70
    print(bar)
    print(" ideal-cnt-cb-network | piezoresistive gauge-factor generator")
    print(" Monte Carlo ENSEMBLE -- reported value is mean +/- std over N runs")
    print(bar)
    print(f" phi target    : CB={args.phi_cb_vol:g}  CNT={args.phi_cnt_vol:g}  [vol fraction]")
    print(f" strain grid   : {','.join(f'{e:g}' for e in strain_grid)}   poisson_nu={args.poisson_nu:g}")
    print(f" opening alpha : ss={args.opening_alpha_ss:g} sc={args.opening_alpha_sc:g} cc={args.opening_alpha_cc:g} nm/strain  (0 = ideal/affine)")
    print(f" RVE           : {l_rve:g} nm   tunnel={args.tunnel_nm:g} nm  d_min={args.d_min_nm:g} nm  G_cut={args.g_cutoff_s:g} S")
    print(f" realizations  : {len(gf_r)} ok / {n_real}   (seed base {args.seed_base}, workers {args.workers})   n_cb_max={n_cb_max} n_cnt_max={n_cnt_max}")
    print("-" * 70)
    if not gf_r:
        print(" GF (MC)       : NaN -- no realization gave a spanning network over the strain grid")
        print(" hint          : raise phi above threshold, or enlarge --l-rve-nm")
    else:
        print(f" GF_r (MC)     : {m_r:.4g} +/- {s_r:.2g}   [d ln R / d eps]")
        print(f" decomposition : gf_sigma={m_sig:.4g}   gf_geom={m_geo:.4g}   (gf_r ~ gf_sigma + gf_geom)")
        print(" channel power : " + _format_channel_line(ch))
    print(bar)


def _format_channel_line(ch: dict[str, float]) -> str:
    eps0 = (
        f"eps0 ss={_fmt_frac(ch['ss_eps0'])} "
        f"sc={_fmt_frac(ch['sc_eps0'])} "
        f"cc={_fmt_frac(ch['cc_eps0'])}"
    )
    epsmax = (
        f"epsmax ss={_fmt_frac(ch['ss_epsmax'])} "
        f"sc={_fmt_frac(ch['sc_epsmax'])} "
        f"cc={_fmt_frac(ch['cc_epsmax'])}"
    )
    return f"{eps0}; {epsmax} [median fraction]"


def _write_csv(
    path: Path,
    args,
    l_rve,
    n_real,
    strain_grid,
    gf_r,
    gf_sigma,
    gf_geom,
    channel_power,
) -> None:
    m_r, s_r = _ms(gf_r)
    m_sig, _ = _ms(gf_sigma)
    m_geo, _ = _ms(gf_geom)
    ch = _channel_medians(channel_power)
    if path.exists():
        print(f"[warn] overwriting existing {path}", file=sys.stderr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["phi_cb_vol", "phi_cnt_vol", "gf_r_mean", "gf_r_std", "gf_sigma_mean",
             "gf_geom_mean",
             "power_frac_ss_eps0_median", "power_frac_sc_eps0_median",
             "power_frac_cc_eps0_median", "power_frac_ss_epsmax_median",
             "power_frac_sc_epsmax_median", "power_frac_cc_epsmax_median",
             "n_ok", "n_real", "strain_grid", "poisson_nu",
             "opening_alpha_ss", "opening_alpha_sc", "opening_alpha_cc",
             "l_rve_nm", "tunnel_nm", "d_min_nm", "g_cutoff_s", "workers", "seed_base"]
        )
        w.writerow(
            [args.phi_cb_vol, args.phi_cnt_vol, m_r, s_r, m_sig, m_geo,
             ch["ss_eps0"], ch["sc_eps0"], ch["cc_eps0"],
             ch["ss_epsmax"], ch["sc_epsmax"], ch["cc_epsmax"], len(gf_r),
             n_real, ";".join(f"{e:g}" for e in strain_grid), args.poisson_nu,
             args.opening_alpha_ss, args.opening_alpha_sc, args.opening_alpha_cc,
             l_rve, args.tunnel_nm, args.d_min_nm, args.g_cutoff_s, args.workers,
             args.seed_base]
        )
    print(f" wrote: {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ideal CNT/CB piezoresistive gauge factor as a Monte Carlo ensemble (mean +/- std).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Volume-fraction targets
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
    # Strain protocol
    p.add_argument("--strain-grid", default="0,0.0025,0.005,0.0075,0.01",
                   help="Comma-separated affine strain values for the GF fit.")
    p.add_argument("--poisson-nu", type=float, default=0.36)
    p.add_argument("--opening-alpha-ss", type=float, default=0.0, help="Non-affine sphere-sphere junction opening [nm/strain].")
    p.add_argument("--opening-alpha-sc", type=float, default=0.0, help="Non-affine sphere-CNT junction opening [nm/strain].")
    p.add_argument("--opening-alpha-cc", type=float, default=0.0, help="Non-affine CNT-CNT junction opening [nm/strain].")
    # Transport / RVE knobs
    p.add_argument("--l-rve-nm", "--rve-nm", type=float, default=None, help="RVE edge length [nm]; should exceed the longest CNT (recommended).")
    p.add_argument("--tunnel-nm", type=float, default=10.0)
    p.add_argument("--d-min-nm", type=float, default=0.34)
    p.add_argument("--g-cutoff-s", type=float, default=1.0e-15)
    p.add_argument("--seg-len-nm", type=float, default=50.0)
    # Stochastic
    p.add_argument("--n-real", type=int, default=None, help="Number of realizations to average.")
    p.add_argument("--workers", type=int, default=1,
                   help="Worker processes for the realization ensemble; every "
                        "realization has its own RNG seed, so any count gives "
                        "bit-identical numbers (only the wall time changes).")
    p.add_argument("--progress-every", type=int, default=10,
                   help="Print progress after this many realizations; use 0 to disable.")
    p.add_argument("--seed-base", "--seed", type=int, default=20260617)
    p.add_argument("--quick", action="store_true", help="Fast low-statistics demo (small RVE, N=2).")
    p.add_argument("--out", default=None, help="Optional CSV path for the summary row.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
