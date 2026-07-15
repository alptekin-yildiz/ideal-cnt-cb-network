"""Install-and-run smoke tests for ideal-cnt-cb-network.

These tests check that the package imports and that each CLI generator in
``generate/`` runs end-to-end on a deliberately tiny problem, exits 0, and
produces finite numbers on stdout and in its ``--out`` CSV.

For fixed seeds, the tests also pin tiny deterministic canary values.  These
are NOT publication validation tables, but they are strong guards against
accidentally changing scientific outputs during documentation or dead-code
cleanup.  The tight tolerances are regression guards on the CI platforms, not
portability claims: on a different BLAS/NumPy stack, a last-digit mismatch
should first be checked as platform variance rather than read as scientific
error.  Total suite runtime is a few seconds.

Parameter choices (why these and not the README quick demos)
------------------------------------------------------------
* percolation_threshold.py: the ``--quick`` presets with ``--n-real 1`` and a
  pinned seed percolate for every seed tried (both fillers), in ~0.2 s.
* conductivity.py / gauge_factor.py: the repo's original CB-only demo
  (``--phi-cb-vol 0.35 --l-rve-nm 1000``, since replaced in the README) is
  NaN-prone at ``--n-real 1``:
  after the ``--g-cutoff-s`` pruning of weak Simmons edges, a CB-only network
  at that volume fraction spans only for some seeds (about half, even at
  phi = 0.50).  A CNT-only network at phi = 0.02 with monodisperse 500 nm
  tubes (``--cnt-sigma-l-nm 0``, a documented option) is far above its
  percolation threshold, satisfies the documented "RVE larger than the
  longest CNT" constraint (500 nm < 1000 nm), and produced a spanning
  network for 11/11 seed bases tried.  Seeds below are pinned to the script
  defaults, which are inside that verified set.
"""

from __future__ import annotations

import csv
import importlib
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATE_DIR = REPO_ROOT / "generate"

# Each script finishes in well under 5 s on a 2020s laptop; the timeout only
# guards CI against a hang.
SCRIPT_TIMEOUT_S = 300

# Fixed-seed canaries captured from the committed reference implementation.
# Keep these tiny runs cheap; use data/reference plus reproduce_paper/ for
# full publication-output verification.
EXPECTED_QUICK_PHIC = {
    "cnt": 0.008772097542575668,
    "cb": 0.35815104593026004,
}
EXPECTED_QUICK_SIGMA = 209.6891457694971
EXPECTED_QUICK_GF_R = -17.927781120392794
EXPECTED_QUICK_GF_SIGMA = -19.644113324531702
EXPECTED_QUICK_GF_GEOM = 1.7163322041392755


# --------------------------------------------------------------------------
# 1. Package imports
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "module_name",
    [
        "cntcb",
        "cntcb.kernel",
        "cntcb.kernel.constants",
        "cntcb.kernel.materials",
        "cntcb.kernel.percolation_finder",
        "cntcb.engines",
        "cntcb.engines.percolation_engine",
        "cntcb.engines.conductivity_engine",
        "cntcb.engines.simmons_conductivity_engine",
        "cntcb.engines.gf_engine",
    ],
)
def test_import(module_name: str) -> None:
    importlib.import_module(module_name)


# --------------------------------------------------------------------------
# Helpers for running the generate/ scripts as an end user would
# --------------------------------------------------------------------------

def _run_script(script_name: str, args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run ``generate/<script_name>`` in a subprocess.

    ``PYTHONPATH`` is prepended with the repo root so the script can import
    ``cntcb`` even on a bare checkout (running ``python generate/x.py`` puts
    ``generate/``, not the repo root, on ``sys.path``).  With the package
    pip-installed (CI) this is redundant but harmless.  ``cwd`` is the pytest
    tmp dir so any stray relative output could never land in the repo tree.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(GENERATE_DIR / script_name), *args]
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=SCRIPT_TIMEOUT_S,
    )


def _assert_ok(proc: subprocess.CompletedProcess) -> None:
    assert proc.returncode == 0, (
        f"script exited with {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


def _read_single_csv_row(path: Path) -> dict[str, str]:
    assert path.is_file(), f"expected output CSV at {path}"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1, f"expected exactly one summary row in {path}, got {len(rows)}"
    return rows[0]


# --------------------------------------------------------------------------
# 2. generate/percolation_threshold.py
# --------------------------------------------------------------------------

@pytest.mark.parametrize("filler", ["cnt", "cb"])
def test_percolation_threshold_script(tmp_path: Path, filler: str) -> None:
    out_csv = tmp_path / f"phic_{filler}.csv"
    proc = _run_script(
        "percolation_threshold.py",
        [
            "--filler", filler,
            "--quick",            # small preset RVE / n_max
            "--n-real", "1",      # single realization: cheapest valid run
            "--seed", "42",
            "--out", str(out_csv),
        ],
        cwd=tmp_path,
    )
    _assert_ok(proc)
    assert "percolation threshold generator" in proc.stdout
    assert "phi_c (MC)" in proc.stdout

    row = _read_single_csv_row(out_csv)
    phi_mc = float(row["phi_c_mean_vol"])
    phi_theory = float(row["phi_c_analytical_vol"])
    assert math.isfinite(phi_mc), "MC threshold is NaN (realization did not percolate)"
    assert 0.0 < phi_mc < 1.0, f"phi_c out of (0, 1): {phi_mc}"
    assert math.isfinite(phi_theory) and 0.0 < phi_theory < 1.0
    assert int(row["n_realizations"]) >= 1
    assert phi_mc == pytest.approx(EXPECTED_QUICK_PHIC[filler], rel=1e-12, abs=1e-15)


# --------------------------------------------------------------------------
# 3. generate/conductivity.py
# --------------------------------------------------------------------------

def test_conductivity_script(tmp_path: Path) -> None:
    out_csv = tmp_path / "sigma.csv"
    proc = _run_script(
        "conductivity.py",
        [
            "--phi-cnt-vol", "0.02",     # well above the CNT threshold
            "--cnt-sigma-l-nm", "0",     # monodisperse 500 nm tubes < RVE
            "--l-rve-nm", "1000",
            "--n-real", "1",
            "--seed-base", "20260610",   # script default, in the verified seed set
            "--out", str(out_csv),
        ],
        cwd=tmp_path,
    )
    _assert_ok(proc)
    assert "conductivity generator" in proc.stdout
    assert "sigma (MC)" in proc.stdout
    assert "channel power" in proc.stdout

    row = _read_single_csv_row(out_csv)
    assert int(row["n_ok"]) >= 1, "no realization produced a spanning network"
    sigma = float(row["sigma_mean_s_per_m"])
    # Smoke contract only: finite and within an absurdly generous range.
    assert math.isfinite(sigma), "sigma is NaN"
    assert 0.0 < sigma < 1.0e9, f"sigma outside plausible range [S/m]: {sigma}"
    assert sigma == pytest.approx(EXPECTED_QUICK_SIGMA, rel=1e-10, abs=1e-10)
    assert float(row["power_frac_ss_median"]) == pytest.approx(0.0, abs=1e-15)
    assert float(row["power_frac_sc_median"]) == pytest.approx(0.0, abs=1e-15)
    assert float(row["power_frac_cc_median"]) == pytest.approx(1.0, abs=1e-15)


# --------------------------------------------------------------------------
# 4. generate/gauge_factor.py
# --------------------------------------------------------------------------

def test_gauge_factor_script(tmp_path: Path) -> None:
    out_csv = tmp_path / "gf.csv"
    proc = _run_script(
        "gauge_factor.py",
        [
            "--phi-cnt-vol", "0.02",
            "--cnt-sigma-l-nm", "0",
            "--l-rve-nm", "1000",
            "--n-real", "1",
            "--strain-grid", "0,0.005,0.01",  # 3 strain points = 3 network solves
            "--seed-base", "20260617",        # script default, verified
            "--out", str(out_csv),
        ],
        cwd=tmp_path,
    )
    _assert_ok(proc)
    assert "gauge-factor generator" in proc.stdout
    assert "GF_r (MC)" in proc.stdout
    assert "channel power" in proc.stdout

    row = _read_single_csv_row(out_csv)
    assert int(row["n_ok"]) >= 1, "no realization spanned over the strain grid"
    gf_r = float(row["gf_r_mean"])
    gf_sigma = float(row["gf_sigma_mean"])
    gf_geom = float(row["gf_geom_mean"])
    # GF of an ideal affine network can legitimately be negative; assert
    # finiteness and magnitude only.
    assert math.isfinite(gf_r), "gf_r is NaN"
    assert abs(gf_r) < 1.0e4, f"gf_r outside plausible magnitude: {gf_r}"
    assert math.isfinite(gf_sigma) and math.isfinite(gf_geom)
    assert gf_r == pytest.approx(EXPECTED_QUICK_GF_R, rel=1e-10, abs=1e-10)
    assert gf_sigma == pytest.approx(EXPECTED_QUICK_GF_SIGMA, rel=1e-10, abs=1e-10)
    assert gf_geom == pytest.approx(EXPECTED_QUICK_GF_GEOM, rel=1e-10, abs=1e-10)
    assert float(row["power_frac_ss_eps0_median"]) == pytest.approx(0.0, abs=1e-15)
    assert float(row["power_frac_sc_eps0_median"]) == pytest.approx(0.0, abs=1e-15)
    assert float(row["power_frac_cc_eps0_median"]) == pytest.approx(1.0, abs=1e-15)
    assert float(row["power_frac_ss_epsmax_median"]) == pytest.approx(0.0, abs=1e-15)
    assert float(row["power_frac_sc_epsmax_median"]) == pytest.approx(0.0, abs=1e-15)
    assert float(row["power_frac_cc_epsmax_median"]) == pytest.approx(1.0, abs=1e-15)
