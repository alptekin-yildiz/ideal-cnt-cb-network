"""Distribution-path tests for the percolation threshold surface.

Covers the engine additions behind ``generate/percolation_threshold.py``'s
polydisperse/hybrid path and the CLI itself:

* equivalence of the distilled engine realizations against the inline
  machinery of the paper reproduction wrappers (Fig. 4 CNT sweep, Fig. 5 CB
  sweep) on one small same-seed realization each;
* per-realization seeding: the ensemble drivers reproduce direct
  single-realization calls, and the worker count never changes the numbers;
* CLI smoke runs for the CNT-poly, CB-poly, and hybrid modes;
* the monodisperse path stays untouched (canary + --workers warning);
* the honest CLI errors for unsupported flag combinations.

Everything here is CI-cheap (tiny RVEs, 1-4 realizations). Full-scale
Fig. 4/5/6 point comparisons are part of the release acceptance runs, not
this suite.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from cntcb.engines.percolation_engine import (
    run_cb_threshold_ensemble,
    run_cb_threshold_realization,
    run_cnt_threshold_ensemble,
    run_hybrid_cnt_threshold_realization,
    solve_truncated_lognormal,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATE_DIR = REPO_ROOT / "generate"
WRAPPER_DIR = REPO_ROOT / "reproduce_paper"

SCRIPT_TIMEOUT_S = 300

# Same canary value as tests/test_smoke.py: the distribution path must not
# perturb the monodisperse path in any way.
EXPECTED_QUICK_PHIC_CNT = 0.008772097542575668


def _load_wrapper(module_name: str):
    """Import a reproduce_paper wrapper from its file path (defs only)."""
    spec = importlib.util.spec_from_file_location(
        module_name, WRAPPER_DIR / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves string annotations via sys.modules[cls.__module__],
    # so the module must be registered before its body executes.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _run_script(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(GENERATE_DIR / "percolation_threshold.py"), *args]
    return subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=SCRIPT_TIMEOUT_S
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
    assert len(rows) == 1
    return rows[0]


def _results_equal(a, b) -> None:
    """Field-by-field equality of realization results, ignoring wall time."""
    assert type(a) is type(b)
    for field in fields(a):
        if field.name == "elapsed_s":
            continue
        va, vb = getattr(a, field.name), getattr(b, field.name)
        if isinstance(va, float) and math.isnan(va):
            assert math.isnan(vb), field.name
        else:
            assert va == vb, f"{field.name}: {va} != {vb}"


# --------------------------------------------------------------------------
# 1. Engine vs paper-wrapper equivalence (same seed, small box)
# --------------------------------------------------------------------------

def test_cb_realization_matches_fig5_wrapper() -> None:
    wrapper = _load_wrapper("run_cb_phic_sweep_pub")
    mu_d, sigma_d, d_min, d_max = 148.0, 83.0, 20.0, 500.0
    l_rve, phi_max, n_max, seed = 700.0, 0.45, 500, 20260718

    wrapper_model = wrapper._solve_parent_for_truncated_moments(
        mu_d, sigma_d, d_min, d_max
    )
    engine_model = solve_truncated_lognormal(mu_d, sigma_d, d_min, d_max)
    assert engine_model.mu_ln == pytest.approx(wrapper_model.mu_ln, rel=1e-10)
    assert engine_model.sigma_ln == pytest.approx(wrapper_model.sigma_ln, rel=1e-10)

    wres = wrapper._run_realization(
        (wrapper_model.mu_ln, wrapper_model.sigma_ln, n_max, phi_max,
         l_rve, d_min, d_max, seed)
    )
    eres = run_cb_threshold_realization(
        cb_model=engine_model,
        phi_cb_max_vol=phi_max,
        l_rve_nm=l_rve,
        tunnel_nm=wrapper.TUNNEL_NM,
        seed=seed,
        n_cb_max=n_max,
    )
    assert eres.percolated == wres.percolated
    assert eres.n_cb == wres.n_cb
    assert eres.phi_cb_stop_vol == pytest.approx(wres.phi_stop_vol, rel=1e-9)


def test_cnt_realization_matches_fig4_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = _load_wrapper("run_cnt_phic_sweep_fixedstep")
    l_rve = 1000.0
    monkeypatch.setattr(wrapper, "L_RVE", l_rve)
    mu_l, sigma_l = 500.0, 300.0
    phi_target, n_max, seed = 0.02, 600, 20260718

    mu_ln, sigma_ln = wrapper._solve_lognormal_truncated(
        mu_l, sigma_l, wrapper.L_MIN_NM, wrapper.L_MAX_NM
    )
    cnt_model = solve_truncated_lognormal(
        mu_l, sigma_l, wrapper.L_MIN_NM, wrapper.L_MAX_NM
    )
    # Different root solvers (fsolve vs damped Newton) on the same moment
    # equations: parameters must agree tightly but not bit-for-bit.
    assert cnt_model.mu_ln == pytest.approx(mu_ln, rel=1e-6)
    assert cnt_model.sigma_ln == pytest.approx(sigma_ln, rel=1e-6)

    wres = wrapper._run_realization((mu_ln, sigma_ln, n_max, phi_target, seed))
    eres = run_hybrid_cnt_threshold_realization(
        cb_model=solve_truncated_lognormal(148.0, 0.0, 20.0, 500.0),
        cnt_model=cnt_model,
        phi_cb_vol=0.0,
        phi_cnt_max_vol=phi_target,
        l_rve_nm=l_rve,
        cnt_diam_nm=wrapper.CNT_DIAM_NM,
        waviness=wrapper.WAVINESS,
        tunnel_nm=wrapper.TUNNEL_NM,
        seg_len_unit_nm=wrapper.SEG_LEN_UNIT_NM,
        seed=seed,
        n_cnt_max=n_max,
    )
    assert eres.percolated == wres.percolated
    assert eres.n_cnt == wres.n_cnt
    assert eres.phi_cnt_stop_vol == pytest.approx(wres.phi_stop_vol, rel=1e-6)


# --------------------------------------------------------------------------
# 2. Ensemble drivers: seeding and worker-count invariance
# --------------------------------------------------------------------------

def _small_cnt_ensemble(workers: int):
    return run_cnt_threshold_ensemble(
        cnt_model=solve_truncated_lognormal(500.0, 300.0, 50.0, 1500.0),
        cb_model=solve_truncated_lognormal(148.0, 83.0, 20.0, 500.0),
        phi_cb_vol=0.02,
        phi_cnt_max_vol=0.03,
        l_rve_nm=800.0,
        cnt_diam_nm=10.0,
        waviness=0.7,
        tunnel_nm=10.0,
        seg_len_unit_nm=50.0,
        seed_base=42,
        n_realizations=4,
        workers=workers,
    )


def test_ensemble_reproduces_direct_realization_calls() -> None:
    results = _small_cnt_ensemble(workers=1)
    direct = run_hybrid_cnt_threshold_realization(
        cb_model=solve_truncated_lognormal(148.0, 83.0, 20.0, 500.0),
        cnt_model=solve_truncated_lognormal(500.0, 300.0, 50.0, 1500.0),
        phi_cb_vol=0.02,
        phi_cnt_max_vol=0.03,
        l_rve_nm=800.0,
        cnt_diam_nm=10.0,
        waviness=0.7,
        tunnel_nm=10.0,
        seg_len_unit_nm=50.0,
        seed=42 + 2,
    )
    _results_equal(results[2], direct)


def test_cnt_ensemble_workers_bit_identical() -> None:
    serial = _small_cnt_ensemble(workers=1)
    parallel = _small_cnt_ensemble(workers=2)
    assert len(serial) == len(parallel) == 4
    for a, b in zip(serial, parallel):
        _results_equal(a, b)


def test_cb_ensemble_workers_bit_identical() -> None:
    kwargs = dict(
        cb_model=solve_truncated_lognormal(148.0, 83.0, 20.0, 500.0),
        phi_cb_max_vol=0.45,
        l_rve_nm=700.0,
        tunnel_nm=10.0,
        seed_base=42,
        n_realizations=4,
    )
    serial = run_cb_threshold_ensemble(workers=1, **kwargs)
    parallel = run_cb_threshold_ensemble(workers=2, **kwargs)
    for a, b in zip(serial, parallel):
        _results_equal(a, b)


# --------------------------------------------------------------------------
# 3. CLI smoke runs for the three distribution modes
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("mode", "extra_args"),
    [
        ("cnt_poly", ["--filler", "cnt", "--cnt-sigma-l-nm", "300"]),
        ("cb_poly", ["--filler", "cb", "--cb-sigma-d-nm", "83"]),
        ("hybrid", ["--filler", "cnt", "--phi-cb-vol", "0.02"]),
    ],
)
def test_distribution_cli_smoke(tmp_path: Path, mode: str, extra_args: list[str]) -> None:
    out_csv = tmp_path / f"{mode}.csv"
    proc = _run_script(
        [*extra_args, "--quick", "--n-real", "2", "--seed", "42", "--out", str(out_csv)],
        cwd=tmp_path,
    )
    _assert_ok(proc)
    assert "DISTRIBUTION" in proc.stdout
    assert "phi_c (MC)" in proc.stdout

    row = _read_single_csv_row(out_csv)
    assert row["path"] == "distribution"
    assert int(row["n_total"]) == 2
    phi_median = float(row["phi_c_median_vol"])
    assert math.isfinite(phi_median), "median threshold is NaN (all runs censored)"
    assert 0.0 < phi_median < 1.0


def test_distribution_cli_workers_match_serial(tmp_path: Path) -> None:
    rows = []
    for workers in ("1", "2"):
        out_csv = tmp_path / f"w{workers}.csv"
        proc = _run_script(
            ["--filler", "cnt", "--cnt-sigma-l-nm", "300", "--quick",
             "--n-real", "2", "--workers", workers, "--out", str(out_csv)],
            cwd=tmp_path,
        )
        _assert_ok(proc)
        rows.append(_read_single_csv_row(out_csv))
    for key, value in rows[0].items():
        if key != "workers":
            assert rows[1][key] == value, key


def test_distribution_cli_progress_every_reports(tmp_path: Path) -> None:
    proc = _run_script(
        ["--filler", "cnt", "--cnt-sigma-l-nm", "300", "--quick",
         "--n-real", "2", "--progress-every", "1"],
        cwd=tmp_path,
    )
    _assert_ok(proc)
    assert "progress 1/2" in proc.stdout
    assert "progress 2/2" in proc.stdout


def test_mono_cli_progress_every_reports(tmp_path: Path) -> None:
    proc = _run_script(
        ["--filler", "cnt", "--quick", "--n-real", "2",
         "--progress-every", "1"],
        cwd=tmp_path,
    )
    _assert_ok(proc)
    assert "progress 1/2" in proc.stdout
    assert "progress 2/2" in proc.stdout


# --------------------------------------------------------------------------
# 4. The monodisperse path stays exactly as it was
# --------------------------------------------------------------------------

def test_mono_path_ignores_workers_with_warning(tmp_path: Path) -> None:
    out_csv = tmp_path / "mono.csv"
    proc = _run_script(
        ["--filler", "cnt", "--quick", "--n-real", "1", "--seed", "42",
         "--workers", "4", "--out", str(out_csv)],
        cwd=tmp_path,
    )
    _assert_ok(proc)
    assert "monodisperse path" in proc.stderr
    row = _read_single_csv_row(out_csv)
    assert float(row["phi_c_mean_vol"]) == pytest.approx(
        EXPECTED_QUICK_PHIC_CNT, rel=1e-12, abs=1e-15
    )


# --------------------------------------------------------------------------
# 5. Honest errors for unsupported combinations
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("args", "needle"),
    [
        (["--filler", "cb", "--phi-cb-vol", "0.1"], "--filler cnt --phi-cb-vol"),
        (["--filler", "cb", "--cnt-mu-l-nm", "500"], "not implemented"),
        (["--filler", "cnt", "--cb-mu-d-nm", "148"], "--phi-cb-vol > 0"),
        # A negative background fraction must error out loudly instead of
        # failing the > 0 dispatch test and silently running the mono path.
        (["--filler", "cnt", "--phi-cb-vol", "-0.1"], "--phi-cb-vol must be >= 0"),
        # A non-positive stop cap must error instead of censoring every
        # realization into a NaN ensemble.
        (
            ["--filler", "cnt", "--cnt-sigma-l-nm", "300", "--phi-max-vol", "-0.1"],
            "--phi-max-vol must be > 0",
        ),
        (["--filler", "cnt", "--phi-cb-vol", "1.5"], "--phi-cb-vol must be < 1"),
        (["--filler", "cnt", "--cnt-sigma-l-nm", "300", "--n-real", "0"],
         "--n-real must be >= 1"),
    ],
)
def test_distribution_cli_rejects_unsupported(
    tmp_path: Path, args: list[str], needle: str
) -> None:
    proc = _run_script([*args, "--quick"], cwd=tmp_path)
    assert proc.returncode != 0
    assert needle in proc.stderr
