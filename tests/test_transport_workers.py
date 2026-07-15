"""--workers parity for the transport generators.

Both ``generate/conductivity.py`` and ``generate/gauge_factor.py`` run their
Monte Carlo ensembles through the engine-level ensemble drivers, which give
every realization its own RNG seed and return results in realization order.
These tests pin the contract that the worker count never changes the numbers:

1. engine level -- the ensemble reproduces direct realization calls, and a
   process pool gives bit-identical results to the serial run;
2. CLI level -- ``--workers 2`` writes a CSV row identical to ``--workers 1``
   except for the ``workers`` provenance column itself.

The serial path is additionally pinned by the fixed-seed canaries in
``test_smoke.py`` (rel 1e-10), which did not change with this feature.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from cntcb.engines.gf_engine import run_simmons_gf_ensemble  # noqa: E402
from cntcb.engines.percolation_engine import solve_truncated_lognormal  # noqa: E402
from cntcb.engines.simmons_conductivity_engine import (  # noqa: E402
    run_simmons_conductivity_ensemble,
    run_simmons_conductivity_realization,
)

# Cheap-but-real transport settings: monodisperse 500 nm CNTs well above
# threshold in a 1000 nm box (the same regime as the quick CLI demos).
CNT_KWARGS = dict(
    phi_cb_target_vol=0.0,
    phi_cnt_target_vol=0.02,
    l_rve_nm=1000.0,
    cnt_diam_nm=10.0,
    waviness=0.7,
    tunnel_nm=10.0,
    seg_len_unit_nm=50.0,
    d_min_nm=0.34,
    g_cutoff_s=1.0e-15,
)
SEED_BASE = 20260718


def _models():
    cb_model = solve_truncated_lognormal(148.0, 83.0, 20.0, 500.0)
    cnt_model = solve_truncated_lognormal(500.0, 0.0, 50.0, 1500.0)
    return cb_model, cnt_model


def _results_equal(a, b) -> None:
    for f in fields(a):
        if f.name == "elapsed_s":
            continue
        va, vb = getattr(a, f.name), getattr(b, f.name)
        assert va == vb or (va != va and vb != vb), f.name


# --------------------------------------------------------------------------
# 1. Engine level
# --------------------------------------------------------------------------

def test_conductivity_ensemble_reproduces_direct_calls() -> None:
    cb_model, cnt_model = _models()
    ensemble = run_simmons_conductivity_ensemble(
        cb_model=cb_model,
        cnt_model=cnt_model,
        seed_base=SEED_BASE,
        n_realizations=2,
        **CNT_KWARGS,
    )
    direct = run_simmons_conductivity_realization(
        cb_model=cb_model, cnt_model=cnt_model, seed=SEED_BASE + 1, **CNT_KWARGS
    )
    assert len(ensemble) == 2
    _results_equal(ensemble[1], direct)


def test_conductivity_ensemble_workers_bit_identical() -> None:
    cb_model, cnt_model = _models()
    common = dict(
        cb_model=cb_model,
        cnt_model=cnt_model,
        seed_base=SEED_BASE,
        n_realizations=2,
        **CNT_KWARGS,
    )
    serial = run_simmons_conductivity_ensemble(workers=1, **common)
    parallel = run_simmons_conductivity_ensemble(workers=2, **common)
    for a, b in zip(serial, parallel, strict=True):
        _results_equal(a, b)


def test_gf_ensemble_workers_bit_identical() -> None:
    cb_model, cnt_model = _models()
    common = dict(
        cb_model=cb_model,
        cnt_model=cnt_model,
        strain_grid=(0.0, 0.005, 0.01),
        poisson_nu=0.36,
        seed_base=SEED_BASE,
        n_realizations=2,
        **CNT_KWARGS,
    )
    serial = run_simmons_gf_ensemble(workers=1, **common)
    parallel = run_simmons_gf_ensemble(workers=2, **common)
    for a, b in zip(serial, parallel, strict=True):
        _results_equal(a, b)


# --------------------------------------------------------------------------
# 2. CLI level
# --------------------------------------------------------------------------

def _run_script(script: str, argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "generate" / script), *argv],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=600,
    )


def _read_single_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize(
    ("script", "extra_args"),
    [
        ("conductivity.py", []),
        ("gauge_factor.py", ["--strain-grid", "0,0.005,0.01"]),
    ],
)
def test_cli_workers_match_serial(
    tmp_path: Path, script: str, extra_args: list[str]
) -> None:
    rows = []
    for workers in (1, 2):
        out_csv = tmp_path / f"{script}_{workers}.csv"
        proc = _run_script(
            script,
            ["--phi-cnt-vol", "0.02", "--cnt-sigma-l-nm", "0",
             "--l-rve-nm", "1000", "--n-real", "2",
             "--workers", str(workers), "--out", str(out_csv), *extra_args],
            cwd=tmp_path,
        )
        assert proc.returncode == 0, proc.stderr
        assert f"workers {workers}" in proc.stdout
        rows.append(_read_single_csv_row(out_csv))
    for key, value in rows[0].items():
        if key != "workers":
            assert rows[1][key] == value, key


@pytest.mark.parametrize(
    ("script", "extra_args"),
    [
        ("conductivity.py", []),
        ("gauge_factor.py", ["--strain-grid", "0,0.005,0.01"]),
    ],
)
def test_cli_progress_every_reports(
    tmp_path: Path, script: str, extra_args: list[str]
) -> None:
    proc = _run_script(
        script,
        ["--phi-cnt-vol", "0.02", "--cnt-sigma-l-nm", "0",
         "--l-rve-nm", "1000", "--n-real", "2",
         "--progress-every", "1", *extra_args],
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "progress 1/2" in proc.stdout
    assert "progress 2/2" in proc.stdout


@pytest.mark.parametrize("script", ["conductivity.py", "gauge_factor.py"])
def test_cli_rejects_bad_workers(tmp_path: Path, script: str) -> None:
    proc = _run_script(
        script,
        ["--phi-cnt-vol", "0.02", "--workers", "0"],
        cwd=tmp_path,
    )
    assert proc.returncode != 0
    assert "--workers must be >= 1" in proc.stderr
