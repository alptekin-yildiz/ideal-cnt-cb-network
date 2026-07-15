"""Conductivity-defined threshold fit checks.

The helper in ``tools/fit_conductivity_powerlaw.py`` is intentionally small:
it fits the transport threshold and exponent from a loading/conductivity CSV.
These tests cover both the mathematical fit and the Pan & Li Table 2 evidence
contract shipped in ``data/article_evidence``.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from tools.fit_conductivity_powerlaw import fit_powerlaw


REPO_ROOT = Path(__file__).resolve().parents[1]
FIT_SCRIPT = REPO_ROOT / "tools" / "fit_conductivity_powerlaw.py"
PAN_DIR = REPO_ROOT / "data" / "article_evidence" / "pan2013_case_study"
PAN_SWEEP = PAN_DIR / "pan2013_conductivity_sweep_summary.csv"
PAN_FIT = PAN_DIR / "pan2013_conductivity_powerlaw_fit.csv"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_fit_powerlaw_recovers_synthetic_threshold() -> None:
    x_c = 0.5
    t = 2.0
    sigma0 = 3.0
    x = [1.0, 1.5, 2.0, 3.0, 5.0]
    sigma = [sigma0 * (xi - x_c) ** t for xi in x]

    result = fit_powerlaw(x, sigma, threshold_step=0.001)

    assert result.x_c == pytest.approx(x_c, abs=1e-12)
    assert result.t == pytest.approx(t, abs=1e-12)
    assert result.sigma0 == pytest.approx(sigma0, abs=1e-12)
    assert result.r2 == pytest.approx(1.0, abs=1e-12)


def test_pan_conductivity_fit_matches_shipped_evidence(tmp_path: Path) -> None:
    out_csv = tmp_path / "pan_fit.csv"
    proc = subprocess.run(
        [
            sys.executable,
            str(FIT_SCRIPT),
            str(PAN_SWEEP),
            "--group-col",
            "mwcnt_type",
            "--fit-include-col",
            "fit_include",
            "--min-success-fraction",
            "0.9",
            "--out",
            str(out_csv),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    assert "AR51" in proc.stdout
    assert "AR84" in proc.stdout
    assert "AR167" in proc.stdout

    generated = {row["group"]: row for row in _read_rows(out_csv)}
    expected = {
        row["mwcnt_type"]: row
        for row in _read_rows(PAN_FIT)
        if row["source"] == "model"
    }

    assert set(generated) == {"AR51", "AR84", "AR167"}
    for mwcnt_type, exp in expected.items():
        got = generated[mwcnt_type]
        assert float(got["x_c"]) == pytest.approx(
            float(exp["w_c_e_wt_pct"]), abs=1e-10
        )
        assert float(got["t"]) == pytest.approx(float(exp["t"]), abs=1e-10)
        assert float(got["sigma0"]) == pytest.approx(
            float(exp["sigma0_s_per_m"]), rel=1e-10
        )
        assert float(got["r2"]) == pytest.approx(float(exp["r2"]), abs=1e-10)
        assert int(got["n_fit"]) == int(exp["n_fit"])
        assert got["fit_include_col"] == exp["fit_include_col"]


def test_pan_sweep_rows_record_public_cli_seed_bases() -> None:
    rows = _read_rows(PAN_SWEEP)
    geom_offset = {"AR51": 0, "AR84": 100000, "AR167": 200000}
    loading_order = {"1.0": 0, "2.0": 1, "3.0": 2, "5.0": 3, "7.0": 4, "10.0": 5}

    assert len(rows) == 18
    for row in rows:
        assert row["campaign_seed_base"] == "20260706"
        expected = (
            20260706
            + geom_offset[row["mwcnt_type"]]
            + 1000 * loading_order[row["wt_pct"]]
        )
        assert int(row["row_seed_base"]) == expected


def test_pan_fit_include_is_objective_success_window() -> None:
    rows = _read_rows(PAN_SWEEP)

    for row in rows:
        try:
            sigma = float(row["sigma_s_per_m_median"])
        except ValueError:
            sigma = float("nan")
        success_fraction = int(row["n_ok"]) / int(row["n_total"])
        expected = sigma > 0.0 and success_fraction >= 0.9

        assert (row["fit_include"] == "1") is expected
