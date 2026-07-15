"""Reference-table integrity checks.

The main CSVs in ``data/reference`` are compact publication-output anchors,
not raw Monte Carlo ensembles. This test keeps the comparison tools and the
curated reference set wired into CI without re-running the long paper sweeps.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "data" / "reference"
EVIDENCE_DIR = REPO_ROOT / "data" / "article_evidence"
VERIFY_SCRIPT = REPO_ROOT / "tools" / "verify_reference_outputs.py"
VERIFY_GF_MECHANISM_SCRIPT = REPO_ROOT / "tools" / "verify_gf_mechanism_support.py"
STAGE_REFERENCE_DIRS = ("percolation", "conductivity", "gf")


def test_reference_tables_verify_against_themselves() -> None:
    reference_files = sorted(
        path
        for stage in STAGE_REFERENCE_DIRS
        for path in (REFERENCE_DIR / stage).rglob("*.csv")
    )
    assert len(reference_files) == 11
    assert any(
        path.parts[-2:] == ("percolation", "cnt_phic_sweep_fixedstep_Lmin50.csv")
        for path in reference_files
    )
    assert any(
        path.parts[-2:]
        == ("conductivity", "simmons_conductivity_regime_n300_m10_Lcb8000_Lcnt4000_Lhyb4000.csv")
        for path in reference_files
    )
    assert any(
        path.parts[-2:] == ("gf", "gf_simmons_gf_affine_cnt_3wt_n300.csv")
        for path in reference_files
    )
    assert not (REFERENCE_DIR / "percolation" / "cnt_phic_sweep_fixedstep.csv").exists()
    assert not (
        REFERENCE_DIR / "percolation" / "cb_phic_sweep_Loverd60_momentmatched.csv"
    ).exists()
    # The article-evidence package lives outside data/reference on purpose.
    assert not (REFERENCE_DIR / "gf_mechanism").exists()
    assert not (REFERENCE_DIR / "manuscript_inventory.csv").exists()
    assert not (
        EVIDENCE_DIR / "gf_mechanism" / "tableS6_gf_weaklink_upper_bound.csv"
    ).exists()

    proc = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            "--reference-dir",
            str(REFERENCE_DIR),
            "--generated-dir",
            str(REFERENCE_DIR),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, (
        f"reference verification failed\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "Compared 11 file(s); skipped 0." in proc.stdout

    assert (EVIDENCE_DIR / "manuscript_inventory.csv").is_file()
    assert (
        EVIDENCE_DIR / "gf_mechanism" / "table3_conduction_pathway_gf.csv"
    ).is_file()

    mechanism_proc = subprocess.run(
        [
            sys.executable,
            str(VERIFY_GF_MECHANISM_SCRIPT),
            "--reference-dir",
            str(REFERENCE_DIR),
            "--evidence-dir",
            str(EVIDENCE_DIR),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert mechanism_proc.returncode == 0, (
        f"GF mechanism support verification failed\n"
        f"--- stdout ---\n{mechanism_proc.stdout}\n--- stderr ---\n{mechanism_proc.stderr}"
    )
    assert "Compared 4 file(s); skipped 0." in mechanism_proc.stdout
