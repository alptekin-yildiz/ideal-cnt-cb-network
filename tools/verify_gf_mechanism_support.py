#!/usr/bin/env python3
"""Verify the compact GF mechanism support package.

This check is separate from ``verify_reference_outputs.py`` because the support
package lives under ``data/article_evidence/`` and contains a derived audit
table, not one-to-one regenerated summary CSVs. Values derived from formatted
per-realization CSVs are compared with a small explicit tolerance instead of
the stricter reference-output tolerance.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


REL_TOL = 1.0e-6
ABS_TOL = 1.0e-8

MECH_DIR = Path("gf_mechanism")
AUDIT_FILE = MECH_DIR / "gf_mech100_20260620_205743_audit.csv"
H1_IDEAL_FILE = MECH_DIR / "gf_simmons_gf_mech100_20260620_205743_h1_ideal.csv"
WEAKLINK_FILE = MECH_DIR / "gf_weaklink_upper_bound.csv"
TABLE3_FILE = MECH_DIR / "table3_conduction_pathway_gf.csv"

MAIN_CB_FILE = Path("gf/gf_simmons_gf_affine_cb_40wt_50wt_n300.csv")
MAIN_CNT_FILE = Path("gf/gf_simmons_gf_affine_cnt_3wt_n300.csv")

REFERENCE_ROOT = Path("data/reference")
EVIDENCE_ROOT = Path("data/article_evidence")
GENERATED_GF_PREFIX = "data/processed/gf/"


def _repo_path(path: Path) -> str:
    return path.as_posix()


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _one(rows: list[dict[str, str]], *, source: Path, **criteria: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if all(str(row.get(key, "")) == str(value) for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise AssertionError(f"{source}: expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def _num(value: str) -> float:
    if value == "":
        return float("nan")
    return float(value)


def _assert_close(path: Path, label: str, got: str, expected: str) -> None:
    got_f = _num(got)
    exp_f = _num(expected)
    if math.isnan(got_f) and math.isnan(exp_f):
        return
    if not math.isclose(got_f, exp_f, rel_tol=REL_TOL, abs_tol=ABS_TOL):
        raise AssertionError(
            f"{path}: {label} mismatch: got {got!r}, expected {expected!r} "
            f"(rel_tol={REL_TOL:g}, abs_tol={ABS_TOL:g})"
        )


def _check_generated_source_paths(
    path: Path,
    rows: list[dict[str, str]],
    columns: tuple[str, ...],
) -> None:
    for line_no, row in enumerate(rows, start=2):
        for column in columns:
            value = row.get(column, "")
            if not value:
                continue
            if not value.startswith(GENERATED_GF_PREFIX):
                raise AssertionError(
                    f"{path}: line {line_no} {column} should start with "
                    f"{GENERATED_GF_PREFIX!r}, got {value!r}"
                )


def _expected_table3_rows(reference_dir: Path, evidence_dir: Path) -> list[dict[str, str]]:
    mech_path = evidence_dir / AUDIT_FILE
    weak_path = evidence_dir / WEAKLINK_FILE
    cb_path = reference_dir / MAIN_CB_FILE
    cnt_path = reference_dir / MAIN_CNT_FILE

    mech = _read_rows(mech_path)
    weak = _read_rows(weak_path)
    cb = _read_rows(cb_path)
    cnt = _read_rows(cnt_path)

    cb50 = _one(cb, source=cb_path, system="cb", phi_multiplier="1.3132717")
    cnt3 = _one(cnt, source=cnt_path, system="cnt", phi_multiplier="3.402")

    rows: list[dict[str, str]] = [
        {
            "composition_code": "cb50",
            "composition": "CB (50 wt.%)",
            "power_frac_ss_eps0_median": "1",
            "power_frac_sc_eps0_median": "0",
            "power_frac_cc_eps0_median": "0",
            "GF_R_ideal_median": cb50["GF_R_median"],
            "GF_R_cc_alpha10_median": "",
            "GF_R_cc_alpha40_median": "",
            "source_power": "definition: single-filler CB network",
            "source_ideal": _repo_path(REFERENCE_ROOT / MAIN_CB_FILE),
            "source_alpha10": "",
            "source_alpha40": "",
        },
        {
            "composition_code": "cnt3",
            "composition": "CNT (3 wt.%)",
            "power_frac_ss_eps0_median": "0",
            "power_frac_sc_eps0_median": "0",
            "power_frac_cc_eps0_median": "1",
            "GF_R_ideal_median": cnt3["GF_R_median"],
            "GF_R_cc_alpha10_median": _one(
                weak, source=weak_path, system="Pure CNT", alpha_nm_per_strain="10"
            )["GF_R_median"],
            "GF_R_cc_alpha40_median": _one(
                weak, source=weak_path, system="Pure CNT", alpha_nm_per_strain="40"
            )["GF_R_median"],
            "source_power": "definition: single-filler CNT network",
            "source_ideal": _repo_path(REFERENCE_ROOT / MAIN_CNT_FILE),
            "source_alpha10": _repo_path(EVIDENCE_ROOT / WEAKLINK_FILE),
            "source_alpha40": _repo_path(EVIDENCE_ROOT / WEAKLINK_FILE),
        },
    ]

    for comp, label in (
        ("h1", "Hybrid (3 wt.% CNT/1 wt.% CB)"),
        ("h5", "Hybrid (3 wt.% CNT/5 wt.% CB)"),
        ("h10", "Hybrid (3 wt.% CNT/10 wt.% CB)"),
    ):
        ideal = _one(
            mech,
            source=mech_path,
            composition_code=comp,
            channel="ideal",
            alpha_nm_per_strain="0",
        )
        a10 = _one(
            mech,
            source=mech_path,
            composition_code=comp,
            channel="CC",
            alpha_nm_per_strain="10",
        )
        a40 = _one(
            mech,
            source=mech_path,
            composition_code=comp,
            channel="CC",
            alpha_nm_per_strain="40",
        )
        rows.append(
            {
                "composition_code": comp,
                "composition": label,
                "power_frac_ss_eps0_median": ideal["power_frac_ss_eps0_median"],
                "power_frac_sc_eps0_median": ideal["power_frac_sc_eps0_median"],
                "power_frac_cc_eps0_median": ideal["power_frac_cc_eps0_median"],
                "GF_R_ideal_median": ideal["GF_R_median"],
                "GF_R_cc_alpha10_median": a10["GF_R_median"],
                "GF_R_cc_alpha40_median": a40["GF_R_median"],
                "source_power": _repo_path(EVIDENCE_ROOT / AUDIT_FILE),
                "source_ideal": _repo_path(EVIDENCE_ROOT / AUDIT_FILE),
                "source_alpha10": _repo_path(EVIDENCE_ROOT / AUDIT_FILE),
                "source_alpha40": _repo_path(EVIDENCE_ROOT / AUDIT_FILE),
            }
        )
    return rows


def _check_required_fig10_rows(evidence_dir: Path) -> None:
    audit_path = evidence_dir / AUDIT_FILE
    weak_path = evidence_dir / WEAKLINK_FILE
    audit = _read_rows(audit_path)
    weak = _read_rows(weak_path)
    _check_generated_source_paths(
        audit_path,
        audit,
        ("source_summary_csv", "source_realizations_csv"),
    )
    _check_generated_source_paths(weak_path, weak, ("source_summary_csv",))

    for channel in ("all", "CC", "SC", "SS"):
        for alpha in ("5", "10", "20", "40"):
            _one(
                audit,
                source=audit_path,
                composition_code="h1",
                channel=channel,
                alpha_nm_per_strain=alpha,
            )
    for comp in ("h1", "h5", "h10"):
        _one(audit, source=audit_path, composition_code=comp, channel="ideal")
        for alpha in ("5", "10", "20", "40"):
            _one(
                audit,
                source=audit_path,
                composition_code=comp,
                channel="CC",
                alpha_nm_per_strain=alpha,
            )
    for alpha in ("5", "10", "20", "40"):
        _one(weak, source=weak_path, system="Pure CNT", alpha_nm_per_strain=alpha)


def _check_h1_ideal(evidence_dir: Path) -> None:
    audit_path = evidence_dir / AUDIT_FILE
    h1_path = evidence_dir / H1_IDEAL_FILE
    audit_h1 = _one(
        _read_rows(audit_path),
        source=audit_path,
        composition_code="h1",
        channel="ideal",
        alpha_nm_per_strain="0",
    )
    h1_row = _read_rows(h1_path)[0]
    for key in ("GF_R_median", "GF_R_p25", "GF_R_p75", "GF_sigma_median"):
        _assert_close(h1_path, key, h1_row[key], audit_h1[key])


def _check_table3(reference_dir: Path, evidence_dir: Path) -> None:
    table3_path = evidence_dir / TABLE3_FILE
    rows = _read_rows(table3_path)
    expected = _expected_table3_rows(reference_dir, evidence_dir)
    if len(rows) != len(expected):
        raise AssertionError(f"{table3_path}: expected {len(expected)} rows, found {len(rows)}")

    comparable = (
        "composition_code",
        "composition",
        "power_frac_ss_eps0_median",
        "power_frac_sc_eps0_median",
        "power_frac_cc_eps0_median",
        "GF_R_ideal_median",
        "GF_R_cc_alpha10_median",
        "GF_R_cc_alpha40_median",
        "source_power",
        "source_ideal",
        "source_alpha10",
        "source_alpha40",
    )
    for idx, (row, exp) in enumerate(zip(rows, expected), start=1):
        for key in comparable:
            if key.startswith(("power_", "GF_")):
                _assert_close(table3_path, f"row {idx} {key}", row[key], exp[key])
            elif row[key] != exp[key]:
                raise AssertionError(
                    f"{table3_path}: row {idx} {key} mismatch: "
                    f"got {row[key]!r}, expected {exp[key]!r}"
                )


def verify(reference_dir: Path, evidence_dir: Path) -> None:
    _check_required_fig10_rows(evidence_dir)
    print(f"OK   {AUDIT_FILE}")
    print(f"OK   {WEAKLINK_FILE}")

    _check_h1_ideal(evidence_dir)
    print(f"OK   {H1_IDEAL_FILE}")

    _check_table3(reference_dir, evidence_dir)
    print(f"OK   {TABLE3_FILE}")
    print("Compared 4 file(s); skipped 0.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("data/reference"),
        help="Root of the main stage reference directory (gf/ tables).",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path("data/article_evidence"),
        help="Root of the article-evidence directory (gf_mechanism/ tables).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify(args.reference_dir, args.evidence_dir)


if __name__ == "__main__":
    main()
