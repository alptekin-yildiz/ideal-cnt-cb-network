#!/usr/bin/env python3
"""Compare regenerated paper summary CSVs with frozen reference tables.

This script is intentionally read-only. It does not run Monte Carlo sweeps; it
checks CSV files that were already generated under data/processed against the
curated stage-summary tables in data/reference.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


DEFAULT_IGNORED_COLUMNS = ("elapsed_s",)
DEFAULT_REFERENCE_SUBDIRS = ("percolation", "conductivity", "gf")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        return list(reader.fieldnames), list(reader)


def _as_float(value: str) -> float | None:
    text = value.strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _values_match(expected: str, actual: str, *, rtol: float, atol: float) -> bool:
    if expected == actual:
        return True

    exp_num = _as_float(expected)
    act_num = _as_float(actual)
    if exp_num is None or act_num is None:
        return False

    if math.isnan(exp_num) or math.isnan(act_num):
        return math.isnan(exp_num) and math.isnan(act_num)

    return math.isclose(exp_num, act_num, rel_tol=rtol, abs_tol=atol)


def _compare_file(
    reference_path: Path,
    generated_path: Path,
    label: str,
    *,
    ignored_columns: set[str],
    rtol: float,
    atol: float,
    max_errors: int,
) -> list[str]:
    ref_header, ref_rows = _read_csv(reference_path)
    gen_header, gen_rows = _read_csv(generated_path)

    ref_cols = [col for col in ref_header if col not in ignored_columns]
    gen_cols = [col for col in gen_header if col not in ignored_columns]

    errors: list[str] = []
    if ref_cols != gen_cols:
        errors.append(
            f"{label}: header mismatch\n"
            f"  reference: {ref_cols}\n"
            f"  generated: {gen_cols}"
        )
        return errors

    if len(ref_rows) != len(gen_rows):
        errors.append(
            f"{label}: row-count mismatch "
            f"(reference={len(ref_rows)}, generated={len(gen_rows)})"
        )
        return errors

    for row_idx, (ref_row, gen_row) in enumerate(zip(ref_rows, gen_rows), start=2):
        for col in ref_cols:
            expected = ref_row[col]
            actual = gen_row[col]
            if _values_match(expected, actual, rtol=rtol, atol=atol):
                continue
            errors.append(
                f"{label}: row {row_idx}, column {col!r}: "
                f"reference={expected!r}, generated={actual!r}"
            )
            if len(errors) >= max_errors:
                errors.append(f"{label}: stopped after {max_errors} mismatches")
                return errors

    return errors


def _selected_reference_files(reference_dir: Path, names: list[str] | None) -> list[Path]:
    if names:
        return [reference_dir / name for name in names]

    files: list[Path] = []
    for subdir in DEFAULT_REFERENCE_SUBDIRS:
        stage_dir = reference_dir / subdir
        if stage_dir.is_dir():
            files.extend(stage_dir.rglob("*.csv"))
    return sorted(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare regenerated data/processed CSVs with the main "
            "percolation/conductivity/GF reference tables."
        )
    )
    parser.add_argument("--reference-dir", default="data/reference")
    parser.add_argument("--generated-dir", default="data/processed")
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        help=(
            "Reference CSV path relative to --reference-dir. May be repeated. "
            "Defaults to the main percolation/, conductivity/, and gf/ "
            "reference CSVs."
        ),
    )
    parser.add_argument(
        "--ignore-column",
        action="append",
        default=list(DEFAULT_IGNORED_COLUMNS),
        help="Column to ignore during comparison. May be repeated. Default: elapsed_s.",
    )
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip reference files that do not have a same-named generated CSV.",
    )
    parser.add_argument("--max-errors", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference_dir = Path(args.reference_dir)
    generated_dir = Path(args.generated_dir)
    ignored_columns = set(args.ignore_column or [])

    if not reference_dir.is_dir():
        print(f"Reference directory not found: {reference_dir}", file=sys.stderr)
        return 2
    if not generated_dir.is_dir():
        print(f"Generated directory not found: {generated_dir}", file=sys.stderr)
        return 2

    all_errors: list[str] = []
    compared = 0
    skipped = 0

    for reference_path in _selected_reference_files(reference_dir, args.files):
        if not reference_path.is_file():
            all_errors.append(f"Reference file not found: {reference_path}")
            continue

        relative_path = reference_path.relative_to(reference_dir)
        label = relative_path.as_posix()
        generated_path = generated_dir / relative_path
        if not generated_path.is_file():
            msg = f"{label}: generated file missing"
            if args.allow_missing:
                print(f"SKIP {msg}")
                skipped += 1
                continue
            all_errors.append(msg)
            continue

        errors = _compare_file(
            reference_path,
            generated_path,
            label,
            ignored_columns=ignored_columns,
            rtol=args.rtol,
            atol=args.atol,
            max_errors=args.max_errors,
        )
        if errors:
            all_errors.extend(errors)
        else:
            print(f"OK   {label}")
            compared += 1

    if all_errors:
        print("\nReference-output verification failed:", file=sys.stderr)
        for err in all_errors:
            print(f"- {err}", file=sys.stderr)
        return 1

    print(f"\nCompared {compared} file(s); skipped {skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
