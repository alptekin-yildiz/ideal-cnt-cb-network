"""Focused tests for deterministic kernel helpers.

These tests complement the end-to-end smoke tests by pinning small pieces of
core behavior that should not change during documentation, comment, or output
cleanup.
"""

from __future__ import annotations

import numpy as np
import pytest

from cntcb.kernel.materials import carbon_black, carbon_nanotube
from cntcb.kernel.percolation_finder import (
    FACE_X_HIGH,
    FACE_X_LOW,
    FACE_Y_HIGH,
    FACE_Y_LOW,
    PercolationResult,
    _UnionFind,
    _is_percolating,
    _phi_c_theory,
    _sphere_flags,
)


def test_material_factories_keep_expected_defaults() -> None:
    cb = carbon_black()
    cnt = carbon_nanotube()

    assert cb.name == "Carbon Black"
    assert cb.filler_type == "CB"
    assert cb.diameter_nm == pytest.approx(148.0)
    assert cb.length_nm == pytest.approx(cb.diameter_nm)
    assert cb.aspect_ratio == pytest.approx(1.0)
    assert cb.n_segments == 1

    assert cnt.name == "Carbon Nanotube"
    assert cnt.filler_type == "CNT"
    assert cnt.diameter_nm == pytest.approx(10.0)
    assert cnt.length_nm == pytest.approx(5000.0)
    assert cnt.aspect_ratio == pytest.approx(500.0)
    assert cnt.waviness == pytest.approx(0.7)
    assert cnt.n_segments == 10


def test_tunnel_corrected_theory_matches_reference_values() -> None:
    cb = carbon_black(diameter_nm=100.0)
    cnt = carbon_nanotube(
        diameter_nm=10.0,
        length_um=1.0,
        waviness=1.0,
        n_segments=1,
    )

    assert _phi_c_theory(cb, tunnel_cutoff_nm=0.0) == pytest.approx(0.3418)
    assert _phi_c_theory(cb, tunnel_cutoff_nm=5.0) == pytest.approx(
        0.3418 * (100.0 / 105.0) ** 3
    )
    assert _phi_c_theory(cnt, tunnel_cutoff_nm=0.0) == pytest.approx(0.007)
    assert _phi_c_theory(cnt, tunnel_cutoff_nm=5.0) == pytest.approx(
        0.007 * (10.0 / 15.0)
    )


def test_boundary_flags_and_union_find_detect_opposite_faces() -> None:
    low_x = _sphere_flags(np.array([1.0, 50.0, 50.0]), margin=5.0, L=100.0)
    high_x = _sphere_flags(np.array([99.0, 50.0, 50.0]), margin=5.0, L=100.0)
    high_y = _sphere_flags(np.array([50.0, 99.0, 50.0]), margin=5.0, L=100.0)

    assert low_x == FACE_X_LOW
    assert high_x == FACE_X_HIGH
    assert high_y == FACE_Y_HIGH
    assert _is_percolating(low_x | high_x)
    assert not _is_percolating(low_x | high_y)

    uf = _UnionFind(2)
    uf.set_boundary(0, FACE_X_LOW)
    uf.set_boundary(1, FACE_X_HIGH)
    assert uf.union(0, 1)


def test_boundary_flags_are_strict_at_exact_margin() -> None:
    flags = _sphere_flags(np.array([5.0, 5.0, 95.0]), margin=5.0, L=100.0)

    assert flags == 0
    assert _sphere_flags(np.array([4.999, 50.0, 50.0]), 5.0, 100.0) == FACE_X_LOW
    assert (
        _sphere_flags(np.array([50.0, 4.999, 50.0]), 5.0, 100.0)
        == FACE_Y_LOW
    )


def test_percolation_result_summary_success_and_failure_text() -> None:
    success = PercolationResult(
        system="CB",
        phi_c_mean=0.12,
        phi_c_std=0.01,
        phi_c_analytical=0.10,
        n_realizations=5,
        n_failed=1,
        rve_size_nm=1000.0,
        tunnel_cutoff_nm=5.0,
    )
    failed = PercolationResult(
        system="CNT",
        phi_c_mean=float("nan"),
        phi_c_std=float("nan"),
        phi_c_analytical=0.02,
        n_realizations=0,
        n_failed=3,
        rve_size_nm=1000.0,
        tunnel_cutoff_nm=5.0,
    )

    success_summary = success.summary()
    assert "[CB]" in success_summary
    assert "phi_c (MC) = 0.1200 +/- 0.0100" in success_summary
    assert "phi_c (Analytical) = 0.1000" in success_summary
    assert "Deviation = +20.0%" in success_summary
    assert "(N=5, failed=1)" in success_summary

    failed_summary = failed.summary()
    assert "[CNT] phi_c = NaN" in failed_summary
    assert "all realizations failed" in failed_summary
    assert "Analytical: 0.0200" in failed_summary
