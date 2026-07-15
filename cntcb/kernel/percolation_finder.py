"""
Percolation threshold finder for 3D continuum Monte Carlo models.

Supported systems
-----------------
  CB     : isolated spherical particles
  CNT    : wavy nanotubes using a freely rotating chain (FRC) model
  Hybrid : CNTs added on top of a fixed phi_CB carbon-black background

Connectivity rule
-----------------
  Two particles are connected when their surface-to-surface distance is below
  tunnel_cutoff_nm:
    CB-CB  : center-center distance < d_CB + tunnel_cutoff
    CNT-CNT: segment-axis distance < 2*r_CNT + tunnel_cutoff
    CNT-CB : segment-axis to sphere-center distance < r_CNT + r_CB + tunnel_cutoff

Percolation criterion
---------------------
  A connected cluster percolates as soon as it touches opposite RVE faces along
  x, y, or z. A 6-bit boundary flag gives O(1) detection.

Algorithmic lineage
-------------------
  The boundary-flag cluster tracking generalizes the 2D direction-cut
  percolation bookkeeping of:

  Yıldız, A. (2020). Direction-Cut Method and Two-Dimensional Bond
  Percolation in Basic Archimedean Lattices Addressed on a Square Grid
  [Turkish original]. Avrupa Bilim ve Teknoloji Dergisi, (18), 515-530.

Performance features
--------------------
  - Segment-level spatial hash for O(N) neighbor detection instead of O(N^2)
  - Vectorized distance kernels (_batch_seg_dist, _batch_point_seg_dist)
  - Lazy CNT generation using flat NumPy segment arrays
  - Path-halving Union-Find

Low-level entry points
----------------------
  find_threshold_cb(filler, ...)             -> PercolationResult
  find_threshold_cnt(filler, ...)            -> PercolationResult
  find_threshold_cnt_in_hybrid(cb, cnt, ...) -> PercolationResult

These functions are kept as deterministic kernel utilities and monodisperse
percolation checks. Moment-matched polydisperse threshold calculations used by
the manuscript and the public CLI live in cntcb.engines.percolation_engine and
generate/percolation_threshold.py.
"""

from dataclasses import dataclass
import time
from typing import Optional

import numpy as np

from cntcb.kernel.materials import FillerMaterial


# ============================================================================
# Spatial hash: O(1) neighbor queries
# ============================================================================

class _SpatialHash:
    """
    3D grid-based spatial hash that reduces neighbor detection from O(N^2) to O(N).

    The cell_size choice is critical:
      - Segment-segment: cell_size = seg_len + connect_d
        (two segment midpoints can be this far apart and still connect)
      - Sphere-segment:  cell_size = seg_len + connect_cnt_cb
        (maximum distance between a segment midpoint and a sphere center)
      - Sphere-sphere:   cell_size = connect_cb_cb

    A 27-cell neighborhood query (3x3x3) guarantees discovery of all particles
    within a radius of one cell_size.
    """

    __slots__ = ('_cs', '_grid')

    def __init__(self, cell_size: float) -> None:
        self._cs   = cell_size
        self._grid: dict = {}

    def _key(self, pos: np.ndarray) -> tuple:
        cs = self._cs
        return (int(pos[0] / cs), int(pos[1] / cs), int(pos[2] / cs))

    def insert(self, idx: int, pos: np.ndarray) -> None:
        k = self._key(pos)
        if k not in self._grid:
            self._grid[k] = []
        self._grid[k].append(idx)

    def query(self, pos: np.ndarray) -> list:
        """Return all indices in the 27 neighboring cells."""
        cx, cy, cz = self._key(pos)
        result = []
        grid = self._grid
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    bucket = grid.get((cx + dx, cy + dy, cz + dz))
                    if bucket:
                        result.extend(bucket)
        return result


# ============================================================================
# Vectorized helpers
# ============================================================================

def _batch_seg_dist(
    s1: np.ndarray, e1: np.ndarray,
    starts2: np.ndarray, ends2: np.ndarray,
) -> np.ndarray:
    """
    Vectorized axis-to-axis distances between one segment and N segments.

    This is a NumPy adaptation of the Lumelsky (1985) algorithm. It replaces N
    Python loops with a single matrix operation, typically 100-1000x faster.

    Parameters
    ----------
    s1, e1          : (3,)    query segment endpoints
    starts2, ends2  : (N, 3) candidate segment endpoints

    Returns
    -------
    (N,) minimum axis distances from s1->e1 to each candidate segment
    """
    u  = e1 - s1                               # (3,)
    v  = ends2 - starts2                       # (N, 3)
    w0 = s1 - starts2                          # (N, 3)

    # np.errstate: macOS Accelerate BLAS can report spurious FP exceptions
    # during large matmul operations. The results are correct, so silence them.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        a  = float(np.dot(u, u))               # scalar
        b  = v @ u                             # (N,)
        c  = np.einsum('ij,ij->i', v, v)       # (N,)
        d  = w0 @ u                            # (N,)
        e  = np.einsum('ij,ij->i', w0, v)      # (N,)

    denom = a * c - b * b                      # (N,)
    par   = np.abs(denom) < 1e-8 * (a * c + 1e-30)

    # Free minimizer, then clamp.
    s = np.clip(
        np.where(par, 0.0, (b * e - c * d) / np.where(par, 1.0, denom)),
        0.0, 1.0,
    )
    # Optimal t for clamped s, then clamp.
    t = np.clip((e + s * b) / np.where(c > 1e-30, c, 1.0), 0.0, 1.0)
    # Optimal s for clamped t, then clamp; second pass handles corner cases.
    s = np.clip((t * b - d) / (a + 1e-30), 0.0, 1.0)

    p1 = s1      + s[:, np.newaxis] * u    # (N, 3)
    p2 = starts2 + t[:, np.newaxis] * v   # (N, 3)
    return np.linalg.norm(p1 - p2, axis=1) # (N,)


def _batch_point_seg_dist(
    points:  np.ndarray,
    s1:      np.ndarray,
    e1:      np.ndarray,
) -> np.ndarray:
    """
    Compute axis distances from N points to one segment (s1->e1).

    points : (N, 3)
    s1, e1 : (3,)
    Returns : (N,) distance from each point to the segment axis
    """
    v     = e1 - s1                                          # (3,)
    w     = points - s1                                      # (N, 3)

    # np.errstate: macOS Accelerate BLAS can report spurious FP warnings
    # during vectorized matmul, as in _batch_seg_dist above. The magnitudes
    # here are nm-scale and the denominator is explicitly guarded.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        denom = float(np.dot(v, v))
        t     = np.clip(w @ v / (denom + 1e-30), 0.0, 1.0)  # (N,)

    closest = s1 + t[:, np.newaxis] * v                      # (N, 3)
    return np.linalg.norm(points - closest, axis=1)          # (N,)


# ============================================================================
# Tunnel-corrected theoretical percolation threshold
# ============================================================================

def _phi_c_theory(filler: FillerMaterial, tunnel_cutoff_nm: float) -> float:
    """
    Theoretical percolation threshold corrected for the tunneling cutoff.

    CB (soft/penetrable sphere):
        eta_c * V_eff ~= 0.3418 -> phi_c = 0.3418 * (d / (d + t))^3
        Sources: Pike & Seager (1974); Rintoul & Torquato (1997)

    CNT (excluded volume, tunnel-corrected):
        When the connection distance is d_eff = d + t,
        excluded volume scales as L^2 * d_eff, so
        phi_c = 0.7/(AR*w) * d/d_eff.
        AR*w is a heuristic effective-aspect-ratio guide (w is the
        bend-angle cosine of the generated chain), not a derived
        formula for the realized geometry.
        Sources: Balberg et al. (1984); Bauhofer & Kovacs (2009)
    """
    if filler.filler_type == "CB":
        d     = filler.diameter_nm
        d_eff = d + tunnel_cutoff_nm
        return 0.3418 * (d / d_eff) ** 3

    elif filler.filler_type == "CNT":
        d      = filler.diameter_nm
        d_eff  = d + tunnel_cutoff_nm
        eff_ar = max(filler.aspect_ratio * filler.waviness, 1.0)
        return (0.7 / eff_ar) * (d / d_eff)

    else:
        raise ValueError(f"Unknown filler type: {filler.filler_type}")


# ============================================================================
# 6-face boundary flags for percolation detection. These are the 3D RVE
# analogue of the edge-touch flags used in the 2D direction-cut work.
# ============================================================================

FACE_X_LOW  = 0b000001   # x < margin  (left wall)
FACE_X_HIGH = 0b000010   # x > L−margin (right wall)
FACE_Y_LOW  = 0b000100   # y < margin
FACE_Y_HIGH = 0b001000   # y > L−margin
FACE_Z_LOW  = 0b010000   # z < margin
FACE_Z_HIGH = 0b100000   # z > L−margin

_PERC_X = FACE_X_LOW | FACE_X_HIGH
_PERC_Y = FACE_Y_LOW | FACE_Y_HIGH
_PERC_Z = FACE_Z_LOW | FACE_Z_HIGH


def _is_percolating(flags: int) -> bool:
    """Detect percolation from a 6-bit boundary flag."""
    return (
        (flags & _PERC_X) == _PERC_X or
        (flags & _PERC_Y) == _PERC_Y or
        (flags & _PERC_Z) == _PERC_Z
    )


# ============================================================================
# Union-Find with path halving and 6-face boundary tracking. This is the
# disjoint-set form of the reference-number cluster bookkeeping in the 2D work.
# ============================================================================

class _UnionFind:
    """
    Path-halving Union-Find.

    Each cluster root stores a 6-bit `boundary` flag indicating which RVE faces
    the cluster touches. Boundary flags are merged after union, allowing
    immediate percolation detection.
    """

    __slots__ = ('_parent', '_rank', '_boundary')

    def __init__(self, n: int) -> None:
        self._parent   = list(range(n))
        self._rank     = [0] * n
        self._boundary = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]   # path halving
            x = self._parent[x]
        return x

    def set_boundary(self, node: int, flags: int) -> None:
        """Add boundary flags to the node's cluster."""
        root = self.find(node)
        self._boundary[root] |= flags

    def union(self, a: int, b: int) -> bool:
        """
        Union a and b.
        Return True if the resulting cluster percolates.
        """
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return _is_percolating(self._boundary[ra])

        # Union by rank.
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb]   = ra
        self._boundary[ra] |= self._boundary[rb]
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

        return _is_percolating(self._boundary[ra])

    def is_root_percolating(self, node: int) -> bool:
        return _is_percolating(self._boundary[self.find(node)])


# ============================================================================
# Boundary-flag calculators
# ============================================================================

def _sphere_flags(center: np.ndarray, margin: float, L: float) -> int:
    """
    Compute the 6-bit RVE boundary flag from a sphere-center position.

    margin : center-to-face distance threshold for contact with an RVE face.
             Use r_CB for CB geometric contact and r_CNT for CNTs.
    """
    flags = 0
    if center[0] < margin:       flags |= FACE_X_LOW
    if center[0] > L - margin:   flags |= FACE_X_HIGH
    if center[1] < margin:       flags |= FACE_Y_LOW
    if center[1] > L - margin:   flags |= FACE_Y_HIGH
    if center[2] < margin:       flags |= FACE_Z_LOW
    if center[2] > L - margin:   flags |= FACE_Z_HIGH
    return flags


# ============================================================================
# Particle generation
# ============================================================================

def _gen_cb_centers(n: int, L: float, rng: np.random.Generator) -> np.ndarray:
    """Generate n CB centers uniformly in [0, L]^3. Returns an (n, 3) array."""
    return rng.uniform(0.0, L, size=(n, 3))


def _lognormal_params_from_mean_std(mu: float, sigma: float) -> tuple[float, float]:
    """
    Compute log-normal parameters (mu_ln, sigma_ln) from linear-space mean
    and standard deviation.
    """
    cv2      = (sigma / mu) ** 2
    sigma_ln = float(np.sqrt(np.log(1.0 + cv2)))
    mu_ln    = float(np.log(mu) - 0.5 * sigma_ln ** 2)
    return mu_ln, sigma_ln


def _sample_cb_radii(
    mu_nm:    float,
    sigma_nm: float,
    n:        int,
    rng:      np.random.Generator,
    d_min:    float = 20.0,
    d_max:    float = 500.0,
) -> np.ndarray:
    """
    Sample n radii from a truncated log-normal CB aggregate diameter distribution.

    Parameters
    ------------
    mu_nm    : target mean diameter [nm]
    sigma_nm : target diameter standard deviation [nm]
    n        : number of particles to sample
    rng      : NumPy random number generator
    d_min    : lower diameter bound [nm], default 20 nm primary-particle floor
    d_max    : upper diameter bound [nm], default 500 nm large-aggregate ceiling

    Returns
    --------
    (n,) ndarray of radii [nm], equal to diameter / 2
    """
    mu_ln, sigma_ln = _lognormal_params_from_mean_std(mu_nm, sigma_nm)
    result = []
    batch_size = max(n * 3, 200)
    while len(result) < n:
        diams = np.exp(rng.normal(mu_ln, sigma_ln, size=batch_size))
        diams = diams[(diams >= d_min) & (diams <= d_max)]
        result.extend(diams.tolist())
    return np.asarray(result[:n]) / 2.0  # diameter -> radius


def _sample_cnt_lengths(
    mu_nm:    float,
    sigma_nm: float,
    n:        int,
    rng:      np.random.Generator,
    l_min:    float = 50.0,
    l_max:    float = 1500.0,
) -> np.ndarray:
    """
    Sample n CNT lengths from a truncated log-normal distribution [nm].
    """
    mu_ln, sigma_ln = _lognormal_params_from_mean_std(mu_nm, sigma_nm)
    result = []
    batch_size = max(n * 3, 200)
    while len(result) < n:
        lengths = np.exp(rng.normal(mu_ln, sigma_ln, size=batch_size))
        lengths = lengths[(lengths >= l_min) & (lengths <= l_max)]
        result.extend(lengths.tolist())
    return np.asarray(result[:n])


# ============================================================================
# Single realization: CB
# ============================================================================

def _run_cb(
    filler: FillerMaterial,
    L: float,
    tunnel_cutoff_nm: float,
    n_max: int,
    rng: np.random.Generator,
) -> float:
    """
    Run one Monte Carlo realization for CB.

    Connectivity: center-center distance < diameter + tunnel_cutoff, equivalent
    to surface-to-surface distance < tunnel_cutoff.
    Boundary: a sphere center is within radius of a wall, i.e. geometric contact.

    Returns
    --------
    float
        phi_c, the percolation volume fraction. Returns np.nan if no
        percolation occurs.
    """
    r          = filler.diameter_nm / 2.0
    margin     = r                                 # geometric wall contact
    connect_d  = filler.diameter_nm + tunnel_cutoff_nm
    V_sphere   = (4.0 / 3.0) * np.pi * r ** 3
    V_rve      = L ** 3

    centers = _gen_cb_centers(n_max, L, rng)
    order   = rng.permutation(n_max)

    uf    = _UnionFind(n_max)
    shash = _SpatialHash(cell_size=connect_d)  # O(N) neighbor queries

    for step in range(n_max):
        i      = order[step]
        center = centers[i]

        flags = _sphere_flags(center, margin, L)
        uf.set_boundary(i, flags)

        percolated = uf.is_root_percolating(i)

        for j in shash.query(center):
            dist = np.linalg.norm(centers[j] - center)
            if dist < connect_d:
                if uf.union(i, j):
                    percolated = True

        shash.insert(i, center)

        if percolated:
            return (step + 1) * V_sphere / V_rve

    return np.nan


# ============================================================================
# Single realization: CNT
# ============================================================================

def _run_cnt(
    filler: FillerMaterial,
    L: float,
    tunnel_cutoff_nm: float,
    n_max: int,
    rng: np.random.Generator,
) -> float:
    """
    Run one Monte Carlo realization for CNT.

    Connectivity: segment-segment axis distance < tunnel_cutoff + 2*r_cnt.
    _batch_seg_dist is vectorized at every step; n_seg=1 (straight) and
    n_seg>1 (wavy) both use the same path.
    """
    r_cnt  = filler.diameter_nm / 2.0
    # Approximate CNT volume, cylindrical and ignoring waviness.
    V_cnt  = np.pi * r_cnt ** 2 * filler.length_nm
    V_rve  = L ** 3

    n_seg     = filler.n_segments
    connect_d = tunnel_cutoff_nm + 2 * r_cnt   # axis-to-axis threshold

    # CNT generation constants, precomputed to avoid per-loop recomputation.
    seg_len = filler.length_nm / n_seg
    cos_b   = filler.waviness                        # cos(bend) = waviness
    sin_b   = np.sqrt(max(0.0, 1.0 - cos_b ** 2))   # sin(bend)

    uf = _UnionFind(n_max)

    # Segment-level spatial hash: cell size = seg_len + connect_d.
    # This gives O(N) behavior for near segment pairs instead of O(N^2).
    seg_shash = _SpatialHash(cell_size=seg_len + connect_d)

    # Preallocated segment arrays.
    act_starts  = np.zeros((n_max * n_seg, 3))
    act_ends    = np.zeros((n_max * n_seg, 3))
    act_ids_arr = np.empty(n_max, dtype=np.intp)
    n_act = 0

    # Temporary CNT segment arrays, reused at every step.
    segs_s = np.empty((n_seg, 3))   # segment start points
    segs_e = np.empty((n_seg, 3))   # segment end points
    mids   = np.empty((n_seg, 3))   # midpoints for hash queries
    _seg_range = np.arange(n_seg)   # [0, 1, ..., n_seg-1] cache

    for step in range(n_max):
        # Lazy generation of one CNT directly into flat segment arrays.
        pos   = rng.uniform(0.0, L, size=3)
        d_vec = rng.standard_normal(3)
        d_vec /= np.linalg.norm(d_vec)

        for s in range(n_seg):
            segs_s[s] = pos
            segs_e[s] = pos + d_vec * seg_len
            pos = segs_e[s]
            if s < n_seg - 1:
                perturb = rng.standard_normal(3)
                perturb -= np.dot(perturb, d_vec) * d_vec
                norm_p   = np.linalg.norm(perturb)
                if norm_p > 1e-12:
                    perturb /= norm_p
                    d_vec    = cos_b * d_vec + sin_b * perturb
                    d_vec   /= np.linalg.norm(d_vec)

        np.add(segs_s, segs_e, out=mids)
        mids *= 0.5   # midpoints, in place

        # 6-face boundary flags (NumPy).
        pts   = np.vstack((segs_s, segs_e))   # (2*n_seg, 3)
        flags = 0
        if np.any(pts[:, 0] < r_cnt):        flags |= FACE_X_LOW
        if np.any(pts[:, 0] > L - r_cnt):    flags |= FACE_X_HIGH
        if np.any(pts[:, 1] < r_cnt):        flags |= FACE_Y_LOW
        if np.any(pts[:, 1] > L - r_cnt):    flags |= FACE_Y_HIGH
        if np.any(pts[:, 2] < r_cnt):        flags |= FACE_Z_LOW
        if np.any(pts[:, 2] > L - r_cnt):    flags |= FACE_Z_HIGH

        uf.set_boundary(step, flags)
        percolated = uf.is_root_percolating(step)

        if n_act > 0:
            # Spatial-hash query for nearby CNT candidates.
            # Segments are stored as flat_id = cnt_idx * n_seg + seg_no.
            # flat_id // n_seg gives the owning CNT.
            cand_set = set()
            for s in range(n_seg):
                for flat_id in seg_shash.query(mids[s]):
                    cand_set.add(flat_id // n_seg)

            if cand_set:
                # Gather candidate CNT segment arrays using NumPy fancy indexing.
                cand_arr = np.fromiter(cand_set, dtype=np.intp)  # (n_cand,)
                seg_idx  = (cand_arr[:, None] * n_seg + _seg_range[None, :]).ravel()
                c_starts = act_starts[seg_idx]   # (n_cand*n_seg, 3)
                c_ends   = act_ends[seg_idx]
                n_cand   = len(cand_arr)

                connected = np.zeros(n_cand, dtype=bool)
                for s in range(n_seg):
                    dists = _batch_seg_dist(
                        segs_s[s], segs_e[s], c_starts, c_ends,
                    )   # (n_cand * n_seg,)
                    connected |= dists.reshape(n_cand, n_seg).min(axis=1) < connect_d
                for k in np.flatnonzero(connected):
                    if uf.union(step, act_ids_arr[cand_arr[k]]):
                        percolated = True

        # Store this CNT's segments using bulk assignment.
        base = n_act * n_seg
        act_starts[base:base + n_seg] = segs_s
        act_ends[base:base + n_seg]   = segs_e
        act_ids_arr[n_act] = step

        # Insert segments into the spatial hash.
        for s in range(n_seg):
            seg_shash.insert(n_act * n_seg + s, mids[s])

        n_act += 1

        if percolated:
            return (step + 1) * V_cnt / V_rve

    return np.nan


# ============================================================================
# Single realization: Hybrid v2, fixed phi_CB background and CNT sweep
# ============================================================================

def _sample_vmf_x(kappa: float, rng: np.random.Generator) -> np.ndarray:
    """
    Sample a unit vector from a von Mises-Fisher distribution biased toward x.

    kappa = 0 -> isotropic distribution on the unit sphere
    kappa -> infinity -> fully aligned with +x

    Algorithm: exact inverse-CDF method
      z = (1/kappa) * log((1-u)*exp(-kappa) + u*exp(kappa)), u ~ Uniform[0,1]
      phi ~ Uniform[0, 2*pi]
      d = [z, sqrt(1-z^2)*cos(phi), sqrt(1-z^2)*sin(phi)]
    """
    if kappa < 1e-6:
        d = rng.standard_normal(3)
        return d / np.linalg.norm(d)
    u   = rng.random()
    # logaddexp computes log(exp(a) + exp(b)) without overflow.
    a   = np.log(max(1.0 - u, 1e-300)) - kappa   # log((1-u)·e^(-κ))
    b   = np.log(max(u,       1e-300)) + kappa    # log(u·e^(κ))
    z   = np.clip(np.logaddexp(a, b) / kappa, -1.0, 1.0)
    phi = 2.0 * np.pi * rng.random()
    s   = np.sqrt(max(0.0, 1.0 - z * z))
    return np.array([z, s * np.cos(phi), s * np.sin(phi)])


def _run_hybrid_cnt_sweep(
    filler_cb:        FillerMaterial,
    filler_cnt:       FillerMaterial,
    n_cb:             int,
    n_cnt_max:        int,
    L:                float,
    tunnel_cutoff_nm: float,
    rng:              np.random.Generator,
    kappa:            float = 0.0,
    r_cb_arr:         Optional[np.ndarray] = None,
    cnt_lengths_arr:  Optional[np.ndarray] = None,
) -> float:
    """
    Add CNTs one by one on a fixed N_CB carbon-black background and find
    the CNT percolation threshold phi_c_CNT.

    UnionFind index space:
        [0 .. n_cb)              -> CB particles
        [n_cb .. n_cb+n_cnt_max) -> CNT particles

    r_cb_arr : (n_cb,) array of polydisperse CB radii [nm].
               If None, filler_cb.diameter_nm / 2 is used (monodisperse).

    Returns
    --------
    float
        CNT volume fraction at percolation, or np.nan if percolation fails.
    """
    # CB radii: monodisperse or polydisperse.
    if n_cb > 0:
        if r_cb_arr is None:
            r_cb_arr = np.full(n_cb, filler_cb.diameter_nm / 2.0)
        r_max_cb = float(r_cb_arr.max())
    else:
        r_cb_arr = np.empty(0)
        r_max_cb = filler_cb.diameter_nm / 2.0

    r_cnt = filler_cnt.diameter_nm / 2.0
    V_rve = L**3

    n_seg           = filler_cnt.n_segments
    # CB-CB: maximum pair distance for the spatial hash.
    connect_cb_cb_max  = 2.0 * r_max_cb + tunnel_cutoff_nm
    connect_cnt_cnt    = tunnel_cutoff_nm + 2.0 * r_cnt
    # CNT-CB: base component, with per-CB r_cb_arr[i] added later.
    connect_cnt_cb_base = tunnel_cutoff_nm + r_cnt

    # Polydisperse case: each CNT has its own length; the hash needs max seg_len.
    if cnt_lengths_arr is not None:
        seg_len_max = float(np.max(cnt_lengths_arr)) / n_seg
    else:
        seg_len_max = filler_cnt.length_nm / n_seg
    cos_b   = filler_cnt.waviness
    sin_b   = np.sqrt(max(0.0, 1.0 - cos_b**2))

    # UnionFind: CB + CNT
    uf = _UnionFind(n_cb + n_cnt_max)

    # CB background.
    cb_centers = rng.uniform(0.0, L, size=(n_cb, 3))
    cb_shash   = _SpatialHash(cell_size=connect_cb_cb_max)
    for i in range(n_cb):
        flags = _sphere_flags(cb_centers[i], r_cb_arr[i], L)
        uf.set_boundary(i, flags)
        for j in cb_shash.query(cb_centers[i]):
            connect_ij = r_cb_arr[i] + r_cb_arr[j] + tunnel_cutoff_nm
            if np.linalg.norm(cb_centers[j] - cb_centers[i]) < connect_ij:
                uf.union(i, j)
        cb_shash.insert(i, cb_centers[i])

    # CB spatial hash for CNT-CB queries: segment midpoint -> nearby CBs.
    cb_seg_shash = _SpatialHash(cell_size=seg_len_max + connect_cnt_cb_base + r_max_cb)
    for i in range(n_cb):
        cb_seg_shash.insert(i, cb_centers[i])

    # Incremental CNT insertion.
    seg_shash   = _SpatialHash(cell_size=seg_len_max + connect_cnt_cnt)
    act_starts  = np.zeros((n_cnt_max * n_seg, 3))
    act_ends    = np.zeros((n_cnt_max * n_seg, 3))
    act_ids_arr = np.empty(n_cnt_max, dtype=np.intp)
    n_act       = 0

    segs_s     = np.empty((n_seg, 3))
    segs_e     = np.empty((n_seg, 3))
    mids       = np.empty((n_seg, 3))
    _seg_range = np.arange(n_seg)
    total_vol_cnt = 0.0

    for step in range(n_cnt_max):
        cnt_uf = n_cb + step   # UnionFind index for this CNT

        # Per-CNT length: polydisperse or fixed.
        if cnt_lengths_arr is not None:
            seg_len_i = cnt_lengths_arr[step] / n_seg
            v_i = np.pi * r_cnt**2 * cnt_lengths_arr[step]
        else:
            seg_len_i = filler_cnt.length_nm / n_seg
            v_i = np.pi * r_cnt**2 * filler_cnt.length_nm

        # Lazy CNT generation.
        pos   = rng.uniform(0.0, L, size=3)
        d_vec = _sample_vmf_x(kappa, rng)
        for s in range(n_seg):
            segs_s[s] = pos
            segs_e[s] = pos + d_vec * seg_len_i
            pos = segs_e[s]
            if s < n_seg - 1:
                perturb = rng.standard_normal(3)
                perturb -= np.dot(perturb, d_vec) * d_vec
                norm_p   = np.linalg.norm(perturb)
                if norm_p > 1e-12:
                    perturb /= norm_p
                    d_vec    = cos_b * d_vec + sin_b * perturb
                    d_vec   /= np.linalg.norm(d_vec)

        np.add(segs_s, segs_e, out=mids)
        mids *= 0.5

        # Boundary flags.
        pts   = np.vstack((segs_s, segs_e))
        flags = 0
        if np.any(pts[:, 0] < r_cnt):      flags |= FACE_X_LOW
        if np.any(pts[:, 0] > L - r_cnt):  flags |= FACE_X_HIGH
        if np.any(pts[:, 1] < r_cnt):      flags |= FACE_Y_LOW
        if np.any(pts[:, 1] > L - r_cnt):  flags |= FACE_Y_HIGH
        if np.any(pts[:, 2] < r_cnt):      flags |= FACE_Z_LOW
        if np.any(pts[:, 2] > L - r_cnt):  flags |= FACE_Z_HIGH

        uf.set_boundary(cnt_uf, flags)
        percolated = uf.is_root_percolating(cnt_uf)

        # CNT-CNT connectivity using the segment spatial hash.
        if n_act > 0:
            cand_set = set()
            for s in range(n_seg):
                for flat_id in seg_shash.query(mids[s]):
                    cand_set.add(flat_id // n_seg)
            if cand_set:
                cand_arr = np.fromiter(cand_set, dtype=np.intp)
                seg_idx  = (cand_arr[:, None] * n_seg + _seg_range[None, :]).ravel()
                c_starts = act_starts[seg_idx]
                c_ends   = act_ends[seg_idx]
                n_cand   = len(cand_arr)
                connected = np.zeros(n_cand, dtype=bool)
                for s in range(n_seg):
                    dists = _batch_seg_dist(segs_s[s], segs_e[s], c_starts, c_ends)
                    connected |= dists.reshape(n_cand, n_seg).min(axis=1) < connect_cnt_cnt
                for k in np.flatnonzero(connected):
                    if uf.union(cnt_uf, n_cb + act_ids_arr[cand_arr[k]]):
                        percolated = True

        # CNT-CB connectivity using the CB segment spatial hash.
        if n_cb > 0:
            cb_cand_set = set()
            for s in range(n_seg):
                for cb_idx in cb_seg_shash.query(mids[s]):
                    cb_cand_set.add(cb_idx)
            if cb_cand_set:
                cb_cand_arr     = np.fromiter(cb_cand_set, dtype=np.intp)
                cb_cand_centers = cb_centers[cb_cand_arr]   # (n_cand_cb, 3)
                # Polydisperse case: separate connection threshold for each CB.
                connect_thresh  = connect_cnt_cb_base + r_cb_arr[cb_cand_arr]
                connected_cb    = np.zeros(len(cb_cand_arr), dtype=bool)
                for s in range(n_seg):
                    dists = _batch_point_seg_dist(cb_cand_centers, segs_s[s], segs_e[s])
                    connected_cb |= dists < connect_thresh
                for k in np.flatnonzero(connected_cb):
                    if uf.union(cnt_uf, cb_cand_arr[k]):
                        percolated = True

        # Store segments.
        base = n_act * n_seg
        act_starts[base:base + n_seg] = segs_s
        act_ends[base:base + n_seg]   = segs_e
        act_ids_arr[n_act] = step
        for s in range(n_seg):
            seg_shash.insert(n_act * n_seg + s, mids[s])
        n_act += 1

        total_vol_cnt += v_i
        if percolated:
            return total_vol_cnt / V_rve

    return np.nan


# ============================================================================
# Result data structure
# ============================================================================

@dataclass
class PercolationResult:
    """Result from a Monte Carlo percolation-threshold search."""
    system:          str    # "CB", "CNT", "Hybrid"
    phi_c_mean:      float  # Mean critical volume fraction
    phi_c_std:       float  # Standard deviation
    phi_c_analytical: float # Analytical estimate
    n_realizations:  int    # Completed realizations
    n_failed:        int    # Realizations that did not percolate
    rve_size_nm:     float
    tunnel_cutoff_nm: float

    def summary(self) -> str:
        if self.n_realizations == 0:
            return (f"[{self.system}] phi_c = NaN  "
                    f"(all realizations failed; n_max may be too low?)\n"
                    f"  Analytical: {self.phi_c_analytical:.4f}")
        dev = (self.phi_c_mean - self.phi_c_analytical) / self.phi_c_analytical * 100
        return (
            f"[{self.system}]  "
            f"phi_c (MC) = {self.phi_c_mean:.4f} +/- {self.phi_c_std:.4f}   "
            f"phi_c (Analytical) = {self.phi_c_analytical:.4f}   "
            f"Deviation = {dev:+.1f}%   "
            f"(N={self.n_realizations}, failed={self.n_failed})"
        )


def find_threshold_cb(
    filler: FillerMaterial,
    rve_size: float = 1000.0,
    tunnel_cutoff_nm: float = 10.0,
    n_max: int = 5000,
    n_realizations: int = 30,
    seed: Optional[int] = 42,
    verbose: bool = True,
    progress_every: int = 0,
) -> PercolationResult:
    """
    Monte Carlo percolation threshold for isolated spherical CB particles.

    Parameters
    ----------
    filler : FillerMaterial
        CB material properties; diameter_nm is used.
    rve_size : float
        RVE side length [nm]. Recommended: at least 20 * diameter_nm.
    tunnel_cutoff_nm : float
        Tunneling cutoff distance [nm].
    n_max : int
        Maximum particle count per realization. It should be large enough that
        phi_c < n_max * V_p / V_rve.
    n_realizations : int
        Number of Monte Carlo realizations.
    seed : int, optional
        Random seed.
    verbose : bool
        Print progress.
    """
    rng     = np.random.default_rng(seed)
    samples = []
    failed  = 0
    t0 = time.time()

    if verbose:
        print(f"Searching CB percolation threshold "
              f"(d={filler.diameter_nm:.1f} nm, L={rve_size:.0f} nm, "
              f"tunnel={tunnel_cutoff_nm:.1f} nm, N_max={n_max}) ...")

    for rl in range(n_realizations):
        phi_c = _run_cb(filler, rve_size, tunnel_cutoff_nm, n_max, rng)
        if np.isnan(phi_c):
            failed += 1
            if verbose:
                print(f"  [{rl+1}/{n_realizations}] No percolation (n_max too low)")
        else:
            samples.append(phi_c)
            if verbose:
                print(f"  [{rl+1}/{n_realizations}] φ_c = {phi_c:.4f}")
        n_done = rl + 1
        if (
            progress_every > 0
            and n_realizations >= progress_every
            and (n_done % progress_every == 0 or n_done == n_realizations)
        ):
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0.0
            eta = (n_realizations - n_done) / rate if rate > 0 else float("nan")
            print(
                f"    progress {n_done}/{n_realizations}  ok={len(samples)}  "
                f"fail={failed}  elapsed={elapsed:.1f}s  eta={eta:.1f}s",
                flush=True,
            )

    phi_anal = _phi_c_theory(filler, tunnel_cutoff_nm)
    if samples:
        return PercolationResult(
            system="CB",
            phi_c_mean=float(np.mean(samples)),
            phi_c_std=float(np.std(samples, ddof=1) if len(samples) > 1 else 0.0),
            phi_c_analytical=phi_anal,
            n_realizations=len(samples),
            n_failed=failed,
            rve_size_nm=rve_size,
            tunnel_cutoff_nm=tunnel_cutoff_nm,
        )
    return PercolationResult(
        system="CB", phi_c_mean=np.nan, phi_c_std=np.nan,
        phi_c_analytical=phi_anal, n_realizations=0, n_failed=failed,
        rve_size_nm=rve_size, tunnel_cutoff_nm=tunnel_cutoff_nm,
    )


def find_threshold_cnt(
    filler: FillerMaterial,
    rve_size: float = 2000.0,
    tunnel_cutoff_nm: float = 10.0,
    n_max: int = 300,
    n_realizations: int = 20,
    seed: Optional[int] = 42,
    verbose: bool = True,
    progress_every: int = 0,
) -> PercolationResult:
    """
    Monte Carlo percolation threshold for a CNT system.

    CNT-CNT connectivity is computed from vectorized segment-axis distances.
    n_max can be relatively small because CNTs have large excluded volume.
    """
    rng     = np.random.default_rng(seed)
    samples = []
    failed  = 0
    t0 = time.time()

    if verbose:
        print(f"Searching CNT percolation threshold "
              f"(d={filler.diameter_nm:.1f} nm, L={filler.length_nm:.0f} nm, "
              f"waviness={filler.waviness:.2f}, RVE={rve_size:.0f} nm, "
              f"N_max={n_max}) ...")

    for rl in range(n_realizations):
        phi_c = _run_cnt(filler, rve_size, tunnel_cutoff_nm, n_max, rng)
        if np.isnan(phi_c):
            failed += 1
            if verbose:
                print(f"  [{rl+1}/{n_realizations}] No percolation (n_max too low)")
        else:
            samples.append(phi_c)
            if verbose:
                print(f"  [{rl+1}/{n_realizations}] φ_c = {phi_c:.4f}")
        n_done = rl + 1
        if (
            progress_every > 0
            and n_realizations >= progress_every
            and (n_done % progress_every == 0 or n_done == n_realizations)
        ):
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0.0
            eta = (n_realizations - n_done) / rate if rate > 0 else float("nan")
            print(
                f"    progress {n_done}/{n_realizations}  ok={len(samples)}  "
                f"fail={failed}  elapsed={elapsed:.1f}s  eta={eta:.1f}s",
                flush=True,
            )

    phi_anal = _phi_c_theory(filler, tunnel_cutoff_nm)
    if samples:
        return PercolationResult(
            system="CNT",
            phi_c_mean=float(np.mean(samples)),
            phi_c_std=float(np.std(samples, ddof=1) if len(samples) > 1 else 0.0),
            phi_c_analytical=phi_anal,
            n_realizations=len(samples),
            n_failed=failed,
            rve_size_nm=rve_size,
            tunnel_cutoff_nm=tunnel_cutoff_nm,
        )
    return PercolationResult(
        system="CNT", phi_c_mean=np.nan, phi_c_std=np.nan,
        phi_c_analytical=phi_anal, n_realizations=0, n_failed=failed,
        rve_size_nm=rve_size, tunnel_cutoff_nm=tunnel_cutoff_nm,
    )


def find_threshold_cnt_in_hybrid(
    filler_cb:        FillerMaterial,
    filler_cnt:       FillerMaterial,
    phi_cb:           float,
    rve_size:         float = 1500.0,
    tunnel_cutoff_nm: float = 10.0,
    n_cnt_max:        Optional[int] = None,
    n_realizations:   int  = 50,
    seed:             Optional[int] = 42,
    verbose:          bool = True,
    kappa:            float = 0.0,
    cb_sigma_nm:      float = 0.0,
    cnt_sigma_L_nm:   float = 0.0,
    cnt_l_min_nm:     float = 50.0,
    cnt_l_max_nm:     float = 1500.0,
) -> PercolationResult:
    """
    Find phi_c_CNT with a fixed phi_CB background.

    This low-level convenience function predates the moment-matched
    distribution engine. It remains useful for monodisperse and exploratory
    kernel checks. For paper-compatible polydisperse thresholds, use
    generate/percolation_threshold.py or cntcb.engines.percolation_engine.

    Parameters
    ----------
    phi_cb : float
        Fixed CB volume fraction; 0 is equivalent to a pure CNT system.
    n_cnt_max : int, optional
        Maximum CNT count per realization. If None, it is estimated
        automatically from the theoretical threshold with a 3x safety factor.
    cb_sigma_nm : float
        Standard deviation of the CB aggregate diameter distribution [nm].
        0 -> monodisperse (filler_cb.diameter_nm); >0 -> log-normal(mu, sigma).
    cnt_sigma_L_nm : float
        Standard deviation of the CNT length distribution [nm].
        0 -> monodisperse (filler_cnt.length_nm); >0 -> log-normal(mu_L, sigma_L).
    cnt_l_min_nm, cnt_l_max_nm : float
        Lower and upper bounds for the truncated log-normal distribution [nm].
    """
    # n_cb is computed from the mean volume.
    if cb_sigma_nm > 0.0 and phi_cb > 0:
        mu_r, sigma_r = filler_cb.diameter_nm / 2.0, cb_sigma_nm / 2.0
        mu_ln_r, sigma_ln_r = _lognormal_params_from_mean_std(mu_r, sigma_r)
        E_r3 = np.exp(3.0 * mu_ln_r + 4.5 * sigma_ln_r ** 2)
        E_V  = (4.0 / 3.0) * np.pi * E_r3
    else:
        E_V = (4.0 / 3.0) * np.pi * (filler_cb.diameter_nm / 2.0) ** 3
    n_cb = max(0, int(round(phi_cb * rve_size ** 3 / E_V))) if phi_cb > 0 else 0

    if n_cnt_max is None:
        phi_c_est = _phi_c_theory(filler_cnt, tunnel_cutoff_nm)
        V_cnt     = np.pi * (filler_cnt.diameter_nm / 2.0) ** 2 * filler_cnt.length_nm
        n_cnt_max = max(50, int(np.ceil(phi_c_est * rve_size ** 3 / V_cnt * 3.0)))

    rng     = np.random.default_rng(seed)
    samples = []
    failed  = 0

    cb_dist_str  = (f"log-normal(μ={filler_cb.diameter_nm:.0f}, σ={cb_sigma_nm:.0f} nm)"
                    if cb_sigma_nm > 0.0 else f"monodisperse d={filler_cb.diameter_nm:.0f} nm")
    cnt_dist_str = (f"log-normal(μ={filler_cnt.length_nm:.0f}, σ={cnt_sigma_L_nm:.0f} nm)"
                    if cnt_sigma_L_nm > 0.0 else f"monodisperse L={filler_cnt.length_nm:.0f} nm")
    if verbose:
        print(f"Hybrid CNT threshold: phi_CB={phi_cb:.4f}  N_CB={n_cb}  CB: {cb_dist_str}"
              f"  CNT: {cnt_dist_str}  N_CNT_max={n_cnt_max}  RVE={rve_size:.0f} nm ...")

    for rl in range(n_realizations):
        # Independent CB radius sampling for each realization.
        r_cb_arr_rl: Optional[np.ndarray] = None
        if cb_sigma_nm > 0.0 and n_cb > 0:
            r_cb_arr_rl = _sample_cb_radii(
                filler_cb.diameter_nm, cb_sigma_nm, n_cb, rng)
        # Independent CNT length sampling for each realization.
        cnt_lengths_rl: Optional[np.ndarray] = None
        if cnt_sigma_L_nm > 0.0:
            cnt_lengths_rl = _sample_cnt_lengths(
                filler_cnt.length_nm, cnt_sigma_L_nm, n_cnt_max, rng,
                l_min=cnt_l_min_nm, l_max=cnt_l_max_nm)
        phi_c = _run_hybrid_cnt_sweep(
            filler_cb, filler_cnt, n_cb, n_cnt_max,
            rve_size, tunnel_cutoff_nm, rng,
            kappa=kappa, r_cb_arr=r_cb_arr_rl, cnt_lengths_arr=cnt_lengths_rl,
        )
        if np.isnan(phi_c):
            failed += 1
            if verbose:
                print(f"  [{rl+1}/{n_realizations}] No percolation (n_cnt_max too low?)")
        else:
            samples.append(phi_c)
            if verbose:
                print(f"  [{rl+1}/{n_realizations}] φ_c_CNT = {phi_c:.4f}")

    phi_anal = _phi_c_theory(filler_cnt, tunnel_cutoff_nm)
    if samples:
        return PercolationResult(
            system="Hybrid",
            phi_c_mean=float(np.mean(samples)),
            phi_c_std=float(np.std(samples, ddof=1) if len(samples) > 1 else 0.0),
            phi_c_analytical=phi_anal,
            n_realizations=len(samples),
            n_failed=failed,
            rve_size_nm=rve_size,
            tunnel_cutoff_nm=tunnel_cutoff_nm,
        )
    return PercolationResult(
        system="Hybrid", phi_c_mean=np.nan, phi_c_std=np.nan,
        phi_c_analytical=phi_anal, n_realizations=0, n_failed=failed,
        rve_size_nm=rve_size, tunnel_cutoff_nm=tunnel_cutoff_nm,
    )
