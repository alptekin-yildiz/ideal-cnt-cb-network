# Manuscript Anchor Checks with `generate/`

These short runs use the public `generate/` tools, not the frozen paper
wrappers. They are meant as manuscript-scale sanity checks: with matching
geometry and loading, the same kernels recover the scale and trend of selected
manuscript anchors in seconds to a few minutes. For exact regeneration of the
archived CSV tables, use `reproduce_paper/` and compare against
`data/reference/`.

The commands below deliberately use far fewer realizations than the manuscript
tables. The reference anchors are listed so the comparison is transparent.
Worker count changes wall time only; the seeds and reported statistics are
worker-invariant for these paths.
Progress lines may appear during the run; the result excerpts below keep only
the final scientific summary lines.
The non-round seed bases below are inherited from the corresponding frozen
campaign rows where useful; they are reproducibility labels, not fitted
parameters.

## 1. CNT Percolation Threshold

Manuscript anchor:

- Reference table: `data/reference/percolation/cnt_phic_sweep_fixedstep_Lmin50.csv`
- Selected row: CNT length `500 +/- 300 nm`, truncated to `[50, 1500] nm`
- Reference value: $\phi_c = 0.0064527297$ from `N=100` realizations

Quick `generate/` check:

```bash
python3 generate/percolation_threshold.py --filler cnt \
    --cnt-mu-l-nm 500 --cnt-sigma-l-nm 300 \
    --cnt-l-min-nm 50 --cnt-l-max-nm 1500 \
    --cnt-diameter-nm 10 --cnt-waviness 0.7 \
    --rve-nm 4000 --phi-max-vol 0.02 \
    --n-real 20 --workers 8 --seed 520320527
```

Observed result lines:

```text
realizations  : 20 percolated / 20   (seed base 520320527, workers 8)
phi_c (MC)    : median 0.00640265   IQR [0.00611571, 0.00658613]  [vol fraction]
                 mean 0.00629784 +/- 0.00038
censored      : 0 / 20 hit the cap without percolating
```

This short ensemble lands on the same threshold scale as the archived
`N=100` manuscript row.

## 2. CNT Simmons Conductivity Near 3 wt.%

Manuscript anchor:

- Reference table: `data/reference/conductivity/simmons_conductivity_regime_n300_cnt_high_to_3wt_L4000.csv`
- Selected row: pure CNT, $\phi_{\rm CNT} = 0.021952186$
- Reference value: $\sigma_{\rm mean} = 1339.9816$ S/m,
  $\sigma_{\rm median} = 1341.9009$ S/m from `N=300` realizations

Quick `generate/` check:

```bash
python3 generate/conductivity.py \
    --phi-cnt-vol 0.021952186 \
    --cnt-mu-l-nm 500 --cnt-sigma-l-nm 300 \
    --cnt-l-min-nm 50 --cnt-l-max-nm 1500 \
    --cnt-diam-nm 10 --cnt-waviness 0.7 \
    --l-rve-nm 4000 --tunnel-nm 10 --d-min-nm 0.34 \
    --g-cutoff-s 1e-15 --seg-len-nm 50 \
    --n-real 20 --workers 8 --seed-base 20760610
```

Observed result lines:

```text
realizations  : 20 ok / 20   (seed base 20760610, workers 8)   n_cb_max=201 n_cnt_max=89642
sigma (MC)    : 1346 +/- 43  [S/m]
sigma median  : 1352  [p25 1315, p75 1386]  [S/m]
channel power : ss=0 sc=0 cc=1 [median fraction]
```

The short `generate/` run recovers the manuscript-scale conductivity without
using any precomputed paper table. The channel-power diagnostic also matches
the physical expectation for a pure CNT network: current is carried by CNT-CNT
junctions.

## 3. CNT Ideal-Affine Gauge Factor Near 3 wt.%

Manuscript anchor:

- Reference table: `data/reference/gf/gf_simmons_gf_affine_cnt_3wt_n300.csv`
- Selected row: pure CNT, $\phi_{\rm CNT} = 0.021952186$, affine strain,
  zero junction-opening alpha
- Reference value: $GF_{R,\rm mean} = 0.2509714$,
  $GF_{R,\rm median} = 0.24795603$, and
  $GF_{\sigma,\rm median} = -1.4683748$ from `N=300` realizations

Quick `generate/` check:

```bash
python3 generate/gauge_factor.py \
    --phi-cnt-vol 0.021952186 \
    --cnt-mu-l-nm 500 --cnt-sigma-l-nm 300 \
    --cnt-l-min-nm 50 --cnt-l-max-nm 1500 \
    --cnt-diam-nm 10 --cnt-waviness 0.7 \
    --l-rve-nm 4000 --tunnel-nm 10 --d-min-nm 0.34 \
    --g-cutoff-s 1e-15 --seg-len-nm 50 \
    --strain-grid 0,0.0025,0.005,0.0075,0.01 \
    --poisson-nu 0.36 \
    --n-real 8 --workers 8 --seed-base 20260617
```

Observed result lines:

```text
realizations  : 8 ok / 8   (seed base 20260617, workers 8)   n_cb_max=201 n_cnt_max=89642
GF_r (MC)     : 0.2382 +/- 0.08   [d ln R / d eps]
decomposition : gf_sigma=-1.478   gf_geom=1.716   (gf_r ~ gf_sigma + gf_geom)
channel power : eps0 ss=0 sc=0 cc=1; epsmax ss=0 sc=0 cc=1 [median fraction]
```

The short run reproduces the key ideal-affine message: the CNT-bearing ideal
reference GF is small and positive because a negative conductivity contribution
nearly cancels the geometric term. The channel-power line gives the compact
Table 3 style diagnostic: for a pure CNT network, the conducting power is
carried by CNT-CNT junctions.

## How to Read These Checks

These examples are intentionally not substitutes for `reproduce_paper/`.
They demonstrate that the user-facing generators expose the same computational
kernels on manuscript-scale inputs. The frozen wrappers additionally preserve
the exact campaign grids, seeds, tags, summary schemas, and reference-table
provenance needed for one-to-one reproduction.
