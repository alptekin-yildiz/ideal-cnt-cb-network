# ideal-cnt-cb-network

Stochastic **ideal-network reference generator** for CNT/CB polymer composites —
percolation, conductivity, and piezoresistive (gauge-factor) response along a
single Monte Carlo pipeline.

This is a *reference*, not a fit. It computes what an **ideal** (well-dispersed,
geometrically explicit) CNT/CB network delivers on its own, so that a real
composite's departure from it can be read as a signed, mechanism-specific
effect — agglomeration, non-universal tunneling transport, non-affine junction
opening — rather than absorbed into adjustable parameters.

Companion code for M. Karabal, A. Yıldız, *A stochastic ideal-network reference
for CNT/CB polymer composites: Diagnosing departures in percolation,
conductivity, and piezoresistive response*, Computational Materials Science
275 (2026) 115084, <https://doi.org/10.1016/j.commatsci.2026.115084>.
The code version underlying the paper (v0.1.0) is archived on Zenodo as
<https://doi.org/10.5281/zenodo.21382449>; the concept DOI
<https://doi.org/10.5281/zenodo.21382448> always resolves to the latest archived
version.

## Install

```bash
git clone https://github.com/alptekin-yildiz/ideal-cnt-cb-network.git && cd ideal-cnt-cb-network
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requires Python ≥ 3.10 with `numpy` and `scipy`.

To verify a development checkout:

```bash
pip install -e ".[test]"
pytest -q
```

## Where should I start?

- To run your own CNT/CB geometry, start with `examples/quickstart.md`, then
  use the parameter-driven tools in `generate/` (see `generate/README.md`).
- To see the user-facing generators recover selected manuscript-scale anchor
  values, run through `examples/manuscript_anchor_checks.md`.
- To inspect what happens inside one Monte Carlo realization, open
  `examples/one_run_anatomy.ipynb`.
- To follow an external measured-geometry case study, read
  `examples/pan2013_case_study.md`.
- To reproduce the manuscript outputs exactly, use `reproduce_paper/` and
  compare regenerated tables against `data/reference/`.
- To inspect compact manuscript evidence that is not a full raw campaign
  archive, use `data/article_evidence/`.
- To extend the method, start in `cntcb/` and keep `pytest -q` green.

## The stochastic contract

Every quantity here is a **Monte Carlo ensemble**: a single realization is
noisy, so each generator runs `N` realizations and reports **mean ± std**. That
is why `--n-real` exists, and why a quick demo (`--quick`, few realizations) and
a publication run (many realizations, larger RVE) return the *same* number with
*different* error bars.

Seeds are reproducibility labels, not fitted parameters. Use any fixed
`--seed-base` for your own study, report it with the command, and increase
`--n-real` or the RVE size rather than choosing a favorable seed.

## Mono- or polydisperse?

The generators support both monodisperse and polydisperse inputs. Sizes are
sampled from **truncated lognormal distributions**: the CNT length follows
`--cnt-mu-l-nm` /
`--cnt-sigma-l-nm` (default 500 ± 300 nm, truncated to [`--cnt-l-min-nm`,
`--cnt-l-max-nm`]) and the CB effective aggregate diameter follows
`--cb-mu-d-nm` / `--cb-sigma-d-nm` (default 148 ± 83 nm, truncated to
[`--cb-d-min-nm`, `--cb-d-max-nm`]). In `conductivity.py` and
`gauge_factor.py` **polydispersity is the default**; setting a `sigma` flag to
`0` collapses that filler to the monodisperse limit — exactly what the quick
demos below do with `--cnt-sigma-l-nm 0`.

`percolation_threshold.py` has two explicit paths. Its default is the
original monodisperse threshold — one length or diameter per run, on a
single sequential RNG stream. Passing any distribution flag, or
`--phi-cb-vol` for the CNT-threshold-in-a-CB-background hybrid question,
switches to the polydisperse path: truncated-lognormal sizes, one RNG seed
per realization, parallel with `--workers` (in every generator, any worker
count gives bit-identical numbers). `--help` spells out which flags select
which path. The full polydisperse threshold *sweeps* behind the paper ($\phi_c$
versus length or diameter distribution) remain frozen as `reproduce_paper/`
wrappers with their summaries in `data/reference/`.

The `mu`/`sigma` flags are the target mean and standard deviation of the
**truncated sampled population**, not raw parent lognormal parameters. The code
solves the parent lognormal moments internally, records the realized/truncated
moments in paper-output CSVs, and then samples only inside the requested support
interval. Narrowing the support or changing `sigma` is therefore a real model
choice, not a plotting preference.

## What do you want?

The scripts in `generate/` are the user-facing entry points: enter your own
parameters and get one ensemble-summary row. They are not frozen paper
campaign wrappers. With `--out`, they write that summary CSV directly and
overwrite an existing file at the same path; use distinct output names when
exploring.

| You care about | Run | You get |
|---|---|---|
| First numbers, five minutes | read `examples/quickstart.md` | each tool run once, with real outputs and the shared flags explained |
| **Percolation threshold** (CNT, CB, or hybrid; mono- or polydisperse) | `python generate/percolation_threshold.py --filler cnt --quick` | $\phi_c$ ensemble statistics (mean ± std; median/IQR on the polydisperse path), analytical guide |
| **Conductivity** $\sigma$ (Simmons-weighted, your $\phi$) | `python generate/conductivity.py --phi-cnt-vol 0.02 --cnt-sigma-l-nm 0 --l-rve-nm 1000 --n-real 2` | $\sigma$ [S/m] mean ± std (+ median / IQR) and SS/SC/CC channel-power fractions |
| Conductivity threshold/exponent from a loading sweep | `python tools/fit_conductivity_powerlaw.py my_sweep.csv --group-col sample` | free fit of $\sigma = \sigma_0 (w - w_{c,e})^t$: $w_{c,e}$, $t$, $\sigma_0$, $R^2$ |
| **Gauge factor** (piezoresistive response) | `python generate/gauge_factor.py --phi-cnt-vol 0.02 --cnt-sigma-l-nm 0 --l-rve-nm 1000 --n-real 2` | GF mean ± std, `gf_sigma` / `gf_geom`, and SS/SC/CC channel-power fractions |
| Manuscript-scale sanity checks with `generate/` | read `examples/manuscript_anchor_checks.md` | short percolation, conductivity, and GF runs that recover selected manuscript-anchor scales |
| Understand one run end-to-end | open `examples/one_run_anatomy.ipynb` | step-by-step anatomy of a single realization |
| Study *your own* measured geometry | read `examples/pan2013_case_study.md` | a worked external case study (Pan & Li 2013 MWCNT/PP) using only `generate/` |
| Reproduce a paper figure exactly | `reproduce_paper/` (see its README) | the exact settings behind each figure |

### Example — your own CNTs

```bash
python generate/percolation_threshold.py --filler cnt \
    --cnt-diameter-nm 12 --cnt-length-um 1.0 --cnt-waviness 0.8 \
    --rve-nm 3000 --n-max 2000 --n-real 20 --out my_threshold.csv
```

Swap the fixed length for a distribution flag (e.g. `--cnt-sigma-l-nm 300`)
and the same tool thresholds the sampled population instead —
`examples/pan2013_case_study.md` walks a measured literature geometry
through exactly that.

The percolation-only user never touches the conductivity or gauge-factor code:
each generator is scoped to a single question.

> **Reading `NaN`:** ensemble statistics average over *successful* (spanning)
> realizations only. In a small RVE near or below the transport threshold —
> e.g. a CB-only network at moderate φ, where the Simmons `--g-cutoff-s`
> prunes weak junctions — some or all seeds may fail to span; `n_ok = 0` with
> `NaN` means "no conducting path formed in this box", not a numerical bug.
> Grow the RVE, raise φ, or add realizations.

### Reading the ideal gauge factor

With the junction-opening coefficients at zero, the network deforms purely
**affinely**. In the $N=300$, $L_{\rm RVE}=4000$ nm reference ensembles, the
CNT-bearing ideal-affine reference GF is about 0.25–0.27, below the geometric
baseline $1 + 2\nu \approx 1.72$, because the conductivity term is negative. This is
a reference value, not a universal constraint: real networks with different
orientation or non-affine junction kinematics may depart on either side of it.
Measured values above this ideal-affine reference are interpreted here as
evidence for **non-affine junction opening**, which you can explore with
`--opening-alpha-ss/-sc/-cc`. Small quickstart-scale demos can yield noisy
single-network GF values; they are illustrative, not evidential.

## Repository layout

```
# 1 — Engine (extend the method)
cntcb/kernel/     material definitions, percolation kernels, spatial hashing
cntcb/engines/    the four question engines (percolation, conductivity, Simmons, GF)

# 2 — User interface (run your own geometry)
generate/         parameter-driven generators; see generate/README.md
examples/         quickstart, anchor checks, one-run anatomy, external case study

# 3 — Paper campaigns and evidence (reproduce the manuscript)
reproduce_paper/  the exact wrappers behind the paper figures → write data/processed/
data/reference/   frozen stage anchors, regenerated one-to-one by the wrappers
data/article_evidence/  derived evidence tables and provenance for manuscript claims

# 4 — Verification (tests and drift checks)
tests/            pytest suite: kernel invariants, CLI smoke, fixed-seed canaries, data contracts
tools/            verifiers and small fitting helpers for regenerated/evidence tables
```

## Assumptions and conventions

The model is an intentionally idealized reference system. The conventions below
are load-bearing for interpreting its outputs; they define the public
interpretation of the code, documentation, and accompanying manuscript.

### Geometry and percolation

**Waviness is a local orientation correlation, not an end-to-end ratio.**
A wavy CNT is generated as a chain of straight segments. The waviness parameter
$w$ is the cosine of the bend angle between successive segment directions
($w=1$ recovers a straight rod). The ensemble RMS end-to-end-to-contour ratio
is therefore lower than $w$ itself, for example about 0.645 for $w=0.7$ with
10 equal segments. The analytical threshold estimate that uses $\mathrm{AR}\,w$ should
be read as a heuristic effective-aspect-ratio guide, not as a derived formula
for the generated chain.

**Two segmentation conventions are documented.**
The monodisperse path discretizes each CNT into 10 equal segments. The
distribution-based path uses a fixed 50 nm contour step plus a terminal
remainder segment. At $L=500$ nm, used in the monodisperse CNT
finite-size-scaling benchmark, the two conventions coincide. The distribution
and transport campaigns use the fixed-step path throughout. Under the
fixed-step convention, a longer CNT contains more bends at the same bend
cosine, so its realized end-to-end-to-contour ratio decreases with contour
length; the effective waviness is length-dependent. The frozen
manuscript scripts are regression-checked against the reusable engines at the
intended equivalence points; they are not independent scientific models.

**Volume fractions are nominal.**
Particles are penetrable and overlap-tolerant. The reported volume fraction is
the nominal sum of particle volumes divided by the RVE volume,
$\sum_i V_i/V_{\rm RVE}$, not an overlap-corrected occupied volume. For a
simple fully penetrable reference, the occupied fraction would be smaller,
e.g. $1-\exp(-\phi)$; experimental loading comparisons should keep this
convention in mind.

**The RVE uses open boundaries.**
No periodic boundary wrapping is applied. Particles may extend beyond the box,
and their full nominal volume is counted. Finite-size effects are quantified by
the finite-size-scaling campaigns rather than removed by periodic boundaries.

**Geometric spanning and transport use different direction conventions.**
The geometric percolation threshold counts spanning along any of the three
axes. Conductivity and gauge-factor calculations use x-direction electrodes.
This deliberate convention difference can contribute a small systematic offset
between geometric and electrical threshold estimates.

**Threshold statistics are conditional on spanning.**
Ensemble means and medians of $\phi_c$ are computed over realizations that
percolate within the simulated range. Non-spanning realizations are reported
separately through `n_fail`.

### Transport model

**Transport is junction-limited.**
Each filler object is represented as an equipotential node with zero intrinsic
filler resistance. Each filler pair contributes at most one junction, located
through the pair's minimum surface-to-surface separation. Network resistance is
therefore entirely junction-limited; trends across loading, waviness, filler
type, and strain are the intended observables.

**The Simmons prefactor is a convention.**
The tunneling conductance uses the low-bias Simmons exponential with one fixed
prefactor convention. Absolute conductivities should be interpreted up to an
order-one prefactor uncertainty. Trends, threshold offsets, and fitted
exponents are controlled primarily by the exponential distance dependence,
$\exp(-2\kappa d)$.

**`--tunnel-nm` is a connectedness cutoff for candidate junctions.**
The default 10 nm value is a surface-to-surface cutoff used to decide which
filler pairs become candidate junctions. In transport runs, those candidate
edges are then Simmons-weighted, and numerically weak edges are pruned by
`G_cutoff`. The cutoff is an explicit model parameter and is recorded as
provenance in output tables.

**Gauge-factor calculations use coordinate-affine strain.**
The network is deformed by homogeneously rescaling all CB centers and CNT
segment endpoint coordinates; particle radii are kept fixed. This is not a
rigid-fiber or non-affine micromechanical model. The reported ideal GF is the
response of the affinely deformed junction network.

**The low CNT-bearing ideal GF is an ensemble-level ideal-affine reference.**
The production claim rests on the $N=300$, $L_{\rm RVE}=4000$ nm reference
ensembles. For CNT-bearing systems, the ideal affine resistance GF is about
0.25–0.27, well below the geometric value $1 + 2\nu \approx 1.72$, because the
conductivity term is negative. Small quickstart-scale demos can yield noisy
single-network GF values; they are illustrative, not evidential.

### Materials idealization

**CB is a spherical-equivalent ideal reference.**
Carbon black is modeled as penetrable spheres with a matched effective diameter
distribution, not as fractal aggregates or image-resolved primary-particle
clusters. Deviations of experimental CB thresholds from this reference can
therefore reflect the spherical-equivalent idealization itself, not only
dispersion effects.

### Reproducibility notes

**RNG streams are part of the reproducibility contract.**
Distribution-based and transport paths derive one independent seed per
realization (`seed_base + r`), so ensembles are extensible and worker-count
invariant. The historical monodisperse path draws all realizations from a
single RNG stream; changing `--n-real` there changes the whole realization
sequence. This asymmetry is intentional and preserved for exact reproducibility
of the published benchmark.

**Canary tests are regression guards, not portability claims.**
Fixed-seed canary tests pin small deterministic values at tight tolerances on
the CI platforms. On a different BLAS/NumPy stack, last-digit differences
should first be checked as platform variance, not immediately interpreted as
scientific error.

**The public API is the CLI layer.**
The supported user interface is the `generate/` command-line tools and the
frozen `reproduce_paper/` scripts. The `cntcb/` package internals, including
underscore-prefixed helpers, are package-internal implementation details and
may change between versions.

## Reproducibility

`generate/` gives you ensemble summaries for *your* parameters. The curated
publication anchors live in `data/reference/` and are frozen with the code in
the Zenodo snapshot: the `percolation/`, `conductivity/`, and `gf/` directories
hold one-to-one regenerated summary tables. Derived evidence — the compact
numerical support for the manuscript's Fig. 10/Table 3 mechanism discussion,
the Pan & Li Table 2 case study, and the figure/table provenance inventory —
lives separately in
`data/article_evidence/`. Raw per-realization ensembles are not archived separately
— they are *regenerated*: every `reproduce_paper/` wrapper derives its random
seeds deterministically from a fixed `--seed-base`, so re-running a wrapper
reproduces its raw data exactly (see `reproduce_paper/README.md` for the
environment used for the paper runs).

The test suite includes two levels of reproducibility checks. Fast canary tests
run small fixed-seed simulations to catch accidental scientific-output changes
during code cleanup. The `data/reference/` tables are compact publication
anchors grouped by scientific stage (`percolation/`, `conductivity/`, `gf/`);
they can be compared against regenerated `data/processed/` outputs with the
same relative layout, without archiving the much larger per-realization
ensembles:

```bash
pytest -q
python tools/verify_reference_outputs.py \
    --reference-dir data/reference \
    --generated-dir data/processed
python tools/verify_gf_mechanism_support.py
```

### What the tests mean

The automated tests are not all the same kind of check:

| Test layer | What it guards |
|---|---|
| `tests/test_kernel.py` | deterministic kernel assumptions such as material defaults, boundary flags, and analytical threshold helpers |
| `tests/test_smoke.py` | importability, end-to-end CLI execution, and fixed-seed canaries for percolation, conductivity, and gauge factor |
| `tests/test_percolation_distribution.py` | the polydisperse/hybrid threshold path: engine equivalence against the frozen paper wrappers, per-realization seeding, CLI validation errors, and the untouched monodisperse path |
| `tests/test_transport_workers.py` | the conductivity/GF ensemble drivers: `--workers` gives bit-identical numbers at both the engine and CLI level |
| `tests/test_conductivity_powerlaw_fit.py` | the conductivity-defined threshold fit helper and the Pan & Li Table 2 evidence contract |
| `tests/test_reference_outputs.py` | integrity of the `data/reference/` publication anchors, the `data/article_evidence/` package, and the CSV comparison tools |
| `tools/verify_reference_outputs.py` | regenerated long-run stage summaries against the frozen percolation/conductivity/GF reference tables |
| `tools/verify_gf_mechanism_support.py` | compact Fig. 10/Table 3 evidence package in `data/article_evidence/` against its derived audit contract |
| `tools/fit_conductivity_powerlaw.py` | user-facing fit of $\sigma = \sigma_0 (w - w_{c,e})^t$ from a conductivity-loading CSV |

The quick tests are designed to be cheap enough for CI; the full paper sweeps
remain deterministic but are intentionally run only when regenerating
`data/processed/`.

## Citing

See [`CITATION.cff`](CITATION.cff). Please cite both the software and the paper:

Karabal, M., & Yıldız, A. (2026). A stochastic ideal-network reference for
CNT/CB polymer composites: Diagnosing departures in percolation, conductivity,
and piezoresistive response. *Computational Materials Science*, 275, 115084.
<https://doi.org/10.1016/j.commatsci.2026.115084>

Karabal, M., & Yıldız, A. (2026). *ideal-cnt-cb-network: Stochastic
ideal-network reference generator for CNT/CB polymer composites* [Computer
software]. Zenodo. <https://doi.org/10.5281/zenodo.21382448>

To reproduce the paper exactly, cite and use version 0.1.0,
<https://doi.org/10.5281/zenodo.21382449>; its computational code is identical
to version 1.0.0.

The cluster-tracking idea used by the percolation kernels builds on the
direction-cut method introduced in:

Yıldız, A. (2020). Yön-Kesme Yöntemi ve Kare Izgarada Adreslenmiş Temel
Arşimet Latislerinde İki-Boyutlu Bağ Perkolasyonu [Direction-Cut Method and
Two-Dimensional Bond Percolation in Basic Archimedean Lattices Addressed on a
Square Grid]. *Avrupa Bilim ve Teknoloji Dergisi*, (18), 515-530.

## License

[MIT](LICENSE).
