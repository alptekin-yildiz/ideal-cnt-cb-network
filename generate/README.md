# generate/

User-facing command-line generators for running your own CNT/CB geometry. These
scripts are the entry points for new calculations; they are not the frozen
paper-campaign wrappers in `reproduce_paper/`.

Each script writes one ensemble-summary row for the parameters you provide:

| Tool | Question answered |
|---|---|
| `percolation_threshold.py` | At what filler volume fraction does the network span? |
| `conductivity.py` | What Simmons-weighted conductivity does this composition produce? |
| `gauge_factor.py` | What gauge factor does this composition produce under the strain protocol? |

Start with `examples/quickstart.md` for first runs, then return here when you
want to choose parameters deliberately.

For manuscript-scale sanity checks using the same user-facing tools, see
`examples/manuscript_anchor_checks.md`. Those runs use `generate/` directly and
recover the scale of selected manuscript anchors; exact archived-table
regeneration remains the role of `reproduce_paper/`.

## Shared Rules

- Inputs use volume fraction (`--phi-*-vol`), not wt.%. Convert experimental
  wt.% values before calling the generators.
- `--n-real` controls the number of Monte Carlo realizations. Increase it to
  reduce seed-to-seed scatter; do not choose a seed because it gives a nicer
  result.
- `--seed-base` is a reproducibility label, not a fitted parameter. Report it
  with any result you share.
- `--workers` changes wall time only. The generators assign one seed per
  realization and return results in realization order, so worker count does not
  change the numbers. On the default monodisperse path of
  `percolation_threshold.py`, `--workers` is ignored with a warning.
- `--progress-every N` prints `ok/fail/elapsed/eta` after every `N`
  completed realizations in longer runs. Use `--progress-every 0` to silence
  it. This changes only terminal output, not seeds, CSV schemas, or numerical
  results.
- `--quick` is a smoke/demo mode. It is useful for checking that a command is
  sensible, not for publication-scale statistics.
- `--out path.csv` writes the summary row directly and overwrites an existing
  file at the same path (a warning is printed to stderr when that happens).
  Use distinct filenames while exploring. There is no skip/resume policy in
  `generate/`; long reproducible campaigns live under `reproduce_paper/`.
- `--cnt-waviness` is the cosine of the bend angle between successive CNT
  segment directions (`1.0` = straight rod), not an end-to-end-to-contour
  ratio. See "Assumptions and conventions" in the top-level README for this
  and the other model conventions.

## Mono and Poly Inputs

`percolation_threshold.py` has two paths:

- Default monodisperse path: one CNT length (`--cnt-length-um`) or one CB
  diameter (`--cb-diameter-nm`).
- Distribution/hybrid path: passing any distribution flag
  (`--cnt-mu-l-nm`, `--cnt-sigma-l-nm`, `--cb-mu-d-nm`, and related bounds) or
  `--phi-cb-vol` switches to truncated-lognormal sampling with one seed per
  realization.

`conductivity.py` and `gauge_factor.py` use distribution inputs by default.
Set a `sigma` flag to `0` to collapse that filler to the monodisperse limit, for
example `--cnt-sigma-l-nm 0`.

The `mu` and `sigma` flags are target moments of the truncated sampled
population, not parent-lognormal parameters. The code solves the parent
distribution internally and samples only inside the requested support bounds.

The two paths also use two documented CNT segmentation conventions: the
monodisperse path discretizes each CNT into 10 equal segments, while the
distribution-based path uses a fixed 50 nm contour step plus a terminal
remainder segment (the conventions coincide at $L=500$ nm). The monodisperse
path additionally draws all realizations from a single sequential RNG stream,
so changing `--n-real` there changes the whole realization sequence; the
distribution path seeds each realization independently (`seed_base + r`).

## Typical Workflow

1. Pick the question: threshold, conductivity, or GF.
2. Start with a small `--quick` or low `--n-real` command.
3. Check that `n_ok`, censoring/fail counts, and reported uncertainty make
   sense.
4. Increase RVE and `--n-real` for the result you intend to report.
5. Save the summary with `--out`.
6. If you need a conductivity-defined threshold or exponent from a loading
   sweep, fit the saved sweep with `tools/fit_conductivity_powerlaw.py`.

## Examples

Monodisperse CNT percolation threshold:

```bash
python generate/percolation_threshold.py --filler cnt \
    --cnt-length-um 1.0 --cnt-diameter-nm 12 \
    --rve-nm 3000 --n-max 2000 --n-real 20 --out my_cnt_threshold.csv
```

Polydisperse CNT percolation threshold:

```bash
python generate/percolation_threshold.py --filler cnt \
    --cnt-mu-l-nm 660 --cnt-sigma-l-nm 437 \
    --cnt-l-min-nm 100 --cnt-l-max-nm 2500 \
    --cnt-diameter-nm 13 --cnt-waviness 0.7 \
    --rve-nm 2500 --phi-max-vol 0.02 --n-real 20 --workers 4
```

Conductivity at a chosen CNT volume fraction:

```bash
python generate/conductivity.py --phi-cnt-vol 0.01566 \
    --cnt-mu-l-nm 660 --cnt-sigma-l-nm 437 \
    --cnt-l-min-nm 100 --cnt-l-max-nm 2500 \
    --cnt-diam-nm 13 --cnt-waviness 0.7 \
    --l-rve-nm 2500 --n-real 10 --workers 4
```

The conductivity summary includes median channel-power fractions for `ss`
(CB-CB), `sc` (CB-CNT), and `cc` (CNT-CNT) junctions. These fractions show
which junction family carries the simulated current at the chosen composition.

Gauge factor at a chosen composition:

```bash
python generate/gauge_factor.py --phi-cnt-vol 0.02 \
    --cnt-sigma-l-nm 0 --l-rve-nm 1000 \
    --strain-grid 0,0.0025,0.005,0.0075,0.01 \
    --n-real 2 --workers 2
```

The GF summary reports the same channel-power fractions at the first and last
strain points. These fractions are a compact way to see which junction family
carries the simulated current while you vary `--opening-alpha-ss/-sc/-cc`.

Fit a conductivity sweep after you have collected rows:

```bash
python tools/fit_conductivity_powerlaw.py my_conductivity_sweep.csv \
    --x-col phi_cnt_vol --sigma-col sigma_median_s_per_m \
    --n-total-col n_real
```

For exact manuscript-output regeneration, use `reproduce_paper/` and compare
the regenerated summaries against `data/reference/`.
