# Case study: your own filler geometry with `generate/`

This walkthrough answers one question: *"I have measured filler geometry from
my own (or a published) composite — how do I get ideal-network reference
numbers for it?"* It uses only the public `generate/` command-line tools, and
every command here runs in seconds on a laptop. The output blocks below are
selected excerpts pasted from real runs (Python 3.11, single machine; your
timings will vary, your numbers will not — the seeds are fixed).

As the external geometry we use the MWCNT/polypropylene composites of
Pan & Li, *Polymer* 54:1218 (2013), the case-study system discussed in the
companion manuscript. Pan & Li measured their MWCNT geometry by TEM after
matrix burn-off, which makes their three filler types an unusually clean
external input:

Source: Pan, Y.; Li, L. "Percolation and gel-like behavior of multiwalled
carbon nanotube/polypropylene composites influenced by nanotube aspect ratio."
*Polymer* 54(3), 1218-1226 (2013).
https://doi.org/10.1016/j.polymer.2012.12.058

| Type | mean L [nm] | SD L [nm] | L support [nm] | d [nm] | Pan $w_{c,e}$ [wt.%] |
|---|---|---|---|---|---|
| AR51  | 660  | 437 | 100–2500 | 13 | 2.11 |
| AR84  | 1230 | 555 | 200–3000 | 14 | 1.51 |
| AR167 | 1840 | 711 | 300–4000 | 11 | 0.74 |

The examples use `--cnt-waviness 0.7`, the same ideal-network waviness
convention used in the companion manuscript. If your microscopy or image
analysis gives a different tortuosity/waviness estimate, replace this value and
treat the change as a model input, not a post-processing adjustment.

## Step 0 — convert wt.% to volume fraction

The model works in volume fractions; experiments usually report weight
fractions. With $\rho_{MWCNT} = 1.75\ \mathrm{g/cm^3}$ and
$\rho_{PP} = 0.9\ \mathrm{g/cm^3}$:

```python
rho_cnt, rho_pp = 1.75, 0.9
w = 2.11  # wt.%
phi = (w / rho_cnt) / ((w / rho_cnt) + ((100 - w) / rho_pp))
# 2.11 wt.% -> phi = 0.01096
```

For later use: 2.11 wt.% → $\phi=0.01096$, 1.51 wt.% → $\phi=0.00782$,
0.74 wt.% → $\phi=0.00382$, 3 wt.% → $\phi=0.01566$.

## Step 1 — a first look with `--quick`

Start with the cheapest sane run: the measured *mean* length as a single
(monodisperse) size, on the tool's default path:

```bash
python generate/percolation_threshold.py --filler cnt \
    --cnt-length-um 0.66 --cnt-diameter-nm 13 --cnt-waviness 0.7 --quick
```

```
 filler        : CNT (d=13 nm, L=0.66 um, waviness=0.7)
 RVE           : 1200 nm     tunnel cutoff : 10 nm
 realizations  : 3 ok (seed base 42)   n_max=450
----------------------------------------------------------------------
 phi_c (MC)    : 0.00816211 +/- 0.0035  [vol fraction]
 phi_c (theory): 0.0111331
 deviation     : -26.7 %  (finite-size / finite-N; shrinks with larger RVE and N)
```

Runtime: about 0.2 s. `--quick` is a *demo*, not a result: a small RVE and
$N=3$ realizations give a ±0.0035 spread on an $\approx 0.008$ value. Its job is to
confirm the setup is sane before you spend real compute. And the mean length
is an *entry point*, not the measurement — Pan & Li measured a wide length
distribution, which is what the next step feeds in.

## Step 2 — the measured distribution, on the polydisperse path

Passing any distribution flag switches `percolation_threshold.py` to its
polydisperse path: CNT lengths are sampled from a truncated lognormal with
the measured moments and support, each realization gets its own RNG seed
(`seed base + r`), and `--workers` parallelizes over realizations without
changing any number. AR51's measured histogram goes in directly:

```bash
python generate/percolation_threshold.py --filler cnt \
    --cnt-mu-l-nm 660 --cnt-sigma-l-nm 437 --cnt-l-min-nm 100 --cnt-l-max-nm 2500 \
    --cnt-diameter-nm 13 --cnt-waviness 0.7 \
    --rve-nm 4000 --phi-max-vol 0.02 --n-real 50 --seed 42 --workers 12
```

The excerpt below omits the progress lines printed during the run.

```
 path          : DISTRIBUTION (truncated lognormal, one seed per realization)
 filler        : CNT poly: L = 660 +/- 437 nm in [100, 2500], d=13 nm, waviness=0.7
 RVE           : 4000 nm     tunnel cutoff : 10 nm
 realizations  : 50 percolated / 50   (seed base 42, workers 12)
 stop caps     : phi_max=0.02   n_max=auto
----------------------------------------------------------------------
 phi_c (MC)    : median 0.00729249   IQR [0.00705027, 0.00767505]  [vol fraction]
                 mean 0.00729519 +/- 0.00053
 censored      : 0 / 50 hit the cap without percolating
```

How to read it:

- The `mu`/`sigma` inputs are the moments of the *truncated sampled
  population*, matching how a measured histogram is reported — see the main
  README. The support bounds are Pan & Li's observed length range.
- On this path each realization adds CNTs until the network spans, so the
  ensemble is summarized by **median and IQR** first (thresholds scatter
  asymmetrically), with mean ± std alongside.
- `--phi-max-vol` is a stop cap, not a target: a realization that reaches it
  without percolating is *censored* and counted, never silently averaged in.
  Here none were.
- The distribution matters: the polydisperse ensemble percolates at
  $\approx 0.0073$, clearly below the $\approx 0.009$ a monodisperse mean-length run gives —
  the long tail of the length distribution builds the spanning cluster
  early. That difference is physics, not noise, and it is why the measured
  distribution, not just the mean, belongs in the model.
- `--seed` (alias `--seed-base`) makes every run bit-reproducible; change it
  to check that your conclusions do not depend on one random stream.

## Step 3 — does the aspect-ratio ordering come out right?

Repeat with the other two measured geometries. The three geometric-threshold
runs below use the same $L_{\rm RVE}=4000$ nm box and $N=50$ ensemble size as
the conductivity sweep used later for the Table 2 fit; the threshold difference
then reflects the model layer, not a different box size or sample count:

```bash
python generate/percolation_threshold.py --filler cnt \
    --cnt-mu-l-nm 1230 --cnt-sigma-l-nm 555 --cnt-l-min-nm 200 --cnt-l-max-nm 3000 \
    --cnt-diameter-nm 14 --cnt-waviness 0.7 \
    --rve-nm 4000 --phi-max-vol 0.02 --n-real 50 --seed 42 --workers 12

python generate/percolation_threshold.py --filler cnt \
    --cnt-mu-l-nm 1840 --cnt-sigma-l-nm 711 --cnt-l-min-nm 300 --cnt-l-max-nm 4000 \
    --cnt-diameter-nm 11 --cnt-waviness 0.7 \
    --rve-nm 4000 --phi-max-vol 0.01 --n-real 50 --seed 42 --workers 12
```

```
 filler        : CNT poly: L = 1230 +/- 555 nm in [200, 3000], d=14 nm, waviness=0.7
 phi_c (MC)    : median 0.0057597   IQR [0.00535875, 0.00592943]  [vol fraction]

 filler        : CNT poly: L = 1840 +/- 711 nm in [300, 4000], d=11 nm, waviness=0.7
 phi_c (MC)    : median 0.00299233   IQR [0.00271795, 0.00319631]  [vol fraction]
```

## Step 4 — geometric threshold is only the first half

Collecting the three percolation runs gives the **geometric** model threshold
$w_{c,g}$. Pan & Li's reported threshold in the manuscript comparison is a
**conductivity-defined** threshold, $w_{c,e}$, so the comparison below is
intentionally not the final Table 2 yet:

| Type | Pan $\varphi_{c,e}$ | model $\varphi_{c,g}$, median [IQR] | model $w_{c,g}$, median [wt.%] |
|---|---|---|---|
| AR51  | 0.01096 | 0.00729 [0.00705, 0.00768] | 1.41 |
| AR84  | 0.00782 | 0.00576 [0.00536, 0.00593] | 1.11 |
| AR167 | 0.00382 | 0.00299 [0.00272, 0.00320] | 0.58 |

The ideal network reproduces the aspect-ratio **ordering** and the ~5×
**span** from TEM geometry alone — no fitted parameters entered at any point.
But $w_{c,g}$ sits below Pan's $w_{c,e}$. That is not a failure of the percolation
calculation; it is the physical distinction the manuscript uses. A spanning
geometric backbone can exist before the Simmons-weighted network carries the
macroscopic power law with measurable strength.

## Step 5 — conductivity-defined threshold and exponent

To get the manuscript's Table 2 transport columns, run conductivity over a
loading sweep and fit

$$
\sigma = \sigma_0\,(w - w_{c,e})^t
$$

with $w_{c,e}$, $t$, and $\sigma_0$ all free. The compact sweep summary used for
the manuscript is shipped as evidence, and the same general-purpose fitting
tool can be applied to your own CSV if it has a loading column and a positive
conductivity column:

```bash
python tools/fit_conductivity_powerlaw.py \
    data/article_evidence/pan2013_case_study/pan2013_conductivity_sweep_summary.csv \
    --group-col mwcnt_type --fit-include-col fit_include \
    --min-success-fraction 0.9
```

```
group  n_fit  x_c  t  sigma0  R2
AR167  6  0.82  2.05096424108  748.701149315  0.999996046967
AR51  4  2.125  2.07427012307  191.408507572  0.999998636386
AR84  5  1.535  2.04436151142  259.277583263  0.999997969896
```

For CSVs written directly by `generate/conductivity.py`, name that generator's
columns explicitly:

```bash
python tools/fit_conductivity_powerlaw.py my_conductivity_sweep.csv \
    --x-col phi_cnt_vol --sigma-col sigma_median_s_per_m \
    --n-total-col n_real
```

Now the manuscript Table 2 logic is visible:

| Type | model $w_{c,g}$ | model $w_{c,e}$ | Pan $w_{c,e}$ | model $t$ | Pan $t$ |
|---|---:|---:|---:|---:|---:|
| AR51  | 1.41 | 2.125 | 2.11 | 2.074 | 2.99 |
| AR84  | 1.11 | 1.535 | 1.51 | 2.044 | 3.10 |
| AR167 | 0.58 | 0.820 | 0.74 | 2.051 | 3.05 |

The ideal network gets the conductivity-defined thresholds close to Pan's
reported values, while its exponent remains near the universal 3D value
$t \approx 2$. That split is informative: the geometry and transport onset are
captured by the ideal reference, while Pan's larger $t$ points to
non-ideal junction-distribution or dispersion heterogeneity outside this
ideal model. Machine-readable copies of the combined Table 2 evidence, the
conductivity sweep summary, and the fit table ship in
`data/article_evidence/pan2013_case_study/`.

## Step 6 — one conductivity point with the full distribution

`generate/conductivity.py` accepts the same truncated log-normal geometry
directly. At 3 wt.% ($\phi = 0.01566$, comfortably above the AR51 threshold):

```bash
python generate/conductivity.py --phi-cnt-vol 0.01566 \
    --cnt-mu-l-nm 660 --cnt-sigma-l-nm 437 \
    --cnt-l-min-nm 100 --cnt-l-max-nm 2500 \
    --cnt-diam-nm 13 --cnt-waviness 0.7 \
    --l-rve-nm 4000 --seg-len-nm 50 --tunnel-nm 10 \
    --n-real 50 --seed-base 20262706 --workers 10
```

The excerpt below omits the progress lines printed during the run.

```
 phi target    : CB=0  CNT=0.0156567  [vol fraction]
 CNT dist      : mu_L=660 sigma_L=437 nm  [100,2500]  d=13 w=0.7
 RVE           : 4000 nm   tunnel=10 nm  d_min=0.34 nm  G_cut=1e-15 S
 realizations  : 50 ok / 50   (seed base 20262706, workers 10)   n_cb_max=201 n_cnt_max=28796
----------------------------------------------------------------------
 sigma (MC)    : 150 +/- 27  [S/m]
 sigma median  : 145.1  [p25 132.9, p75 165.5]  [S/m]
 note          : sigma can span orders of magnitude near threshold; median/IQR is the robust locator
```

This row uses the public generator settings that reproduce the archived AR51
3 wt.% conductivity summary; `--workers` changes wall time only.

**How to read the number.** The reported $\sigma$ is the *raw* network
conductivity: it carries one undetermined material-level junction-conductance
prefactor that the ideal model deliberately does not fit. Comparisons against
experiments are therefore made on prefactor-independent quantities — slopes
versus loading, ratios between filler types, threshold ordering — not on the
absolute S/m value of a single point. One conductivity point is useful for
checking your setup; estimating $w_{c,e}$ and $t$ requires a loading sweep like
the compact evidence table above.

## Scaling up (publication-scale box)

> **Expensive — not the default.** The manuscript-scale campaigns behind the
> paper's figures use RVE edges of 4000 nm and ensembles of 50–300
> realizations per point, which turns seconds into hours and makes memory
> per realization significant at high loading. Those exact campaigns, with
> their frozen outputs, are `reproduce_paper/` wrappers — do not rebuild them
> from this tutorial; rerun the wrappers.

## Where to go next

- `examples/quickstart.md` — first numbers from each tool in five minutes.
- `examples/one_run_anatomy.ipynb` — the anatomy of one realization, if you
  want to see what happens inside a single Monte Carlo run.
- `python generate/<tool>.py --help` and the "What do you want?" table in the
  main README — all generator flags, including the CB and gauge-factor entry
  points.
- `reproduce_paper/` + `data/reference/` — the frozen publication campaigns
  and the tables they must reproduce.
