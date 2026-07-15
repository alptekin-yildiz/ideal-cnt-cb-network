# Quickstart — first numbers in five minutes

You cloned the repo and want output. This page runs the three `generate/`
tools once each — percolation threshold, conductivity, gauge factor — and
tells you how to read what comes back. Every command below finishes in
seconds on a laptop; the outputs shown are real runs of these exact commands.

Only the command-line tools appear here. If you want to see the engine
underneath a single realization, open `examples/one_run_anatomy.ipynb`; if
you want to feed *measured* literature geometry through the tools, read
`examples/pan2013_case_study.md`.

## 0. Install (once)

```bash
git clone https://github.com/alptekin-yildiz/ideal-cnt-cb-network.git && cd ideal-cnt-cb-network
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requires Python ≥ 3.10 with `numpy` and `scipy`. The `generate/` scripts also
run from a bare checkout without installing (they bootstrap the repo root
onto `sys.path`).

## 1. Percolation threshold

```bash
python generate/percolation_threshold.py --filler cnt --quick
```

```text
 filler        : CNT (d=10 nm, L=0.5 um, waviness=0.7)
 RVE           : 1200 nm     tunnel cutoff : 10 nm
 realizations  : 2 ok (seed base 42)   n_max=450
----------------------------------------------------------------------
 phi_c (MC)    : 0.00820396 +/- 0.0008  [vol fraction]
 phi_c (theory): 0.01
 deviation     : -18.0 %  (finite-size / finite-N; shrinks with larger RVE and N)
 failed runs   : 1 / 3
```

Three things to read off:

- **The answer is an ensemble.** `phi_c (MC)` is the mean ± std over the
  realizations that percolated — a single Monte Carlo draw is noisy by
  construction, which is why `--n-real` exists at all.
- **Failures are information.** One of the three quick realizations hit
  `n_max` before spanning the box; it is excluded and counted, not hidden.
- **`--quick` trades statistics for speed.** A publication-style run
  (`--n-real 20`, larger `--rve-nm`) returns the same number with tighter
  error bars and smaller finite-size deviation.

That run was **monodisperse** (one length per run) because no distribution
flag was passed. Pass any `--cnt-sigma-l-nm`-style flag and the tool switches
to the **polydisperse** path — truncated-lognormal sizes, one RNG seed per
realization, parallel with `--workers`:

```bash
python generate/percolation_threshold.py --filler cnt \
    --cnt-sigma-l-nm 300 --quick --workers 2 --out my_threshold.csv
```

```text
 path          : DISTRIBUTION (truncated lognormal, one seed per realization)
 filler        : CNT poly: L = 500 +/- 300 nm in [50, 1500], d=10 nm, waviness=0.7
 realizations  : 3 percolated / 3   (seed base 42, workers 2)
 stop caps     : phi_max=0.05   n_max=auto
----------------------------------------------------------------------
 phi_c (MC)    : median 0.00856656   IQR [0.00802755, 0.00903938]  [vol fraction]
                 mean 0.00852244 +/- 0.001
 censored      : 0 / 3 hit the cap without percolating
```

`--help` documents which flags select which path. The hybrid question — CNT
threshold inside a fixed CB background — is `--filler cnt --phi-cb-vol 0.02`.

## 2. Conductivity

```bash
python generate/conductivity.py --phi-cnt-vol 0.02 --cnt-sigma-l-nm 0 \
    --l-rve-nm 1000 --n-real 2
```

```text
 phi target    : CB=0  CNT=0.02  [vol fraction]
 realizations  : 2 ok / 2   (seed base 20260610, workers 1)   n_cb_max=201 n_cnt_max=1474
----------------------------------------------------------------------
 sigma (MC)    : 105.1 +/- 1.5e+02  [S/m]
 sigma median  : 105.1  [p25 52.87, p75 157.4]  [S/m]
 channel power : ss=0 sc=0 cc=1 [median fraction]
 note          : sigma can span orders of magnitude near threshold; median/IQR is the robust locator
```

Here you set the composition directly (`--phi-cnt-vol`, `--phi-cb-vol`) and
the tool solves the Simmons-weighted resistor network. The huge std over two
realizations is honest: near the threshold, conductivity is log-wide, so the
median/IQR line is the robust locator and more realizations are the cure.
The `channel power` line reports the median share of network power carried by
sphere-sphere (ss), sphere-CNT (sc), and CNT-CNT (cc) junctions.
The demo pins `--cnt-sigma-l-nm 0` (monodisperse 500 nm tubes) so the small
1000 nm box stays taller than the longest tube; with the default
polydisperse population, keep `--l-rve-nm` above `--cnt-l-max-nm`.

## 3. Gauge factor

```bash
python generate/gauge_factor.py --phi-cnt-vol 0.02 --cnt-sigma-l-nm 0 \
    --l-rve-nm 1000 --n-real 2
```

```text
 opening alpha : ss=0 sc=0 cc=0 nm/strain  (0 = ideal/affine)
 realizations  : 2 ok / 2   (seed base 20260617, workers 1)
----------------------------------------------------------------------
 GF_r (MC)     : -9.424 +/- 12   [d ln R / d eps]
 decomposition : gf_sigma=-11.14   gf_geom=1.716   (gf_r ~ gf_sigma + gf_geom)
 channel power : eps0 ss=0 sc=0 cc=1; epsmax ss=0 sc=0 cc=1 [median fraction]
```

With the junction-opening coefficients at zero, the network deforms affinely.
In the production $N=300$, $L_{\rm RVE}=4000$ nm reference ensembles,
CNT-bearing ideal networks give an ideal-affine reference GF of about
0.25–0.27, well below the geometric value $1 + 2\nu \approx 1.72$. This
two-realization quickstart demo is noisy and illustrative, not evidential.
Measured GF above the ideal-affine reference is interpreted here as evidence
for non-affine junction opening — dial it in with `--opening-alpha-ss/-sc/-cc`
and compare. The `channel power` line reports the median share of network
power carried by sphere-sphere (ss), sphere-CNT (sc), and CNT-CNT (cc)
junctions at the first and last strain points; this helps diagnose which
junction family controls the GF.

## The knobs that appear everywhere

- **`--seed` / `--seed-base`** — realization `r` uses seed `seed_base + r`.
  Rerunning any command with the same seed reproduces the same numbers
  exactly; changing the seed gives an independent ensemble draw.
- **`--n-real`** — how many realizations enter the ensemble. More
  realizations = tighter error bars; the reported value is a statistic,
  never a single run.
- **`--workers`** — process-parallelism over realizations. Because every
  realization owns its seed, any worker count gives **bit-identical**
  numbers; only the wall time changes. (The monodisperse percolation path
  is the one exception — it keeps its original single RNG stream and says
  so if you pass `--workers`.)
- **`--progress-every`** — terminal progress for longer runs, printed as
  `ok/fail/elapsed/eta` every `N` completed realizations. It changes only
  stdout; use `--progress-every 0` to silence it.
- **`--out summary.csv`** — writes the one-row ensemble summary, including
  the full parameter provenance (seed base, workers, caps), next to your
  terminal output. Without `--out`, nothing is written to disk.

## Where to go next

- `python generate/<tool>.py --help` — every flag, grouped and documented.
- `examples/one_run_anatomy.ipynb` — the engine underneath one realization.
- `examples/pan2013_case_study.md` — a worked external case study with
  measured MWCNT geometry.
- `reproduce_paper/` — the frozen campaign scripts behind the paper figures.
