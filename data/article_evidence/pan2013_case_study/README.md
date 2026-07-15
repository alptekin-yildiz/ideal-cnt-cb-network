# Pan & Li (2013) case-study evidence (manuscript Table 2)

This directory holds the compact evidence chain for the manuscript's Pan & Li
case study. It keeps two thresholds separate:

- $w_{c,g}$: the **geometric** percolation threshold from the ideal CNT network;
- $w_{c,e}$: the **conductivity-defined** threshold from a free power-law fit
  $\sigma = \sigma_0(w - w_{c,e})^t$ to the Simmons-weighted conductivity sweep.

That distinction is the point of the case study. The geometric threshold
answers when a spanning network first exists; the conductivity-defined
threshold answers when that network carries the fitted transport law.

This is a **case study, not a validation anchor**: Pan & Li's measured geometry
and published conductivity-fit values are external literature inputs, while
the model columns are ideal-network references computed from that geometry
with no fitted microstructure parameters. The files live here, in
`data/article_evidence/`, rather than among the regenerated stage anchors in
`data/reference/`.

## Files

| File | Purpose |
|---|---|
| `pan2013_table2_case_study.csv` | one-row-per-type Table 2 support: geometric threshold, model conductivity threshold/exponent, and Pan published threshold/exponent |
| `pan2013_conductivity_sweep_summary.csv` | compact model conductivity sweep summary used for the power-law fit; raw realizations are not shipped |
| `pan2013_conductivity_powerlaw_fit.csv` | fitted model $w_{c,e}$/$t$ rows plus Pan published $w_{c,e}$/$t$ rows |

## Column groups

- `mwcnt_type`, `aspect_ratio`, `measured_l_mean_um`, `measured_d_nm` — Pan &
  Li's TEM geometry after matrix burn-off.
- `pan_wc_e_wt_pct`, `pan_t` — Pan & Li's published conductivity-threshold fit
  values. These are literature values, not regenerated here.
- `pan_phi_c_e_vol` — `pan_wc_e_wt_pct` converted to volume fraction with
  `rho_cnt_g_cm3` / `rho_pp_g_cm3` ($1.75/0.9\ \mathrm{g/cm^3}$):
  $\phi = (w/\rho_{\rm CNT})/(w/\rho_{\rm CNT} + (100-w)/\rho_{\rm PP})$.
- `model_wc_g_*` and `phi_c_g_*` — geometric-threshold ensemble statistics
  from the polydisperse path of `generate/percolation_threshold.py`
  (truncated-lognormal lengths, one RNG seed per realization).
- `model_wc_e_wt_pct`, `model_t`, `model_fit_*` — conductivity-defined
  threshold and exponent from `tools/fit_conductivity_powerlaw.py`.
- remaining columns — the model input provenance: distribution moments and
  support, diameter, waviness, RVE, caps, seeds, and workers. In
  `pan2013_conductivity_sweep_summary.csv`, `campaign_seed_base` is the
  original campaign seed and `row_seed_base` is the seed base to pass to
  `generate/conductivity.py` for that specific row.

For your own study, choose any fixed `--seed-base`, report it, and increase
`--n-real` rather than choosing a favorable seed. To reproduce a shipped
evidence row exactly, use that row's `row_seed_base`; `campaign_seed_base`
records only the original multi-row campaign numbering.

## Regeneration: geometric threshold

The geometric-threshold columns in `pan2013_table2_case_study.csv` are
regenerated from these exact commands. They use the same $L_{\rm RVE}=4000$ nm
and $N=50$ ensemble size as the compact conductivity sweep below, so the
geometric and conductivity-defined thresholds differ by model layer rather than
by box size or sample count. `--workers` changes wall time only, never the
numbers:

```bash
python generate/percolation_threshold.py --filler cnt \
    --cnt-mu-l-nm 660 --cnt-sigma-l-nm 437 --cnt-l-min-nm 100 --cnt-l-max-nm 2500 \
    --cnt-diameter-nm 13 --cnt-waviness 0.7 \
    --rve-nm 4000 --phi-max-vol 0.02 --n-real 50 --seed 42 --workers 12

python generate/percolation_threshold.py --filler cnt \
    --cnt-mu-l-nm 1230 --cnt-sigma-l-nm 555 --cnt-l-min-nm 200 --cnt-l-max-nm 3000 \
    --cnt-diameter-nm 14 --cnt-waviness 0.7 \
    --rve-nm 4000 --phi-max-vol 0.02 --n-real 50 --seed 42 --workers 12

python generate/percolation_threshold.py --filler cnt \
    --cnt-mu-l-nm 1840 --cnt-sigma-l-nm 711 --cnt-l-min-nm 300 --cnt-l-max-nm 4000 \
    --cnt-diameter-nm 11 --cnt-waviness 0.7 \
    --rve-nm 4000 --phi-max-vol 0.01 --n-real 50 --seed 42 --workers 12
```

Add `--out /tmp/pan_ar51.csv`, `--out /tmp/pan_ar84.csv`, or
`--out /tmp/pan_ar167.csv` if you want the one-row generator summaries as
separate files; the shipped CSV adds the literature columns and collects the
three model rows in one compact evidence table.

## Regeneration: conductivity threshold and exponent

The compact sweep summary is a publication-scale table, not a raw archive. It
was distilled from the frozen manuscript conductivity campaign; the geometry,
loading, RVE, tunneling, segmentation, and seed columns recorded in
`pan2013_conductivity_sweep_summary.csv` give the corresponding public
`generate/conductivity.py` recipe. The AR51 3 wt.% pilot row was rerun with
the public generator and matched the archived row: identical `n_ok`/`n_total`,
and median/mean $\sigma$ equal to the last floating-point digit (relative
differences below $5\times10^{-16}$, from summary-statistic accumulation order). The
archived `sigma_s_per_m_std` column is the population standard deviation
(ddof=0) written by the paper campaign, while `generate/conductivity.py`
reports the sample standard deviation (ddof=1); for `n_total=50` the two
conventions differ by the factor $\sqrt{50/49}$. That pilot row is:

```bash
python generate/conductivity.py --phi-cnt-vol 0.015656712090461006 \
    --cnt-mu-l-nm 660 --cnt-sigma-l-nm 437 \
    --cnt-l-min-nm 100 --cnt-l-max-nm 2500 \
    --cnt-diam-nm 13 --cnt-waviness 0.7 \
    --l-rve-nm 4000 --seg-len-nm 50 --tunnel-nm 10 \
    --n-real 50 --seed-base 20262706
```

The full sweep is expensive, so the shipped table keeps the compact summaries
needed for the fit rather than the raw realization CSVs. These publication-scale
50-realization, 4000 nm commands are intended for regeneration audits, not CI
or quick-start use.

The conductivity-defined threshold is then fitted from that compact model
sweep summary:

```bash
python tools/fit_conductivity_powerlaw.py \
    data/article_evidence/pan2013_case_study/pan2013_conductivity_sweep_summary.csv \
    --group-col mwcnt_type --fit-include-col fit_include \
    --min-success-fraction 0.9
```

Expected output:

```text
group  n_fit  x_c  t  sigma0  R2
AR167  6  0.82  2.05096424108  748.701149315  0.999996046967
AR51  4  2.125  2.07427012307  191.408507572  0.999998636386
AR84  5  1.535  2.04436151142  259.277583263  0.999997969896
```

`fit_include=1` records the audited loading window used for the manuscript fit:
positive median conductivity and `n_ok / n_total >= 0.9`. This is deliberate
provenance: a user applying the same tool to their own conductivity data can
change that column to define their own fit window explicitly. The full
50-realization-per-loading conductivity campaign is not run in CI, and raw
realization CSVs are not shipped.

If you fit CSVs written directly by `generate/conductivity.py`, pass that
generator's column names explicitly, for example:

```bash
python tools/fit_conductivity_powerlaw.py my_conductivity_sweep.csv \
    --x-col phi_cnt_vol --sigma-col sigma_median_s_per_m \
    --n-total-col n_real
```

The walkthrough that builds up to this evidence package is
`examples/pan2013_case_study.md`.

Source: Pan, Y.; Li, L. *Polymer* 54(3), 1218-1226 (2013).
https://doi.org/10.1016/j.polymer.2012.12.058
