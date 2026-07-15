# reproduce_paper

The exact wrapper scripts that generated the paper's figures, kept for
provenance and transparency. They are deliberately self-contained — there is
no shared helper module between them — so each script is the complete, frozen
recipe of one figure, and a change to one can never silently alter another.
The frozen manuscript scripts are regression-checked against the reusable
engines at the intended equivalence points (see
`tests/test_percolation_distribution.py`); they are not independent scientific
models.
Random seeds are derived deterministically
from a fixed `--seed-base`, so re-running a wrapper regenerates its raw
per-realization data exactly; the curated summary tables they produce are
shipped in `data/reference/` and frozen with the code in the Zenodo snapshot.
The compact `data/article_evidence/gf_mechanism/` package separately preserves
the numerical support for the manuscript's Fig. 10/Table 3 mechanism discussion.
Environment used for the paper runs: see the note at the end of this file.

Each wrapper is a heavy Monte Carlo sweep — much larger than the `--quick`
demos in `generate/`. Outputs are written under `data/processed/` (created on
first run, git-ignored) in subfolders matching the scientific stage:
`percolation/`, `conductivity/`, and `gf/`. Run any wrapper with `--help` for
its parameters. Long wrappers accept `--progress-every N` to print terminal
progress (`ok/fail/elapsed/eta`) during a Monte Carlo point; this changes only
stdout, not seeds, CSV schemas, or numerical outputs.

## Two-stage pipeline

1. **Percolation thresholds (self-contained)** — run directly:
   `run_cnt_phic_sweep_fixedstep.py`, `run_cb_phic_sweep_pub.py`,
   `run_hybrid_phic_boundary_pub.py`, `run_fss_monodisperse_benchmark.py`.
   These emit the $\phi_c$ tables under `data/processed/percolation/`
   (e.g. `cnt_phic_sweep_fixedstep_Lmin50.csv` and
   `cb_phic_sweep_Loverd60_momentmatched_D20_500_n300.csv`). The CNT and CB
   distribution sweeps use tagged production filenames by default so the
   generated paths match the shipped references.

2. **Conductivity & gauge factor** —
   `run_conductivity_universality_pub.py`,
   `run_simmons_conductivity_regime_pub.py`, `run_gf_simmons_regime_pub.py`.
   They set $\phi$ targets as multiples of the percolation threshold, so they read
   the stage-1 $\phi_c$ tables from `data/processed/percolation/` and write their
   own summaries under `data/processed/conductivity/` or `data/processed/gf/`.

To run stage 2 or redraw figures without re-running stage 1 first, drop the
shipped compact reference summaries into the generated-output tree:

```bash
mkdir -p data/processed/percolation data/processed/conductivity data/processed/gf
cp data/reference/percolation/*.csv data/processed/percolation/
cp data/reference/conductivity/*.csv data/processed/conductivity/
cp data/reference/gf/*.csv data/processed/gf/
```

> Just exploring with your own parameters? Use `generate/` instead — it is
> self-contained and needs no tables.

## Exact reference commands

Strict reference regeneration should use the commands below, not values copied
back out of the rounded CSV display columns. Some transport targets are stored
as multiples of a reference threshold; the printed `phi_multiplier` and
`phi_*_target_vol` values are rounded provenance fields, and recomputing one
from the other can shift the last displayed digit. The `--workers` value may be
changed to match local hardware; it affects wall time, not the seeded
realization sequence or the reported statistics. Distribution and transport
wrappers assign one deterministic seed to each realization and restore results
to realization order before writing summaries; the public tests include
worker-parity checks for these reusable engines and CLIs. Some percolation
wrappers may cap the requested worker count for memory, so `--workers 10` below
is only a conservative example, not part of the scientific parameter set. Use
more workers, such as 12, when local memory allows it; the RVE size, realization
count, seeds, and geometry flags are the load-bearing settings. Run these
commands in a clean `data/processed/` tree, or use `--overwrite` where the
wrapper supports it.

Percolation references:

```bash
python reproduce_paper/run_fss_monodisperse_benchmark.py \
    --only all --workers 10 --progress-every 10

python reproduce_paper/run_cnt_phic_sweep_fixedstep.py \
    --tag Lmin50 --n-real 100 --workers 10 --progress-every 10

python reproduce_paper/run_cb_phic_sweep_pub.py \
    --tag D20_500_n300 --n-real 300 --workers 10 --progress-every 10

python reproduce_paper/run_hybrid_phic_boundary_pub.py \
    --tag n400_modes --n-real 400 --workers 10 --progress-every 10 \
    --overwrite
```

Simmons conductivity references:

```bash
python reproduce_paper/run_simmons_conductivity_regime_pub.py \
    --tag regime_n300_m10_Lcb8000_Lcnt4000_Lhyb4000 \
    --systems cb,cnt,hybrid \
    --phi-multipliers 0.7,0.85,0.95,1,1.05,1.15,1.3,1.5,1.75,2 \
    --n-real 300 --workers 10 \
    --l-rve-cb-nm 8000 --l-rve-cnt-nm 4000 --l-rve-hybrid-nm 4000 \
    --overwrite

python reproduce_paper/run_simmons_conductivity_regime_pub.py \
    --tag regime_n300_cnt_high_to_3wt_L4000 --systems cnt \
    --phi-multipliers 2.25,2.5,2.75,3,3.25,3.402 \
    --n-real 300 --workers 10 --l-rve-cnt-nm 4000 --overwrite

python reproduce_paper/run_simmons_conductivity_regime_pub.py \
    --tag regime_n300_m17_hybridCB1_Lhyb4000 --systems hybrid \
    --hybrid-cb-wt 1 \
    --phi-multipliers 0.7,0.85,0.95,1,1.05,1.15,1.3,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.578 \
    --n-real 300 --workers 10 --l-rve-hybrid-nm 4000 --overwrite
```

Affine gauge-factor references:

```bash
python reproduce_paper/run_gf_simmons_regime_pub.py \
    --tag gf_affine_cb_40wt_50wt_n300 --systems cb \
    --phi-multipliers 1.0155526592606384,1.313271692092845 \
    --n-real 300 --workers 10 --l-rve-cb-nm 8000 --progress-every 10 \
    --overwrite

python reproduce_paper/run_gf_simmons_regime_pub.py \
    --tag gf_affine_cnt_3wt_n300 --systems cnt \
    --phi-multipliers 3.402 \
    --n-real 300 --workers 10 --l-rve-cnt-nm 4000 --progress-every 10 \
    --overwrite

python reproduce_paper/run_gf_simmons_regime_pub.py \
    --tag gf_affine_hybrid_3CNT_1CB_n300 --systems hybrid \
    --hybrid-cb-wt 1 --phi-multipliers 3.578 \
    --n-real 300 --workers 10 --l-rve-hybrid-nm 4000 --progress-every 10 \
    --overwrite
```

## Verify regenerated summary tables

After regenerating the paper summary tables under `data/processed/`, compare
them against the frozen references before changing or removing code:

```bash
python tools/verify_reference_outputs.py \
    --reference-dir data/reference \
    --generated-dir data/processed
```

Use `--allow-missing` for a partial check while only some long-running wrappers
have been regenerated. The comparison preserves relative paths, so
`data/reference/percolation/*.csv` is checked against
`data/processed/percolation/*.csv`, and likewise for `conductivity/` and `gf/`.
It ignores runtime columns such as `elapsed_s` by default. This tool checks
the `percolation/`, `conductivity/`, and `gf/` stage summaries; the derived
evidence tables in `data/article_evidence/` are intentionally outside this
one-to-one comparison.

The GF mechanism/sign-flip support package used by Fig. 10 and Table 3 is a
compact derived audit set in `data/article_evidence/gf_mechanism/`. Check it
with:

```bash
python tools/verify_gf_mechanism_support.py
```

## Environment (paper runs)

The production sweeps behind the paper's figures were run with:

- Python 3.11.1
- NumPy 2.2.4
- SciPy 1.15.2

Any environment satisfying `pyproject.toml` (`numpy>=1.24`, `scipy>=1.10`)
reproduces the same ensemble statistics; the exact versions above matter only
for bit-level identity of individual realizations.
