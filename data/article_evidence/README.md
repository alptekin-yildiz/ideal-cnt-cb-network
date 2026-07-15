# data/article_evidence

Compact evidence chain for specific claims in the companion manuscript. This
directory is intentionally separate from `data/reference/`, which holds only
the main stage summaries that current wrappers regenerate one-to-one. The
files here are derived support tables and provenance metadata — they document
manuscript claims rather than anchor regenerated outputs.

Path-like CSV fields use repository-root-relative paths (for example
`data/reference/...` or `data/article_evidence/...`) so the tables can be read
consistently from any working directory.

| Item | What it holds |
|---|---|
| `manuscript_inventory.csv` | Figure/table provenance inventory mapping each main-manuscript item to its public support artifacts |
| `gf_mechanism/` | Compact numerical support for the manuscript's GF mechanism/sign-flip discussion (Fig. 10, Table 3); see `gf_mechanism/README.md` |
| `pan2013_case_study/` | Compact Table 2 evidence for the Pan & Li 2013 MWCNT/PP case study: geometric thresholds, conductivity-defined threshold/exponent fit, and literature comparison rows; see `pan2013_case_study/README.md` |

Check the GF mechanism package with:

```bash
python tools/verify_gf_mechanism_support.py
```

The check cross-derives the machine-readable Table 3 from the audit table,
the pure-CNT support table, and the main affine GF references in
`data/reference/gf/`, so the evidence chain stays internally consistent.
