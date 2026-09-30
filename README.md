# A Debye–Waller Temperature for Computed Diffraction References

This repository contains the data behind the manuscript: first-principles
outputs, the processed tables used for the figures and supporting tables, the
literature Debye–Waller values used as references, and the scripts that
regenerate the tables and figures.

Seven calculations: **As, Sb, Bi** (A7, `hR2`, `R-3m`), **Bi_SOC** (same, with
spin–orbit coupling), and **C, Si, Ge** (`cF8`, diamond).

- **`data/dft_raw/`:** AFLOW/VASP data, one directory per calculation (`.xz`-compressed).
  Inputs: `aflow.in` and the APL/AEL/AGL `aflow_*.in` files, plus `INCAR`,
  `KPOINTS` and `POSCAR` for every stage, including each `ARUN.*` sub-calculation
  (POTCARs are not included). Outputs: phonon DOS and displacements (APL),
  elastic constants (AEL), quasi-harmonic thermodynamics (AGL), and the relaxed
  structure (`CONTCAR.static`). The only `OUTCAR`s are those of the As, Sb and
  Bi relaxation, static and harmonic phonon runs, for the core-hours quoted in
  the Methods.
- **`data/processed/`:** Analysis tables (CSV) for the figures and supporting
  tables: `B(T)` curves, residuals, benchmark metrics, phonon descriptors, and
  the Sb powder-pattern analysis. `computational_cost.csv` holds the As, Sb and
  Bi core-hours quoted in the Methods.
- **`data/literature/`:** Experimental and reference `B(T)` values, one row per
  point, with the cited source and the table or figure each value came from.
  The As/Sb/Bi `B` values were read by hand from figure 3 of Fischer, Sosnowska
  and Szymanski, *J. Phys. C* **11**, 1043 (1978), and their measurement
  temperatures are taken from Table 1 of the same paper; see the file header
  for details.
  The same digitized curves are in `scripts/vendor/dwf_plot_config.py`, which
  the benchmark tables are computed from.

## Scripts

`scripts/` rebuilds the tables in `data/processed/` from `data/dft_raw/`,
rewrites the two `data/literature/` files from the values recorded in their
generators, and renders the figure panels.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/run_all.py --check   # regenerate everything, then compare (about a minute)
```

`run_all.py` runs the table generators in dependency order, then the figure
scripts, and stops at the first failure; `--tables-only` skips the figures.
Scripts can also be run one at a time, e.g.
`python3 scripts/figures/fig05_Bi_phonon_dos.py`. The only ordering constraint
between figure scripts is that `fig04_Sb_pattern.py` must run before
`fig04_panel_b_waterfall.py` and `fig04_panel_c_metrics.py`, which read the
tables it writes.

- Tables overwrite the files in `data/processed/` and `data/literature/`;
  `git status` shows what changed and `git checkout data/` restores them.
- Renders go to `figures/main/` and `figures/si/`, and each script's output to
  `logs/`. Neither directory is tracked.
- Figs. 2–6 and SI Figs. S1–S2 were assembled from these renders by hand, with
  the panel letters added at that stage. SI Fig. S3 was not edited by hand.
  Fig. 1 and the table-of-contents graphic are not script-generated.

With the versions in `requirements.txt`, `--check` reproduces every tracked
table byte for byte. (It also lists four new files, three LaTeX table fragments
and a summary memo, which are not tracked.) Other library versions can change a
few values in the last printed digit, or which member of a symmetry-equivalent
reflection family labels a row of `pattern_error_propagation.csv` and
`Sb_pc4_peaks_table.csv` (e.g. `0,-1,-11` instead of `-1,0,-11`); the numbers
are unaffected.

The figures were rendered with matplotlib 3.8.2, and a re-run with that version
reproduces them pixel for pixel. Later versions lay out text slightly
differently, so labels can move by a few pixels.

## Citation

T. Li, G. Han, X. Xu, J. Hu, and C. Oses,
*A Debye–Waller temperature for computed diffraction references*,
manuscript in preparation.

If you use the values in `data/literature/`, please also cite the original
sources listed there.

## License

Copyright © 2026 Entropy for Energy Lab, Johns Hopkins University.

Released under the MIT License; see `LICENSE`. The MIT terms also cover the
data files in this repository. They do not and cannot cover material owned by
others:

- The experimental Debye–Waller values in `data/literature/` are derived from
  published papers, cited per row. The papers themselves are not redistributed
  here. Reuse of those numbers should cite the original sources listed there,
  not only this repository.
- `data/dft_raw/` holds outputs of AFLOW and VASP runs performed by the
  authors. The outputs are the authors' own; the programs that produced them
  are not included and carry their own licences.
