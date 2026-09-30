#!/usr/bin/env python3
"""Regenerate everything in this package, in dependency order.

    python3 scripts/run_all.py                 # tables + figures
    python3 scripts/run_all.py --tables-only    # skip the figure scripts
    python3 scripts/run_all.py --check          # regenerate, then diff against
                                                # the shipped files

Stages run in order and the run stops at the first failure. It does not continue
past a broken stage, because every later table would then be built from a mix of
fresh and stale inputs.

Stage order is a real dependency order, not alphabetical:

  1. rebuild_dwf_revision_package.py   the seven master tables. Reads
                                       data/dft_raw/ and
                                       scripts/vendor/dwf_plot_config.py.
                                       Everything downstream consumes its output.
  2. export_figure_wide_csvs.py        wide-form derivatives of the master tables
  3. bootstrap_ci.py                   bootstrap CIs (fixed seed)
  4. {C,Si,Ge,Bi_SOC}_phonon_analysis  per-material descriptors and curves.
                                       Si first: C reads Si's descriptors, Ge
                                       reads both C's and Si's.
  5. A7 / cF8 literature archives      provenance tables
  6. cross_material_synthesis.py       cross-material tables + synthesis memo.
                                       Reads everything above.
  7. SI Tables S4-S5, main Table III   post-processing only
  8. computational_cost.py             As/Sb/Bi core-hours for the Methods,
                                       from the OUTCARs in data/dft_raw/
  9. figure scripts                    fig04 panel a first: panels b and c read
                                       the intermediate tables it writes
 10. SI verification + SI figures
"""

from __future__ import annotations

import argparse
import filecmp
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from paths import PROCESSED, REPO_ROOT, VERIFICATION_LOGS, ensure_output_dirs  # noqa: E402

GENERATORS = [
    "rebuild_dwf_revision_package.py",
    "export_figure_wide_csvs.py",
    "bootstrap_ci.py",
    # the three cF8 analyses read the literature archive this one writes
    "cF8_literature_sources.py",
    "Si_phonon_analysis.py",
    "C_phonon_analysis.py",
    "Ge_phonon_analysis.py",
    "Bi_SOC_phonon_analysis.py",
    "A7_literature_sources.py",
    "cross_material_synthesis.py",
    "frequency_window_sensitivity.py",
    "frequency_mesh_convergence.py",
    "pattern_error_propagation.py",
    "computational_cost.py",
]

FIGURES = [
    "fig04_Sb_pattern.py",
    "fig04_panel_b_waterfall.py",
    "fig04_panel_c_metrics.py",
    "fig02_AsSbBi_comparison.py",
    "fig03_summary_heatmaps.py",
    "fig05_Bi_phonon_dos.py",
    "fig06_Bi_SOC_sensitivity.py",
]

SI = [
    "verify_debye_mechanism.py",
    "theta_dwf_cutoff_sensitivity.py",
    "make_SI_figures.py",
    "make_unification_figure_grid.py",
]


def run(subdir: str, script: str) -> bool:
    path = BASE_DIR / subdir / script
    log_path = VERIFICATION_LOGS / f"{path.stem}.log"
    print(f"  {subdir}/{script} ... ", end="", flush=True)
    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode == 0:
        print("ok")
        return True
    print("FAILED")
    print(f"    log: {log_path}")
    tail = (result.stdout + result.stderr).strip().splitlines()[-12:]
    for line in tail:
        print(f"    {line}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-only", action="store_true",
                        help="run the generators, skip figures and SI figures")
    parser.add_argument("--check", action="store_true",
                        help="after regenerating, diff data/processed against a "
                             "snapshot taken before the run")
    args = parser.parse_args()

    ensure_output_dirs()

    snapshot: dict[str, bytes] = {}
    if args.check:
        for path in sorted(PROCESSED.iterdir()):
            if path.is_file():
                snapshot[path.name] = path.read_bytes()
        print(f"Snapshot: {len(snapshot)} files in data/processed/\n")

    stages: list[tuple[str, str, list[str]]] = [("Tables", "generators", GENERATORS)]
    if not args.tables_only:
        stages += [("Figures", "figures", FIGURES), ("SI", "si", SI)]

    for title, subdir, scripts in stages:
        print(f"{title}:")
        for script in scripts:
            if not run(subdir, script):
                print(f"\nStopped in stage {title!r} at {script}.")
                return 1
        print()

    if args.check:
        changed, missing = [], []
        for name, before in snapshot.items():
            path = PROCESSED / name
            if not path.exists():
                missing.append(name)
            elif path.read_bytes() != before:
                changed.append(name)
        new = sorted({p.name for p in PROCESSED.iterdir() if p.is_file()} - set(snapshot))
        print("Byte-for-byte check against the pre-run snapshot:")
        print(f"  unchanged : {len(snapshot) - len(changed) - len(missing)}")
        print(f"  changed   : {len(changed)}" + (f" -> {', '.join(changed)}" if changed else ""))
        print(f"  missing   : {len(missing)}" + (f" -> {', '.join(missing)}" if missing else ""))
        print(f"  new       : {len(new)}" + (f" -> {', '.join(new)}" if new else ""))
        if changed or missing:
            print("\nWith library versions other than those in requirements.txt, "
                  "see README.md (Scripts) for the differences to expect.")
            return 2

    print("Done. Tables in data/processed/, figures in figures/main and "
          "figures/si, logs in logs/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
