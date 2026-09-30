"""Single source of truth for every path used by the DWF data package.

Every script in this package resolves its inputs and outputs through this
module.  Nothing outside ``REPO_ROOT`` is ever read or written, which is what
makes the package runnable from a fresh clone in any location.

Usage from a script in ``scripts/<subdir>/``::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paths import PROCESSED, calc_dir

Set ``DWF_PACKAGE_ROOT`` to relocate the whole tree (used by the clean-room
verification run).  ``PROJECT_DWF_ROOT`` is accepted as a legacy alias so
scripts carried over from the manuscript project keep working unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "DATA",
    "DFT_RAW",
    "PROCESSED",
    "LITERATURE",
    "FIGURES",
    "FIGURES_MAIN",
    "FIGURES_SI",
    "VERIFICATION_LOGS",
    "SCRIPTS",
    "VENDOR",
    "MATERIALS",
    "RAW_FILES",
    "COST_MATERIALS",
    "COST_OUTCARS",
    "calc_dir",
    "processed",
    "literature",
    "ensure_output_dirs",
]


def _resolve_root() -> Path:
    for var in ("DWF_PACKAGE_ROOT", "PROJECT_DWF_ROOT"):
        value = os.environ.get(var)
        if value:
            return Path(value).expanduser().resolve()
    # This file lives at <repo>/scripts/paths.py.
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _resolve_root()

DATA = REPO_ROOT / "data"
DFT_RAW = DATA / "dft_raw"
PROCESSED = DATA / "processed"
LITERATURE = DATA / "literature"

# Generated outputs. Neither directory is tracked (see .gitignore): the figures
# are rebuilt by the scripts, and the logs hold each script's output.
FIGURES = REPO_ROOT / "figures"
FIGURES_MAIN = FIGURES / "main"
FIGURES_SI = FIGURES / "si"

VERIFICATION_LOGS = REPO_ROOT / "logs"

SCRIPTS = REPO_ROOT / "scripts"
VENDOR = SCRIPTS / "vendor"

#: The seven first-principles calculations, in the order used by every table.
MATERIALS: tuple[str, ...] = ("As", "Sb", "Bi", "Bi_SOC", "C", "Si", "Ge")

#: The AFLOW output files the analysis scripts open in
#: ``data/dft_raw/<material>/``. The directories also carry the AFLOW/VASP
#: inputs, which no script reads, and each of ``COST_MATERIALS`` carries
#: ``COST_OUTCARS``.
RAW_FILES: tuple[str, ...] = (
    "PHDOSCAR.xz",
    "aflow.apl.phonon_dos.out.xz",
    "CONTCAR.static.xz",
    "aflow.apl.displacements.out.xz",
    "AGL_THERMO.xz",
    "aflow.agl.out.xz",
    "aflow.ael.out.xz",
    "aflow.apl.out.xz",
    "AEL_Elastic_constants.out.xz",
    "AEL_Compliance_tensor.out.xz",
    "AGL_thermal_properties_temperature.out.xz",
)

#: The only VASP OUTCARs in the package, relative to
#: ``data/dft_raw/<material>/`` for each of ``COST_MATERIALS``: the standard
#: relax-static runs and the harmonic phonon runs whose timings
#: ``generators/computational_cost.py`` reads for the Methods.
COST_MATERIALS: tuple[str, ...] = ("As", "Sb", "Bi")
COST_OUTCARS: tuple[str, ...] = (
    "OUTCAR.relax1.xz",
    "OUTCAR.relax2.xz",
    "OUTCAR.static.xz",
    "ARUN.APL_1_A0D0P/OUTCAR.static.xz",
    "ARUN.APL_2_A0D0M/OUTCAR.static.xz",
    "ARUN.APL_3_LRBE/OUTCAR.static.xz",
)


def calc_dir(material: str) -> Path:
    """Return the raw-calculation directory for ``material``.

    The original AFLOW trees used four different nesting patterns and embedded
    ``:`` in directory names (illegal on Windows).  They are flattened here to
    one level per material, so no layout probing is needed.
    """
    if material not in MATERIALS:
        raise KeyError(
            f"Unknown material {material!r}; expected one of {', '.join(MATERIALS)}"
        )
    path = DFT_RAW / material
    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing raw-calculation directory for {material}: {path}\n"
            f"Expected it at <repo>/data/dft_raw/{material}/ "
            f"(repo root resolved to {REPO_ROOT})."
        )
    return path


def processed(name: str) -> Path:
    """Return the path to a table in ``data/processed/``, checking existence."""
    path = PROCESSED / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing processed table: {path}\n"
            f"Regenerate it with scripts/run_all.py "
            f"(repo root resolved to {REPO_ROOT})."
        )
    return path


def literature(name: str) -> Path:
    """Return the path to a table in ``data/literature/``, checking existence."""
    path = LITERATURE / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing literature table: {path}\n"
            f"(repo root resolved to {REPO_ROOT})."
        )
    return path


def ensure_output_dirs() -> None:
    """Create the directories scripts write into.  Safe to call repeatedly."""
    for path in (PROCESSED, FIGURES_MAIN, FIGURES_SI, VERIFICATION_LOGS):
        path.mkdir(parents=True, exist_ok=True)
