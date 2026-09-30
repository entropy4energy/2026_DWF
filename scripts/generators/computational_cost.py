"""Core-hours of the As, Sb and Bi standard and harmonic calculations.

Backs the Computational cost paragraph in the Methods and the ratio quoted in
the Introduction: for Sb the two-step relaxation and static calculation took
about 7 core-hours and the harmonic phonon calculation about 270, about 40
times as much; As and Bi give ratios of about 40 and 70. Post-processing only
-- reads the six OUTCARs per material in data/dft_raw/{As,Sb,Bi}/, the only
OUTCARs the package carries (paths.COST_MATERIALS, paths.COST_OUTCARS).

What is counted. core-hours = elapsed wall time x cores / 3600, from the
"running on N total cores" and "Elapsed time (sec)" lines of each OUTCAR.
  standard  relax1, relax2 and static: the AFLOW relax-static protocol that a
            database entry runs whether or not a displacement model is added.
  harmonic  the two displaced 192-atom supercells (A0D0P, A0D0M) and the
            linear-response run for the Born charges and dielectric tensor
            of the polar correction (LRBE).
Not counted: the phonon preparatory relaxation (relax_apl1/2, under 1
core-hour on 8 cores per material), whose OUTCARs are not in the package. It
would change each harmonic total by less than 0.4%.

Why these three. Every run above used 48 cores and the same VASP build, which
the script asserts, so the core-hours compare like with like. Bi_SOC, C, Si
and Ge mixed core counts between stages, and core-hours from different core
counts carry different parallel efficiencies.
"""

import csv
import lzma
import re
import sys
from pathlib import Path

# Every path resolves through scripts/paths.py, the single source of truth for
# this package. Nothing outside the repository root is read or written.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import COST_MATERIALS, COST_OUTCARS, PROCESSED, calc_dir  # noqa: E402

OUT_CSV = PROCESSED / "computational_cost.csv"

STAGE = {
    "OUTCAR.relax1.xz": "relax1",
    "OUTCAR.relax2.xz": "relax2",
    "OUTCAR.static.xz": "static",
    "ARUN.APL_1_A0D0P/OUTCAR.static.xz": "displaced supercell (+)",
    "ARUN.APL_2_A0D0M/OUTCAR.static.xz": "displaced supercell (-)",
    "ARUN.APL_3_LRBE/OUTCAR.static.xz": "Born charges and dielectric tensor",
}

# As printed in the Methods: (material, quantity) -> value, significant figures.
PUBLISHED = {
    ("Sb", "standard"): (7, 1),
    ("Sb", "harmonic"): (270, 2),
    ("Sb", "ratio"): (40, 1),
    ("As", "ratio"): (40, 1),
    ("Bi", "ratio"): (70, 1),
}

CORES = re.compile(r"running on\s+(\d+) total cores")
NIONS = re.compile(r"NIONS\s*=\s*(\d+)")
KPOINTS = re.compile(r"Found\s+(\d+) irreducible k-points")
ELAPSED = re.compile(r"Elapsed time \(sec\):\s*([\d.]+)")


def round_sig(x, figures):
    return float(f"{x:.{figures}g}")


def read_outcar(path):
    """Return the VASP version line, cores, ions, irreducible k-points and
    elapsed seconds of one OUTCAR, each of which must appear exactly once
    (k-points: the first report)."""
    with lzma.open(path, "rt", errors="replace") as handle:
        text = handle.read()
    version = text.splitlines()[0].strip()
    found = {}
    for key, pattern in (("cores", CORES), ("nions", NIONS), ("elapsed", ELAPSED)):
        hits = pattern.findall(text)
        if len(hits) != 1:
            raise ValueError(f"{path}: expected one {key} line, found {len(hits)}")
        found[key] = hits[0]
    kpoints = KPOINTS.findall(text)
    if not kpoints:
        raise ValueError(f"{path}: no irreducible k-point count")
    return (version, int(found["cores"]), int(found["nions"]), int(kpoints[0]),
            float(found["elapsed"]))


def main():
    if set(STAGE) != set(COST_OUTCARS):
        raise ValueError("STAGE labels and paths.COST_OUTCARS disagree")

    rows, versions, cores = [], set(), set()
    for material in COST_MATERIALS:
        directory = calc_dir(material)
        for name in COST_OUTCARS:
            version, n_cores, nions, kpoints, elapsed = read_outcar(directory / name)
            versions.add(version)
            cores.add(n_cores)
            rows.append({
                "material": material,
                "route": "harmonic" if name.startswith("ARUN.APL_") else "standard",
                "stage": STAGE[name],
                "outcar": name,
                "nions": nions,
                "irreducible_kpoints": kpoints,
                "cores": n_cores,
                "elapsed_s": f"{elapsed:.3f}",
                "core_hours": f"{elapsed * n_cores / 3600:.6g}",
            })
    if len(versions) != 1 or len(cores) != 1:
        raise ValueError(f"runs are not comparable: builds {sorted(versions)}, "
                         f"cores {sorted(cores)}")

    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total = {}
    print(f"{cores.pop()} cores, {versions.pop()}")
    for material in COST_MATERIALS:
        own = [r for r in rows if r["material"] == material]
        for route in ("standard", "harmonic"):
            total[material, route] = sum(float(r["core_hours"]) for r in own
                                         if r["route"] == route)
        total[material, "ratio"] = total[material, "harmonic"] / total[material, "standard"]
        print(material)
        for row in own:
            print(f"  {row['route']:<9} {row['stage']:<35} {row['nions']:>4} ions "
                  f"{row['irreducible_kpoints']:>4} k  "
                  f"{float(row['core_hours']):8.2f} core-h")
        print(f"  standard total {total[material, 'standard']:.2f} core-h, "
              f"harmonic total {total[material, 'harmonic']:.2f} core-h, "
              f"ratio {total[material, 'ratio']:.1f}")

    for key, (value, figures) in PUBLISHED.items():
        if round_sig(total[key], figures) != value:
            raise ValueError(f"{key} = {total[key]:.3g} no longer rounds to the "
                             f"published {value}")
    print("Matches the Methods: Sb about 7 and 270 core-hours, about 40 times; "
          "As and Bi about 40 and 70 times")
    print(f"Wrote {OUT_CSV.relative_to(PROCESSED.parents[1])}")


if __name__ == "__main__":
    main()
