"""Theta_DWF sensitivity to the low-frequency cutoff of the tabulated phonon DOS.

Reads only the existing APL phonon DOS files (no new DFT). For each material the
normalized DOS is restricted to f >= f_min, renormalized on that range, and
Theta_DWF = h*sqrt(3/<f^-2>)/kB is recomputed.
"""
import lzma
import math
import os
import sys
from pathlib import Path

import numpy as np
TRAPZ = getattr(np, "trapezoid", None) or np.trapz

# Every path resolves through scripts/paths.py, the single source of truth
# for this package. Nothing outside the repository root is read or written.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DFT_RAW, FIGURES_SI, LITERATURE, PROCESSED, calc_dir  # noqa: E402
BASES = [DFT_RAW]

H = 6.62607015e-34
KB = 1.380649e-23
THZ = 1e12

REL = {
    "As":     "As",
    "Sb":     "Sb",
    "Bi":     "Bi",
    "Bi_SOC": "Bi_SOC",
    "C":      "C",
    "Si":     "Si",
    "Ge":     "Ge",
}

# Scalar Debye temperatures used in the manuscript (K). emp: Fischer (1978) for
# As, Sb and Bi, Kittel for C, Si and Ge; ael and agl: the elastic-derived and
# quasi-harmonic values.
THETA = {
    "As":     dict(emp=229.0, ael=249.135, agl=334.878),
    "Sb":     dict(emp=204.0, ael=206.693, agl=247.915),
    "Bi":     dict(emp=112.0, ael=115.1,   agl=137.969),
    "Bi_SOC": dict(emp=None,  ael=101.418, agl=117.056),
    "C":      dict(emp=2230.0, ael=2224.75, agl=2098.68),
    "Si":     dict(emp=645.0,  ael=625.371, agl=611.088),
    "Ge":     dict(emp=374.0,  ael=331.699, agl=324.12),
}

CUTS = [0.0, 0.05, 0.10, 0.20, 0.50]


def path_for(rel):
    for b in BASES:
        p = b / rel / "aflow.apl.phonon_dos.out.xz"
        if p.exists():
            return p
    return None


def read_raw(path):
    f, d = [], []
    with lzma.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 4:
                continue
            try:
                f.append(float(p[0])); d.append(float(p[3]))
            except ValueError:
                continue
    f = np.array(f); d = np.array(d)
    return f, d


def theta_dwf(f, d, f_min):
    m = f > max(f_min, 0.0) if f_min > 0 else f > 0
    if f_min > 0:
        m = f >= f_min
    ff, dd = f[m], d[m]
    g = dd / TRAPZ(dd, ff)
    inv2 = TRAPZ(g / ff**2, ff)
    f_dwf = math.sqrt(3.0 / inv2)          # THz
    return H * f_dwf * THZ / KB, inv2


print(f"{'mat':8s} {'f_min_tab':>9s} {'df':>6s} {'neg_wt%':>8s} | " +
      " ".join(f"{c:>8.2f}" for c in CUTS))
rows = {}
for mat, rel in REL.items():
    p = path_for(rel)
    if p is None:
        print(f"{mat:8s} MISSING", file=sys.stderr)
        continue
    f, d = read_raw(p)
    neg = d[f < 0]
    negf = f[f < 0]
    negwt = 0.0
    if len(negf) > 1:
        negwt = abs(TRAPZ(np.abs(neg), negf)) / abs(TRAPZ(d[f > 0], f[f > 0])) * 100
    pos = f[f > 0]
    df = np.median(np.diff(pos))
    vals = []
    for c in CUTS:
        th, _ = theta_dwf(f, d, c)
        vals.append(th)
    rows[mat] = vals
    print(f"{mat:8s} {pos.min():9.4f} {df:6.4f} {negwt:8.3f} | " +
          " ".join(f"{v:8.1f}" for v in vals))

print("\n--- percent change in Theta_DWF relative to f_min=0 ---")
print(f"{'mat':8s} " + " ".join(f"{c:>8.2f}" for c in CUTS))
for mat, vals in rows.items():
    base = vals[0]
    print(f"{mat:8s} " + " ".join(f"{100*(v-base)/base:+8.2f}" for v in vals))

print("\n--- Theta_DWF/Theta_D ratios at f_min = 0 and 0.10 THz ---")
print(f"{'mat':8s} {'emp0':>7s} {'emp.1':>7s} {'ael0':>7s} {'ael.1':>7s} "
      f"{'agl0':>7s} {'agl.1':>7s}")
for mat, vals in rows.items():
    t = THETA[mat]
    out = [mat.ljust(8)]
    for key in ("emp", "ael", "agl"):
        if t[key] is None:
            out += ["      -", "      -"]
        else:
            out += [f"{vals[0]/t[key]:7.3f}", f"{vals[2]/t[key]:7.3f}"]
    print(" ".join(out))
