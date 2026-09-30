#!/usr/bin/env python3
"""SI cross-system moment comparison: first-principles 2x3 grid.

Repeats the same two-panel moment-ratio logic for the THREE first-principles
Debye temperatures requested for the materials-discovery framing:

  column 1: AEL -- first-principles elastic Theta_D
  column 2: AGL -- first-principles quasi-harmonic Theta_D
  column 3: harmonic phonon -- Debye temperature from the harmonic phonon DOS

Top row    : Theta_DWF / Theta_D bar chart, grouped by family.
Bottom row : numerical self-consistency of the high-T ratio B_scalar/B_DOS-only
             (at T=2 Theta_D) against the classical prediction
             (Theta_DWF/Theta_D)^2.

The empirical/literature Theta_D pair from the earlier Fig. S3 is now discussed
in the SI text/caption rather than occupying a panel.

Theta_DWF is fixed by the phonon DOS and is compared against three independently
computed first-principles scalar Debye scales. The bottom panels check the
classical square-law consequence of that definition for AEL, AGL, and the
harmonic phonon Debye scale. In the harmonic phonon column, diamond C is the stiff limiting case
where the phonon-DOS Theta_D is slightly below Theta_DWF, so the ratio is just
above 1; this is a definition-specific edge case, not a failure of the
square-law descriptor.
"""
from __future__ import annotations
import lzma, math
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# numpy 2.0 removed trapz. The getattr() form used previously evaluated
# np.trapz eagerly as the default argument, so it raised AttributeError on
# numpy >= 2.0 even though trapezoid was present. Same function either way.
TRAPEZOID = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

from matplotlib import font_manager

# Arial, matching the manuscript figures. Registering the file makes the lookup
# explicit instead of relying on matplotlib's platform font scan, which is what
# keeps this reproducible off this machine. mathtext is pointed at the same
# family because it otherwise defaults to dejavusans, which would render every
# symbol in a different typeface from the surrounding text.
ARIAL_FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
if ARIAL_FONT_PATH.exists():
    font_manager.fontManager.addfont(str(ARIAL_FONT_PATH))
    ARIAL_FONT = font_manager.FontProperties(fname=str(ARIAL_FONT_PATH)).get_name()
else:
    ARIAL_FONT = "Arial"

mpl.rcParams.update({
    "font.family": ARIAL_FONT,
    "font.sans-serif": [ARIAL_FONT],
    "mathtext.fontset": "custom",
    "mathtext.rm": ARIAL_FONT,
    "mathtext.it": f"{ARIAL_FONT}:italic",
    "mathtext.bf": f"{ARIAL_FONT}:bold",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})

# Every path resolves through scripts/paths.py, the single source of truth
# for this package. Nothing outside the repository root is read or written.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DFT_RAW, FIGURES_SI, LITERATURE, PROCESSED, calc_dir, ensure_output_dirs  # noqa: E402
BASE_CANDIDATES = [DFT_RAW]
OUT = FIGURES_SI

H = 6.62607015e-34; HBAR = H / (2 * math.pi); KB = 1.380649e-23
AMU = 1.66053906660e-27; THZ = 1e12; A2 = 1e20; EIGHT_PI2 = 8 * math.pi**2

# Two restrained family palettes: one neutral (A7), one signal (cF8).
COL_A7, COL_CF8 = "#c2410c", "#1d4ed8"


def _dos(rel):
    for base in BASE_CANDIDATES:
        path = base / rel / "aflow.apl.phonon_dos.out.xz"
        if path.exists():
            return path
    return BASE_CANDIDATES[0] / rel / "aflow.apl.phonon_dos.out.xz"

MAT = {
    "As": dict(fam="A7",  dos=_dos("As"),
               mass=74.921595),
    "Sb": dict(fam="A7",  dos=_dos("Sb"),
               mass=121.760),
    "Bi": dict(fam="A7",  dos=_dos("Bi"),
               mass=208.980),
    "C":  dict(fam="cF8", dos=_dos("C"),
               mass=12.011),
    "Si": dict(fam="cF8", dos=_dos("Si"),
               mass=28.085),
    "Ge": dict(fam="cF8", dos=_dos("Ge"),
               mass=72.63),
}
ORDER = ["As", "Sb", "Bi", "C", "Si", "Ge"]

# First-principles scalar reference Debye temperatures (K). All verified against the CSVs /
# aflow outputs and verification_A7_output.txt:
#   ael : first-principles elastic Debye temperature (aflow.ael ael_debye_temperature).
#   agl : first-principles quasi-harmonic Debye temperature (aflow.agl agl_debye).
#   phonon : computed below from the harmonic phonon DOS by matching <f^2>.
THETA_D = {
    "ael": dict(As=249.135, Sb=206.693, Bi=115.1,  C=2224.75, Si=625.371, Ge=331.699),
    "agl": dict(As=334.878, Sb=247.915, Bi=137.969, C=2098.68, Si=611.088, Ge=324.12),
}
# Column order and labels for the three definitions. Labels use the manuscript
# treatment names and Theta_D symbols (NOT the AFLOW module names AEL/AGL):
#   ael -> elastic-property-derived Debye model, Theta_D^elastic
#   agl -> quasi-harmonic Debye model,           Theta_D^QH
#   phonon -> harmonic phonon Debye temperature, Theta_D^phonon
# title_def : short phrase for panel titles; sym : the Theta_D symbol for axes.
DEFS = [
    ("ael", "elastic-derived", r"$\Theta_D^{\mathrm{elastic}}$"),
    ("agl", "quasi-harmonic", r"$\Theta_D^{\mathrm{QH}}$"),
    ("phonon", "harmonic phonon", r"$\Theta_D^{\mathrm{phonon}}$"),
]

# Each material is evaluated at T = CLASSICAL_FACTOR * Theta_D so all six reach the
# classical regime (a single fixed T cannot, given the first-principles Theta_D
# span 115-2225 K). At 2*Theta_D the square law holds to <0.4% for every material
# and every definition.
CLASSICAL_FACTOR = 2.0

# Shared axis limits so the three columns are directly comparable.
BAR_YLIM = (0.0, 1.15)
PAR_LIM = (0.40, 1.12)


def read_dos(path):
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
    m = f > 0; f = f[m]; d = d[m]
    return f, d / TRAPEZOID(d, f)


def theta_dwf(f, g):
    """Moment temperature from the inverse-square phonon-DOS moment."""
    return H * math.sqrt(3.0 / TRAPEZOID(g / f**2, f)) * THZ / KB


def theta_harmonic_phonon(f, g):
    """Harmonic-phonon Debye temperature from the DOS mean-square frequency.

    For an ideal Debye DOS normalized to INT g(f)df = 1,
    <f^2> = INT g(f) f^2 df = (3/5) f_D^2.
    """
    return H * math.sqrt((5.0 / 3.0) * TRAPEZOID(g * f**2, f)) * THZ / KB


def b_dosonly(f, g, T, mass_amu):
    w = 2 * math.pi * f * THZ
    x = HBAR * w / (2 * KB * T)
    val = TRAPEZOID(g / w * (1.0 / np.tanh(x)), f)
    return EIGHT_PI2 * A2 * HBAR / (2 * mass_amu * AMU) * val


def debye_phi(x):
    t = np.linspace(1e-8, x, 4000)
    return TRAPEZOID(t / np.expm1(t), t) / x


def b_debye(T, theta, mass_amu):
    pref = 6.0 * H**2 / (mass_amu * AMU * KB * theta)
    x = theta / T
    return pref * (debye_phi(x) / x + 0.25) * A2


def compute(defn):
    """Evaluate the classical-limit scalar-to-DOS ratio for each material at its OWN
    high temperature, T = CLASSICAL_FACTOR * Theta_D, for the given Theta_D
    definition. Theta_DWF (numerator) is fixed by the DOS and does not depend
    on the definition; only Theta_D and the evaluation temperature do."""
    theta_d = THETA_D.get(defn, {})
    rows = []
    for m in ORDER:
        c = MAT[m]
        f, g = read_dos(c["dos"])
        theta_dwf_value = theta_dwf(f, g)
        thD = theta_harmonic_phonon(f, g) if defn == "phonon" else theta_d[m]
        r = theta_dwf_value / thD
        T_hi = CLASSICAL_FACTOR * thD
        bsc = b_debye(T_hi, thD, c["mass"])
        bdos = b_dosonly(f, g, T_hi, c["mass"])
        rows.append(dict(mat=m, fam=c["fam"], theta_DWF=theta_dwf_value, theta_D=thD,
                         ratio=r, ratio2=r**2, obs=bsc / bdos, T_hi=T_hi))
    return rows


# Per-material label placement for the parity panels. Offsets are definition-
# specific because the near-coincident pairs change from AEL to AGL to phonon.
PAR_LBL = {
    "ael": {
        "As": (0.018, 0.012, "left", "bottom"),
        "Sb": (-0.014, -0.024, "right", "top"),
        "Bi": (0.014, -0.018, "left", "top"),
        "C":  (-0.014, -0.014, "right", "top"),
        "Si": (0.0, 0.024, "center", "bottom"),
        "Ge": (0.016, -0.006, "left", "top"),
    },
    "agl": {
        "As": (0.016, 0.0, "left", "center"),
        "Sb": (0.014, 0.008, "left", "bottom"),
        "Bi": (0.016, -0.004, "left", "top"),
        "C":  (-0.014, -0.014, "right", "top"),
        "Si": (0.0, 0.024, "center", "bottom"),
        "Ge": (0.016, -0.012, "left", "top"),
    },
    "phonon": {
        "As": (0.024, 0.014, "left", "bottom"),
        "Sb": (0.018, 0.014, "left", "bottom"),
        "Bi": (-0.024, -0.018, "right", "top"),
        "C":  (-0.018, -0.018, "right", "top"),
        "Si": (0.0, 0.034, "center", "bottom"),
        "Ge": (0.018, -0.018, "left", "top"),
    },
}


def draw_bar(ax, rows, letter, defn_label, sym, show_ylabel=False):
    x = np.arange(len(rows))
    cols = [COL_A7 if r["fam"] == "A7" else COL_CF8 for r in rows]
    ax.bar(x, [r["ratio"] for r in rows], color=cols, width=0.66,
           edgecolor="white", linewidth=0.6)
    ax.axhline(1.0, color="#374151", lw=0.8, ls="--")
    ax.text(len(rows) - 0.5, 1.008, r"$\Theta_{\mathrm{DWF}}=\Theta_D$", ha="right",
            va="bottom", fontsize=6.0, color="#374151")
    for i, r in enumerate(rows):
        ax.text(i, r["ratio"] + 0.015, f"{r['ratio']:.2f}", ha="center",
                va="bottom", fontsize=6.0)
    ax.set_xticks(x); ax.set_xticklabels([r["mat"] for r in rows])
    ax.set_ylim(*BAR_YLIM)
    if show_ylabel:
        ax.set_ylabel(r"$\Theta_{\mathrm{DWF}}/\Theta_D$")
    ax.set_title(f"({letter}) {defn_label} ratio", fontsize=8.5)


def draw_parity(ax, rows, letter, defn, defn_label, sym, show_ylabel=False):
    ax.plot(PAR_LIM, PAR_LIM, color="#374151", lw=0.9, ls="--", zorder=1)
    lo = PAR_LIM[0]
    ax.text(lo + 0.03, lo + 0.05, "1:1", fontsize=6.0, color="#374151",
            ha="left", rotation=45, rotation_mode="anchor")
    for r in rows:
        col = COL_A7 if r["fam"] == "A7" else COL_CF8
        mk = "o" if r["fam"] == "A7" else "s"
        ax.scatter(r["ratio2"], r["obs"], s=34, color=col, marker=mk,
                   edgecolors="white", linewidths=0.5, zorder=3)
        dx, dy, ha, va = PAR_LBL[defn][r["mat"]]
        ax.annotate(r["mat"], (r["ratio2"], r["obs"]),
                    xytext=(r["ratio2"] + dx, r["obs"] + dy),
                    fontsize=6.2, color=col, ha=ha, va=va)
    ax.set_xlim(*PAR_LIM); ax.set_ylim(*PAR_LIM)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$(\Theta_{\mathrm{DWF}}/$" + sym + r"$)^2$")
    if show_ylabel:
        ax.set_ylabel(r"$B_{\mathrm{scalar}}/B_{\mathrm{DOS\text{-}only}}$")
    ax.set_title(f"({letter}) {defn_label} self-consistency", fontsize=8.5)


def make_figure():
    all_rows = {defn: compute(defn) for defn, _label, _sym in DEFS}

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.7))
    top_letters = iter("abc")
    bottom_letters = iter("def")
    for col, (defn, label, sym) in enumerate(DEFS):
        rows = all_rows[defn]
        draw_bar(axes[0, col], rows, next(top_letters), label, sym,
                 show_ylabel=(col == 0))
        draw_parity(axes[1, col], rows, next(bottom_letters), defn, label, sym,
                    show_ylabel=(col == 0))

    # one shared family legend below the whole grid
    handles = [
        Patch(facecolor=COL_A7, edgecolor="none", label="A7 (As, Sb, Bi)"),
        Patch(facecolor=COL_CF8, edgecolor="none", label=r"$cF8$ (C, Si, Ge)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5,
               handlelength=1.3, columnspacing=1.6, bbox_to_anchor=(0.5, -0.004))

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "figS3.png", dpi=600, bbox_inches="tight")
    print("wrote", OUT / "figS3.png")

    # echo the plotted numbers for the record / caption
    for defn, label, _sym in DEFS:
        print(f"\n--- {label} Theta_D ---")
        print(f"{'mat':>4} {'fam':>4} {'Theta_DWF':>9} {'Theta_D':>8} {'T_hi(K)':>8}"
              f" {'ratio':>6} {'ratio^2':>8} {'obs':>8} {'dev%':>6}")
        for r in all_rows[defn]:
            dev = 100 * (r["obs"] - r["ratio2"]) / r["ratio2"]
            print(f"{r['mat']:>4} {r['fam']:>4} {r['theta_DWF']:9.1f} {r['theta_D']:8.1f}"
                  f" {r['T_hi']:8.0f} {r['ratio']:6.3f} {r['ratio2']:8.4f}"
                  f" {r['obs']:8.4f} {dev:+6.2f}")


if __name__ == "__main__":
    ensure_output_dirs()
    make_figure()
