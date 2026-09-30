#!/usr/bin/env python3
"""Generate SI figures for the non-A7 (C/Si/Ge) transferability check.

Both are written to figures/si/. The manuscript's SI Figs. S1 and S2 are
assembled from these panels by hand, with the panel letters added there.

figS1.png (SI Fig. S1): B_iso(T) for C, Si, Ge on the top row and residuals on
                        the bottom row, using binned non-model literature
                        reference values. Captions label the top row (a)-(c)
                        and the residual row (d)-(f).
figS2.png (SI Fig. S2): mechanism figure -- three panels, letters added at
                        assembly time rather than baked in: window
                        decomposition of the inverse-square moment <f^-2>
                        (real DOS vs ideal Debye) in reduced frequency f/f_D,
                        scaled to contributions to (Theta_D/Theta_DWF)^2 - 1;
                        Debye-Waller Theta_DWF vs
                        the empirical Theta_D; and the scalar-Debye deficit
                        1 - B_scalar/B_harmonic vs T, approaching the classical
                        limit 1-(Theta_DWF/Theta_D)^2.
"""
from __future__ import annotations
import csv, lzma, math
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# np.trapz was deprecated in numpy 2.0 and is gone in current releases. The
# getattr() form used previously evaluated np.trapz eagerly as the default
# argument, so it raised AttributeError there even though trapezoid was present.
# Same function either way.
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

plt.rcParams.update({
    "font.family": ARIAL_FONT,
    "font.sans-serif": [ARIAL_FONT],
    "mathtext.fontset": "custom",
    "mathtext.rm": ARIAL_FONT,
    "mathtext.it": f"{ARIAL_FONT}:italic",
    "mathtext.bf": f"{ARIAL_FONT}:bold",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
})

# Every path resolves through scripts/paths.py, the single source of truth
# for this package. Nothing outside the repository root is read or written.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DFT_RAW, FIGURES_SI, LITERATURE, PROCESSED, calc_dir, ensure_output_dirs  # noqa: E402
CSV = PROCESSED
BASE_CANDIDATES = [DFT_RAW]
OUT = FIGURES_SI

# main-text palette
C_EMP, C_AEL, C_AGL, C_HARM = "#c97900", "#1d4ed8", "#0f766e", "#c62828"
H = 6.62607015e-34; KB = 1.380649e-23; THZ = 1e12

MAT = {
 "C":  dict(dir="C",  theta_emp=2230),
 "Si": dict(dir="Si", theta_emp=645),
 "Ge": dict(dir="Ge", theta_emp=374),
}

# Fig. S2(a) windows in reduced frequency x = f/f_D, with f_D from theta_emp.
# The Debye reference is 3x^2 in these units (the reduced-DOS convention of
# Sears & Shelley 1991), so the same windows compare C, Si and Ge directly.
REDUCED_WINDOW_EDGES = (0.0, 0.15, 0.3, 0.6, 1.0, math.inf)


def apl_file(rel, filename="aflow.apl.phonon_dos.out.xz"):
    for base in BASE_CANDIDATES:
        path = base / rel / filename
        if path.exists():
            return path
    return BASE_CANDIDATES[0] / rel / filename


def read_curves(mat):
    rows = list(csv.DictReader(open(CSV / f"{mat}_phonon_B_curves.csv")))
    T = np.array([float(r["T_K"]) for r in rows])
    g = lambda k: np.array([float(r[k]) for r in rows])
    emp_key = [k for k in rows[0] if k.startswith("B_emp")][0]
    return dict(T=T, harm=g("B_iso_harmonic_A2"), emp=g(emp_key),
                ael=g("B_AEL_A2"), agl=g("B_AGL_A2"))


# ---- literature reference points (multi-source, full provenance in CSV) ----
LIT_CSV = LITERATURE / "cF8_literature_B_sources.csv"

# Only non-model source types are treated as literature reference data in Fig. S1.
REFERENCE_SOURCE_TYPES = {
    "diffraction_refinement",
    "cbed_direct",
    "pdos_derived",
    "thetaM_derived",
}
# Representative bins for a readable 0-1000 K comparison. The 290-300 K bin
# deliberately groups the common room-temperature conventions.
REFERENCE_BINS = [
    (1, 1, 1), (80, 80, 80), (100, 100, 100), (150, 150, 150),
    (200, 200, 200), (250, 250, 250), (295, 290, 300),
    (400, 400, 400), (500, 500, 500), (600, 600, 600),
    (700, 700, 700), (800, 800, 800), (900, 900, 900),
    (1000, 1000, 1000),
]
# residual panel compares our models against the binned literature medians.
RESIDUAL_TYPES = REFERENCE_SOURCE_TYPES


def read_lit(mat):
    """Return list of literature points for `mat` from the multi-source CSV.

    Rows flagged as "artifact" (physically impossible values kept in the archive
    for fidelity) are skipped here so they never enter the plotted aggregate.
    """
    pts = []
    with open(LIT_CSV, newline="") as fh:
        reader = csv.DictReader(line for line in fh if not line.startswith("#"))
        for i, row in enumerate(reader):
            if row["material"] != mat:
                continue
            if row.get("flag", "").strip() == "artifact":
                continue
            try:
                T = float(row["T_K"]); B = float(row["B_A2"])
            except ValueError:
                continue
            unc = float(row["B_unc_A2"]) if row["B_unc_A2"] else None
            pts.append(dict(T=T, B=B, unc=unc, src_key=row["source_key"],
                            src_label=row["source_label"],
                            src_type=row["source_type"], row_index=i))
    return pts


def aggregate_lit_points(lit):
    """Return binned literature medians and min-max source spread.

    Within each temperature bin, each source contributes one median value so a
    dense table from one paper does not overweight the literature aggregate.
    """
    ref = [pt for pt in lit if pt["src_type"] in REFERENCE_SOURCE_TYPES]
    out = []
    for x, lo, hi in REFERENCE_BINS:
        by_source = {}
        for pt in ref:
            if lo <= pt["T"] <= hi:
                by_source.setdefault(pt["src_key"], []).append(pt["B"])
        if not by_source:
            continue
        vals = np.array([np.median(v) for v in by_source.values()], dtype=float)
        center = float(np.median(vals))
        low = float(vals.min())
        high = float(vals.max())
        out.append(dict(T=x, B=center, err_low=center - low, err_high=high - center,
                        n_sources=len(vals),
                        source_keys=";".join(sorted(by_source))))
    return out


def aggregate_summary(lit):
    by_source = {}
    for pt in lit:
        if pt["src_type"] not in REFERENCE_SOURCE_TYPES:
            continue
        by_source.setdefault(pt["src_key"], []).append(pt)
    return ", ".join(f"{k}({len(v)})" for k, v in sorted(by_source.items()))


def read_dos(mat):
    f, d = [], []
    with lzma.open(apl_file(MAT[mat]["dir"]), "rt", errors="replace") as fh:
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
    d = d / TRAPEZOID(d, f)
    return f, d


def debye_dos(theta, f):
    fD = KB * theta / H / THZ
    return np.where(f <= fD, 3 * f**2 / fD**3, 0.0), fD


def inverse_square_real_window(f, g, lo, hi):
    """Integrate g(f)/f^2 over a window, including interpolated boundaries."""
    lo_clip = max(float(lo), float(f[0]))
    hi_clip = min(float(hi), float(f[-1]))
    if hi_clip <= lo_clip:
        return 0.0
    inner = f[(f > lo_clip) & (f < hi_clip)]
    grid = np.concatenate(([lo_clip], inner, [hi_clip]))
    vals = np.interp(grid, f, g) / grid**2
    return float(TRAPEZOID(vals, grid))


def inverse_square_debye_window(fD, lo, hi):
    """Exact ideal-Debye integral of g_D(f)/f^2 over a frequency window."""
    overlap = max(0.0, min(float(hi), fD) - max(float(lo), 0.0))
    return 3.0 * overlap / fD**3


# ---------------- Figure 1: B(T) comparison ----------------
def _plot_lit_aggregate(ax, lit, color="#111827", zorder=10,
                        label_multi=None, label_single=None):
    """Plot binned literature aggregate.

    Multi-source bins (>=2 independent sources) are filled circles with a
    min-max spread bar; single-source bins are open circles (no real spread to
    show) so the reader can see where the "literature band" is one paper only.
    """
    multi = [p for p in lit if p["n_sources"] >= 2]
    single = [p for p in lit if p["n_sources"] < 2]
    if multi:
        T = np.array([p["T"] for p in multi]); B = np.array([p["B"] for p in multi])
        yerr = np.array([[p["err_low"] for p in multi],
                         [p["err_high"] for p in multi]])
        ax.errorbar(T, B, yerr=yerr, fmt="o", ms=5.5, color=color,
                    mfc=color, mec="white", mew=0.7, ecolor="#6b7280",
                    elinewidth=1.1, capsize=3.0, zorder=zorder + 1,
                    label=label_multi)
    if single:
        T = np.array([p["T"] for p in single]); B = np.array([p["B"] for p in single])
        ax.scatter(T, B, marker="o", s=34, facecolors="none", edgecolors=color,
                   linewidths=1.2, zorder=zorder, label=label_single)


def fig_bt():
    mats = ["C", "Si", "Ge"]
    XMAX = 1000
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.6),
                             gridspec_kw={"height_ratios": [2.0, 1.0]})
    for j, mat in enumerate(mats):
        c = read_curves(mat)
        lit_all = read_lit(mat)
        lit = aggregate_lit_points(lit_all)
        print(f"{mat}: reference sources = {aggregate_summary(lit_all)}")
        print(f"{mat}: binned reference points = {len(lit)}")
        ax = axes[0, j]; axr = axes[1, j]
        ax.plot(c["T"], c["emp"], color=C_EMP, ls="--", lw=2.2, label="empirical Debye reference")
        ax.plot(c["T"], c["ael"], color=C_AEL, ls="-", lw=2.2, label="elastic-property-derived Debye")
        ax.plot(c["T"], c["agl"], color=C_AGL, ls="-", lw=2.2, label="quasi-harmonic Debye")
        ax.plot(c["T"], c["harm"], color=C_HARM, ls="-", lw=2.2, label="harmonic phonon displacement")

        lit = [pt for pt in lit if pt["T"] <= XMAX]
        _plot_lit_aggregate(
            ax, lit,
            label_multi="literature median (>=2 sources, min-max spread)" if j == 0 else None,
            label_single="single-source reference" if j == 0 else None,
        )
        # Panel letters are added when the figure is assembled, not baked in here.
        ax.set_title(mat)
        if j == 0:
            ax.set_ylabel(r"$B_{\mathrm{iso}}$ ($\mathrm{\AA}^2$)")
        ax.set_xlim(0, XMAX)
        mask = c["T"] <= XMAX
        bmax = max([c["harm"][mask].max()] +
                   [p["B"] + p["err_high"] for p in lit])
        ax.set_ylim(0, bmax * 1.10)
        ax.grid(alpha=0.25, lw=0.6)
        if j == 0:
            ax.legend(frameon=False, fontsize=8.0, loc="upper left")

        # residuals vs measured/derived literature only (exclude pure model calc)
        for key, col, ls in [("emp", C_EMP, "--"), ("ael", C_AEL, "-"),
                             ("agl", C_AGL, "-"), ("harm", C_HARM, "-")]:
            res_T, res_d = [], []
            for pt in lit:
                res_T.append(pt["T"])
                res_d.append(np.interp(pt["T"], c["T"], c[key]) - pt["B"])
            order = np.argsort(res_T)
            res_T = np.array(res_T)[order]; res_d = np.array(res_d)[order]
            axr.plot(res_T, res_d, color=col, ls=ls, lw=1.6, marker="o", ms=3.5)
        axr.axhline(0, color="black", lw=0.9, alpha=0.5)
        axr.set_xlim(0, XMAX)
        axr.set_xlabel("Temperature (K)")
        if j == 0:
            axr.set_ylabel("model $-$ lit. median")
        axr.grid(alpha=0.25, lw=0.6)

    fig.tight_layout()
    p = OUT / "figS1.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print("wrote", p)


# ---------------- Figure 2: moment mechanism ----------------
def fig_moment():
    mats = ["C", "Si", "Ge"]
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 4.6))
    panel_color = {"C": "#6b7280", "Si": C_AEL, "Ge": C_AGL}

    # (a) window decomposition of <f^-2> excess (real - Debye at theta_emp) in
    # reduced frequency x = f/f_D. Each bar is (f_D^2/3) * excess, so the bars of
    # one material sum to (Theta_D/Theta_DWF)^2 - 1, the ratio of panel (b).
    ax = axes[0]
    edges = REDUCED_WINDOW_EDGES
    windows = [r"$<0.15$", "0.15-0.3", "0.3-0.6", "0.6-1", r"$>1$"]
    x = np.arange(len(windows)); w = 0.25
    colors = panel_color
    for i, mat in enumerate(mats):
        f, g = read_dos(mat)
        _gD, fD = debye_dos(MAT[mat]["theta_emp"], f)
        excess = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            real = inverse_square_real_window(f, g, lo * fD, hi * fD)
            ideal = inverse_square_debye_window(fD, lo * fD, hi * fD)
            excess.append((real - ideal) * fD**2 / 3.0)
        ax.bar(x + (i - 1) * w, excess, w, label=mat, color=colors[mat])
        theta_DWF = H * math.sqrt(3.0 / TRAPEZOID(g / f**2, f)) * THZ / KB
        print(f"  {mat}: window sum {sum(excess):.4f} vs "
              f"(Theta_D/Theta_DWF)^2 - 1 = {(MAT[mat]['theta_emp'] / theta_DWF) ** 2 - 1:.4f}")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(windows)
    ax.set_xlabel(r"reduced-frequency window $f/f_D$")
    ax.set_ylabel(r"contribution to $(\Theta_D/\Theta_{\mathrm{DWF}})^2-1$")
    ax.set_title(r"where the $\langle f^{-2}\rangle$ mismatch lives")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, lw=0.6, axis="y")

    # (b) Theta_DWF (Debye-Waller) vs Theta_emp
    ax = axes[1]
    tDWF, tE, ratio = [], [], []
    for mat in mats:
        f, g = read_dos(mat)
        inv = TRAPEZOID(g / f**2, f)
        fD_match = math.sqrt(3.0 / inv)
        theta_DWF = H * fD_match * THZ / KB
        tDWF.append(theta_DWF); tE.append(MAT[mat]["theta_emp"])
        ratio.append(theta_DWF / MAT[mat]["theta_emp"])
    xx = np.arange(len(mats))
    ax.bar(xx - 0.2, tE, 0.4, label=r"$\Theta_D^{\mathrm{emp}}$ (literature)", color="#9ca3af")
    ax.bar(xx + 0.2, tDWF, 0.4, label=r"$\Theta_{\mathrm{DWF}}$ (from $\langle f^{-2}\rangle$)", color=C_HARM)
    for i, r in enumerate(ratio):
        ax.text(i, max(tE[i], tDWF[i]) * 1.02, f"{r:.2f}", ha="center", fontsize=10)
    ax.set_xticks(xx); ax.set_xticklabels(mats)
    ax.set_ylabel("Debye temperature (K)")
    ax.set_title(r"$\Theta_{\mathrm{DWF}}<\Theta_D$: ratio $\Theta_{\mathrm{DWF}}/\Theta_D$ annotated")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.6, axis="y")

    # (c) scalar deficit 1 - B_Debye/B_harm approaches 1-(Theta_DWF/Theta_D)^2.
    # The numerator is the B_emp_<Theta>K_A2 column, i.e. the scalar Debye
    # expression evaluated at the literature Theta_D^emp -- NOT a digitized
    # literature curve (that construction applies only to the A7 series).
    # The classical limit uses the SAME Theta_DWF (from <f^-2>) and the empirical
    # Theta_D as panel (b), so this panel is the temperature-resolved consequence
    # of the moment mismatch -- not an independent fit.
    ax = axes[2]
    for mat in mats:
        c = read_curves(mat)
        f, g = read_dos(mat)
        theta_DWF = H * math.sqrt(3.0 / TRAPEZOID(g / f**2, f)) * THZ / KB
        classical_limit = 1.0 - (theta_DWF / MAT[mat]["theta_emp"]) ** 2
        Tm = c["T"] <= 1000
        deficit = 1.0 - c["emp"][Tm] / c["harm"][Tm]
        ax.plot(c["T"][Tm], 100 * deficit, color=panel_color[mat], lw=2.2, label=mat)
        ax.axhline(100 * classical_limit, color=panel_color[mat], ls=":", lw=1.4)
    ax.axvline(295, color="black", lw=0.8, alpha=0.4)
    ax.text(300, ax.get_ylim()[1] * 0.04, "295 K", fontsize=8.5, alpha=0.7)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, None)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"scalar deficit $1-B^{\mathrm{Debye}}/B_{\mathrm{harm}}$ (%)")
    ax.set_title(r"deficit $\rightarrow$ high-$T$ limit $1-(\Theta_{\mathrm{DWF}}/\Theta_D)^2$")
    ax.plot([], [], color="#444", lw=2.2, label="deficit (solid)")
    ax.plot([], [], color="#444", ls=":", lw=1.4, label="high-$T$ limit (dotted)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.6)

    fig.tight_layout()
    p = OUT / "figS2.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print("wrote", p)


if __name__ == "__main__":
    ensure_output_dirs()
    fig_bt()
    fig_moment()
