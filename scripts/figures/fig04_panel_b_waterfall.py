#!/usr/bin/env python3
"""Sb 293 K Cu-Kalpha high-angle XRD profile, clean waterfall insert.

DWF only changes intensity, not 2theta, so coincident peaks are hard to compare
in a literal overlay. This script Gaussian-broadens each set of integrated
intensities into a continuous profile, vertically offsets the profiles, and
writes both a labeled checking figure and a clean no-label transparent insert.

Reads the integrated peak intensities from
    data/processed/Sb_pc4_peaks_table.csv
so the comparison uses the *same* B values, multiplicities, |F|^2, and Lorentz-
polarization factors that produced fig04 -- no recompute drift.

Outputs (figures/main/):
    fig04_panel_b_waterfall.png          labeled waterfall version
    fig04_panel_b_waterfall_nolabel.png  clean transparent insert
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager

# Arial, matching the other panels of this figure. This panel previously listed
# DejaVu Sans first, so both its text and its math rendered in DejaVu while
# panels (b) and (c) used Arial. Registering the file makes the lookup explicit
# rather than relying on matplotlib's platform font scan.
ARIAL_FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
if ARIAL_FONT_PATH.exists():
    font_manager.fontManager.addfont(str(ARIAL_FONT_PATH))
    ARIAL_FONT = font_manager.FontProperties(fname=str(ARIAL_FONT_PATH)).get_name()
else:
    ARIAL_FONT = "Arial"

mpl.rcParams['font.family']      = ARIAL_FONT
mpl.rcParams['font.sans-serif']  = [ARIAL_FONT]
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm']      = ARIAL_FONT
mpl.rcParams['mathtext.it']      = f'{ARIAL_FONT}:italic'
mpl.rcParams['mathtext.bf']      = f'{ARIAL_FONT}:bold'
mpl.rcParams['axes.linewidth']   = 0.8

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent


# Every path resolves through scripts/paths.py, the single source of truth for
# this package. Nothing outside the repository root is read or written.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import FIGURES_MAIN, PROCESSED, REPO_ROOT, calc_dir, ensure_output_dirs  # noqa: E402

PROJECT_ROOT = REPO_ROOT

PEAKS_CSV = PROCESSED / "Sb_pc4_peaks_table.csv"

# Mapping: short key -> (peaks-table column, legend, color, B value).
# Panel B intentionally keeps only four representative profiles for clarity:
# experiment, harmonic phonon, quasi-harmonic Debye, and no DWF.
# Colors match the final Fig. 4 palette.
LAYERS = [
    ("exp",        "I_experimental_Fischer_1978_B=1.0968",                 r"experimental Fischer 1978 (B=1.097)",   "#000000", 1.0968),
    ("harm_phon",  "I_harmonic_phonon_spectrum_isotropic_B=1.0571",        r"harmonic phonon (B=1.057)",             "#4C72B0", 1.0571),
    ("qh_debye",   "I_quasi-harmonic_Debye_B=0.4588",                      r"quasi-harmonic Debye (B=0.459)",        "#55A868", 0.4588),
    ("no_dwf",     "I_no_DWF_static_B=0.0000",                             r"no DWF (B=0)",                          "0.5", 0.0000),
]

REF_KEY = "exp"
# The full high-angle window of Table III and the text. It must reach past
# 160 deg: the Sb reflection at 166.95 deg sets the maxima quoted there.
TWO_TH_MIN, TWO_TH_MAX = 70.0, 180.0

OUT_BASES = {
    "waterfall": FIGURES_MAIN / "fig04_panel_b_waterfall",
    "waterfall_nolabel": FIGURES_MAIN / "fig04_panel_b_waterfall_nolabel",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_peaks() -> pd.DataFrame:
    """Return the peaks table filtered to the high-angle window and to peaks
    with non-trivial experimental intensity (drops absent / forbidden peaks).
    """
    df = pd.read_csv(PEAKS_CSV)
    df = df[(df["two_theta_deg"] >= TWO_TH_MIN) & (df["two_theta_deg"] <= TWO_TH_MAX)].copy()
    ref_col = next(L[1] for L in LAYERS if L[0] == REF_KEY)
    ref_max_global = float(df[ref_col].max())
    keep_thresh = 0.005 * ref_max_global   # drop peaks below 0.5 % of strongest
    df = df[df[ref_col] > keep_thresh].copy()
    df = df.sort_values("two_theta_deg").reset_index(drop=True)
    return df


def normalize_to_exp(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return per-method peak intensities normalized to the experimental Fischer
    *peak-by-peak* maximum across the kept reflections. So the value 100 means
    "this reflection's height under this method = the strongest experimental
    reflection in the high-Q window."
    """
    ref_col = next(L[1] for L in LAYERS if L[0] == REF_KEY)
    ref_max = float(df[ref_col].max())
    out: dict[str, np.ndarray] = {}
    for key, col, *_ in LAYERS:
        out[key] = df[col].to_numpy(dtype=float) / ref_max * 100.0
    return out


def broaden_sticks(
    two_theta: np.ndarray,
    intensities: np.ndarray,
    x_grid: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Gaussian-broaden a discrete set of (2theta, I) reflections onto a fine
    2theta grid. We add the contributions linearly without renormalising the
    Gaussians (i.e. each stick contributes a peak whose *peak height* equals
    the input intensity, rather than its area), so a single isolated peak in
    the profile reads exactly as the integrated stick intensity. This matches
    the convention used by the fig01 NaCl high-angle inset.
    """
    z = np.zeros_like(x_grid)
    inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)
    for tt, I in zip(two_theta, intensities):
        if I <= 0.0:
            continue
        z = z + I * np.exp(-((x_grid - tt) ** 2) * inv_two_sigma2)
    return z


# ---------------------------------------------------------------------------
# Figure: continuous high-angle nested envelope
# ---------------------------------------------------------------------------

def fig_highangle_nested_overlay(
    df: pd.DataFrame,
    *,
    labels: bool = True,
    sigma: float = 0.13,
) -> plt.Figure:
    """Draw all six DWF treatments as same-axis nested envelopes.

    The x positions and amplitudes stay physically comparable: every curve
    uses the same Bragg 2theta grid and the same intensity normalization.
    Color-filled bands between adjacent profiles reveal the outer-to-inner
    nesting that plain line overlays hide.
    """
    norm = normalize_to_exp(df)            # per-method intensities (% of exp max)
    two_theta = df["two_theta_deg"].to_numpy(dtype=float)

    # Render fine enough that the narrowest Gaussian is well resolved. At
    # sigma=0.13 deg (FWHM ~ 0.31 deg) we want ~25 samples per FWHM, i.e.
    # ~0.012 deg/sample, so ~9000 points across the 70-180 window.
    x_grid = np.linspace(TWO_TH_MIN, TWO_TH_MAX, 9000)
    profiles = {
        key: broaden_sticks(two_theta, norm[key], x_grid, sigma)
        for key, *_ in LAYERS
    }

    y_max = max(float(profiles[k].max()) for k in profiles)
    y_top = float(np.ceil(y_max / 25.0)) * 25.0 + 5.0

    color_by_key = {k: c for k, _col, _lg, c, _B in LAYERS}
    legend_by_key = {k: lg for k, _col, lg, _c, _B in LAYERS}
    linestyle_by_key = {"no_dwf": "--"}
    is_ref = lambda k: (k == REF_KEY)

    # Inner-to-outer order follows decreasing B: experimental is the smallest
    # high-angle envelope and no-DWF is the outermost envelope.
    order_inner_to_outer = [k for k, *_ in LAYERS]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.92, bottom=0.135)
    ax.set_facecolor("white" if labels else "none")

    lw_default = 1.25
    lw_ref     = 1.8

    # The filled annuli are the key readability trick: a high-intensity curve
    # no longer erases a low-intensity curve; it becomes the colored shell
    # outside that curve.
    inner = np.zeros_like(x_grid)
    for z, key in enumerate(order_inner_to_outer):
        outer = profiles[key]
        ax.fill_between(
            x_grid,
            inner,
            outer,
            where=(outer >= inner),
            color=color_by_key[key],
            alpha=0.12 if is_ref(key) else 0.38,
            linewidth=0,
            zorder=2 + z,
        )
        inner = np.maximum(inner, outer)

    # Draw the boundaries from outer to inner so the innermost experimental
    # curve stays visible while the tallest envelope still defines the outside.
    for z, key in enumerate(reversed(order_inner_to_outer)):
        color = color_by_key[key]
        ax.plot(
            x_grid, profiles[key],
            color=color,
            linestyle=linestyle_by_key.get(key, "-"),
            lw=lw_ref if is_ref(key) else lw_default,
            label=legend_by_key[key],
            alpha=0.95,
            zorder=20 + z,
        )

    ax.set_xlim(TWO_TH_MIN, TWO_TH_MAX)
    ax.set_ylim(0.0, y_top)
    ax.tick_params(axis='both', direction='in', top=True, right=True,
                   length=4.0, width=0.9)
    for spine in ax.spines.values():
        spine.set_color('#333333')
        spine.set_linewidth(1.0)

    if labels:
        ax.set_xlabel(r'$2\theta$ (deg, Cu K$\alpha_1$, $\lambda$=1.5406 Å)',
                      fontsize=11, labelpad=3)
        ax.set_ylabel('intensity (% of strongest experimental peak)',
                      fontsize=11, labelpad=4)
        ax.set_title(
            f'Sb 293 K high-angle overlay: {TWO_TH_MIN:.0f}-{TWO_TH_MAX:.0f}° '
            f'(nested envelopes, shared scale, $\\sigma$={sigma:.2f}°)',
            fontsize=11, pad=6, loc='left', fontweight='bold',
        )
        ax.set_xticks(np.arange(TWO_TH_MIN, TWO_TH_MAX + 1, 10))
        ax.tick_params(axis='both', labelsize=9.5)
        ax.legend(loc='upper right', frameon=False, fontsize=8.8,
                  handlelength=1.8, labelspacing=0.32, borderaxespad=0.4)
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.tick_params(labelbottom=False, labelleft=False)

    return fig


def fig_highangle_offset_waterfall(
    df: pd.DataFrame,
    *,
    labels: bool = True,
    sigma: float = 0.13,
    offset_step: float | None = None,
) -> plt.Figure:
    """Draw a vertically offset version for visual separation.

    This is the alternative waterfall-style view: every profile keeps the same
    2theta scale and intensity normalization, but each method sits on its own
    baseline so coincident peaks can be compared without occlusion.
    """
    norm = normalize_to_exp(df)
    two_theta = df["two_theta_deg"].to_numpy(dtype=float)

    x_grid = np.linspace(TWO_TH_MIN, TWO_TH_MAX, 9000)
    profiles = {
        key: broaden_sticks(two_theta, norm[key], x_grid, sigma)
        for key, *_ in LAYERS
    }

    y_max = max(float(profiles[k].max()) for k in profiles)
    if offset_step is None:
        offset_step = max(55.0, float(np.ceil((0.40 * y_max) / 5.0) * 5.0))

    color_by_key = {k: c for k, _col, _lg, c, _B in LAYERS}
    legend_by_key = {k: lg for k, _col, lg, _c, _B in LAYERS}
    short_label_by_key = {
        "exp": "experiment",
        "harm_phon": "harm. phonon",
        "qh_debye": "QH Debye",
        "no_dwf": "no DWF",
    }
    linestyle_by_key = {"no_dwf": "--"}
    is_ref = lambda k: (k == REF_KEY)

    order_bottom_to_top = [k for k, *_ in LAYERS]
    offsets = {key: i * offset_step for i, key in enumerate(order_bottom_to_top)}
    y_top = offsets[order_bottom_to_top[-1]] + y_max + 0.18 * offset_step

    if labels:
        # Trial layout requested by the user: swap panel A/B sizes, so panel B
        # now uses the former wide/flat panel-A footprint.
        fig, ax = plt.subplots(figsize=(11.0, 2.8))
        fig.subplots_adjust(left=0.075, right=0.99, top=0.82, bottom=0.22)
    else:
        fig, ax = plt.subplots(figsize=(11.0, 2.8))
        fig.subplots_adjust(left=0.030, right=0.995, top=0.970, bottom=0.060)
    ax.set_facecolor("white" if labels else "none")

    lw_default = 1.35
    lw_ref = 2.0
    line_effect = [
        pe.Stroke(linewidth=3.0, foreground="white", alpha=0.78),
        pe.Normal(),
    ]

    for z, key in enumerate(order_bottom_to_top):
        color = color_by_key[key]
        offset = offsets[key]
        ax.axhline(offset, color="#d0d0d0", lw=0.45, alpha=0.65, zorder=1)
        ax.plot(
            x_grid, profiles[key] + offset,
            color=color,
            linestyle=linestyle_by_key.get(key, "-"),
            lw=lw_ref if is_ref(key) else lw_default,
            alpha=0.96,
            path_effects=line_effect,
            zorder=10 + z,
        )

    ax.set_xlim(TWO_TH_MIN, TWO_TH_MAX)
    ax.set_ylim(0.0, y_top)
    ax.tick_params(axis='both', direction='in', top=True, right=True,
                   length=4.0, width=0.9)
    for spine in ax.spines.values():
        spine.set_color('#333333')
        spine.set_linewidth(1.0)

    if labels:
        ax.set_xlabel(r'$2\theta$ (deg, Cu K$\alpha_1$, $\lambda$=1.5406 Å)',
                      fontsize=11, labelpad=3)
        ax.set_ylabel('intensity + offset (%)',
                      fontsize=9.2, labelpad=2)
        ax.set_title(
            f'Sb 293 K high-angle waterfall: {TWO_TH_MIN:.0f}-{TWO_TH_MAX:.0f}° '
            f'(offset={offset_step:.0f}, $\\sigma$={sigma:.2f}°)',
            fontsize=9.6, pad=5, loc='left', fontweight='bold',
        )
        ax.set_xticks(np.arange(TWO_TH_MIN, TWO_TH_MAX + 1, 10))
        ax.set_yticks([offsets[k] for k in order_bottom_to_top])
        ax.tick_params(axis='y', labelleft=False)
        ax.tick_params(axis='both', labelsize=9.5)

        label_x = TWO_TH_MAX - 0.7
        for key in order_bottom_to_top:
            ax.text(
                label_x,
                offsets[key] + 0.15 * offset_step,
                short_label_by_key[key],
                color=color_by_key[key],
                fontsize=7.0,
                ha="right",
                va="center",
                clip_on=False,
                path_effects=[pe.withStroke(linewidth=2.6, foreground="white", alpha=0.90)],
                zorder=40,
            )

        scale_x = TWO_TH_MAX - 4.2
        scale_y = offsets[order_bottom_to_top[-1]] + 0.42 * offset_step
        scale_h = 50.0
        ax.plot([scale_x, scale_x], [scale_y, scale_y + scale_h],
                color="#333333", lw=0.9, zorder=45)
        ax.plot([scale_x - 0.35, scale_x + 0.35], [scale_y, scale_y],
                color="#333333", lw=0.9, zorder=45)
        ax.plot([scale_x - 0.35, scale_x + 0.35], [scale_y + scale_h, scale_y + scale_h],
                color="#333333", lw=0.9, zorder=45)
        ax.text(scale_x - 0.7, scale_y + 0.5 * scale_h, "50%",
                ha="right", va="center", fontsize=8.2, color="#333333")
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.tick_params(labelbottom=False, labelleft=False)

    return fig


# Keep the old names working for any ad-hoc callers, but route them to the
# clearer same-axis nested-envelope figure.
fig_highangle_overlay = fig_highangle_nested_overlay
fig_highangle_waterfall = fig_highangle_offset_waterfall
fig_grouped_stems = fig_highangle_nested_overlay


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def save(
    fig: plt.Figure,
    base: Path,
    transparent: bool = False,
    *,
    trim: bool = True,
) -> None:
    fc = "none" if transparent else "white"
    kw = dict(dpi=240, facecolor=fc, transparent=transparent)
    if trim:
        kw["bbox_inches"] = "tight"
    p = base.with_suffix(".png")
    fig.savefig(p, **kw)
    print(f"  saved: {p}")
    plt.close(fig)


def cleanup_old_outputs(src_dir: Path) -> None:
    stale = ["fig04_Sb_xrd_stacked.png", "fig04_Sb_xrd_stacked.pdf",
             "fig04_Sb_xrd_stacked_nolabel.png", "fig04_Sb_xrd_stacked_nolabel.pdf",
             "fig04_Sb_xrd_waterfall.png", "fig04_Sb_xrd_waterfall.pdf",
             "fig04_Sb_xrd_waterfall_nolabel.png", "fig04_Sb_xrd_waterfall_nolabel.pdf"]
    for name in stale:
        p = src_dir / name
        if p.exists():
            p.unlink()
            print(f"  removed: {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_output_dirs()
    print(f"[info] loading peaks table {PEAKS_CSV.relative_to(PROJECT_ROOT)}")
    df = load_peaks()
    print(f"[info]   {len(df)} reflections kept in {TWO_TH_MIN:.0f}–{TWO_TH_MAX:.0f} deg "
          f"with I_exp > 0.5% of strongest")

    norm = normalize_to_exp(df)
    print("[info]   per-method ratio of (max stick height) to experimental:")
    ref_max = max(norm[REF_KEY])
    for key, _, legend, _, _B in LAYERS:
        ratio = float(np.max(norm[key])) / ref_max
        print(f"[info]     {legend:42s}  max / exp = {ratio:6.3f}")

    print("[info] cleaning up stale outputs from prior runs")
    cleanup_old_outputs(BASE_DIR)

    print("[info] figure: offset high-angle waterfall (labeled)")
    save(fig_highangle_offset_waterfall(df, labels=True), OUT_BASES["waterfall"], trim=False)

    print("[info] figure: offset high-angle waterfall (no-label, transparent)")
    save(
        fig_highangle_offset_waterfall(df, labels=False),
        OUT_BASES["waterfall_nolabel"],
        transparent=True,
        trim=False,
    )

    print("[info] done.")


if __name__ == "__main__":
    main()
