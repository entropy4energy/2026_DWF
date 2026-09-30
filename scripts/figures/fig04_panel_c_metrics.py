#!/usr/bin/env python3
"""Sb 294 K Cu-Kalpha pattern-comparison metrics for the five DWF treatments.

Reference: experimental Fischer 1978 B_iso = 1.0968 A^2.
Models compared (from fig04_Sb_pattern.py): no DWF (B = 0), empirical Debye
(0.6861), elastic-derived Debye (0.6583), quasi-harmonic Debye (0.4778),
harmonic phonon (1.0606).

Metrics:

Full-profile (continuous y(2theta)):
  - de Gelder triangle-weighted similarity, l = 1.0 deg
    (profile similarity used in powder-pattern comparison workflows)

High-Q window (70-180 deg, same as Fig. 4):
  - R_wp on the windowed profile
    (profile-fitting / Rietveld-style search-match analogue)

Full matching window (0-180 deg; data begin at 10 deg):
  - the same three software-facing metrics exported with `_0_180` column names
    and plotted as an additional 0-180 figure.

Bragg-only (per-reflection integrated intensity):
  - R_B   = sum |I_m - I_ref| / sum I_ref
    (peak-intensity agreement term analogous to peak-based search-match)

The CSV keeps the previous diagnostic metrics as audit columns, but the
figure focuses on the three software-facing quantities above:
R_wp(high-Q), R_B(Bragg), and 1 - S_deGelder(full profile).
For plotting, these lower-is-better residuals/distances are shown directly.
Higher-is-better score columns are still exported in the CSV as an audit aid.

Outputs (the CSV in data/processed/, the renders in figures/main/):
  - fig04_panel_c_metrics.csv          one row per DWF model
  - fig04_panel_c_metrics.png          three matching-facing metrics, 70-180 R_wp
  - fig04_panel_c_metrics_0_180.png    same metrics, 0-180 R_wp
  - fig04_panel_c_metrics_0_180_two_metrics.png  compact panel; all three metrics at W3.5 x H3.7 each
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
# Every path resolves through scripts/paths.py, the single source of truth for
# this package. Nothing outside the repository root is read or written.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (  # noqa: E402
    FIGURES_MAIN, PROCESSED, calc_dir, ensure_output_dirs,
)

FIG04_DIR = FIGURES_MAIN
PATTERN_CSV = PROCESSED / "Sb_pc4_pattern_curves.csv"
PEAKS_CSV = PROCESSED / "Sb_pc4_peaks_table.csv"

# Reference column / "ground truth" intensity baseline
REF_KEY = "experimental (Fischer 1978)"

# Comparison models (order = how rows appear in the output table and bars)
MODEL_ORDER = [
    "no DWF (static)",
    "empirical digitized reference",
    "elastic-derived Debye",
    "quasi-harmonic Debye",
    "harmonic phonon spectrum (isotropic)",
]

MODEL_LABELS = {
    "no DWF (static)":                       r"no DWF (B=0)",
    "empirical digitized reference":         r"empirical Debye (B=0.686)",
    "elastic-derived Debye":                 r"elastic Debye (B=0.658)",
    "quasi-harmonic Debye":                  r"QH Debye (B=0.478)",
    "harmonic phonon spectrum (isotropic)":  r"harmonic phonon (B=1.061)",
}

MODEL_COLORS = {
    "no DWF (static)":                       "0.5",
    "empirical digitized reference":         "#8c8c8c",
    "elastic-derived Debye":                 "#DD8452",
    "quasi-harmonic Debye":                  "#55A868",
    "harmonic phonon spectrum (isotropic)":  "#4C72B0",
}

HIGH_ANGLE_WINDOW = (70.0, 180.0)  # deg
FULL_MATCH_WINDOW = (0.0, 180.0)   # deg; profile CSV starts at 10 deg
DG_TRIANGLE_L = 1.0                # deg, de Gelder bandwidth


def log(msg: str) -> None:
    print(f"[fig04_panel_c_metrics] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom == 0.0:
        return float("nan")
    return float((a * b).sum() / denom)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom == 0.0:
        return float("nan")
    return float((a * b).sum() / denom)


def rp_unweighted(model: np.ndarray, ref: np.ndarray) -> float:
    """Unweighted Rietveld R_p on the profile (Toby 2006 eq. 5)."""
    denom = float(np.sum(np.abs(ref)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(np.abs(model - ref)) / denom)


def rwp_poisson(model: np.ndarray, ref: np.ndarray, eps: float = 1e-12) -> float:
    """R_wp with Poisson-style weights w_i = 1/ref_i (counting statistics)."""
    w = 1.0 / np.maximum(ref, eps)
    num = float(np.sum(w * (model - ref) ** 2))
    den = float(np.sum(w * ref ** 2))
    if den == 0.0:
        return float("nan")
    return float(np.sqrt(num / den))


def r_bragg(model: np.ndarray, ref: np.ndarray) -> float:
    """Bragg-only intensity residual on per-peak integrated intensities."""
    denom = float(np.sum(np.abs(ref)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(np.abs(model - ref)) / denom)


def mare(model: np.ndarray, ref: np.ndarray, eps: float = 1e-12) -> float:
    rel = (model - ref) / np.maximum(np.abs(ref), eps)
    return float(np.mean(np.abs(rel)))


def rmse_rel(model: np.ndarray, ref: np.ndarray, eps: float = 1e-12) -> float:
    rel = (model - ref) / np.maximum(np.abs(ref), eps)
    return float(np.sqrt(np.mean(rel ** 2)))


def de_gelder_similarity(
    a: np.ndarray, b: np.ndarray, x: np.ndarray, l_deg: float
) -> float:
    """de Gelder, Wehrens & Hageman (J. Comput. Chem. 22, 273, 2001).

    Triangle-weighted normalized cross-correlation, with kernel
        w(Delta) = max(0, 1 - |Delta|/l_deg)
    on uniform grid x. Implemented as a moving-window inner product so it
    runs in O(N * window_pts) without scipy.
    """
    dx = float(x[1] - x[0])
    half_window = int(np.ceil(l_deg / dx))
    weights = np.maximum(0.0, 1.0 - np.abs(np.arange(-half_window, half_window + 1)) * dx / l_deg)

    def wcc(u: np.ndarray, v: np.ndarray) -> float:
        # sum_{i,j} u_i * w(x_i - x_j) * v_j
        # = sum_i u_i * (w convolved v)_i
        v_smoothed = np.zeros_like(v)
        n = len(v)
        for offset, w_off in zip(range(-half_window, half_window + 1), weights):
            if w_off == 0.0:
                continue
            j_lo = max(0, -offset)
            j_hi = min(n, n - offset)
            v_smoothed[j_lo:j_hi] += w_off * v[j_lo + offset:j_hi + offset]
        return float(np.sum(u * v_smoothed))

    num = wcc(a, b)
    den = float(np.sqrt(wcc(a, a) * wcc(b, b)))
    if den == 0.0:
        return float("nan")
    return num / den


def wasserstein_1d(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    """W_1 between two profiles on a uniform 1D grid.

    Both inputs are normalized to unit area so they are valid distributions.
    Returns int |F_a(x) - F_b(x)| dx, where F is the CDF.
    """
    dx = float(x[1] - x[0])
    a_pos = np.clip(a, 0.0, None)
    b_pos = np.clip(b, 0.0, None)
    a_norm = a_pos / max(a_pos.sum() * dx, 1e-30)
    b_norm = b_pos / max(b_pos.sum() * dx, 1e-30)
    cdf_a = np.cumsum(a_norm) * dx
    cdf_b = np.cumsum(b_norm) * dx
    return float(np.sum(np.abs(cdf_a - cdf_b)) * dx)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    pat = pd.read_csv(PATTERN_CSV)
    peaks = pd.read_csv(PEAKS_CSV)
    return pat, peaks


def peak_intensity_columns(peaks: pd.DataFrame) -> dict[str, str]:
    """Map MODEL_ORDER + REF_KEY -> peaks-table column name."""
    mapping: dict[str, str] = {}
    keys = MODEL_ORDER + [REF_KEY]
    for key in keys:
        target = key.replace(" ", "_").replace("(", "").replace(")", "")
        match = [c for c in peaks.columns if c.startswith(f"I_{target}_B=")]
        if not match:
            raise RuntimeError(f"No peaks-table column for '{key}' (looked for I_{target}_B=...)")
        mapping[key] = match[0]
    return mapping


def compute_metrics(
    pat: pd.DataFrame,
    peaks: pd.DataFrame,
) -> pd.DataFrame:
    two_theta = pat["two_theta_deg"].to_numpy(dtype=float)
    y_ref = pat[REF_KEY].to_numpy(dtype=float)
    win_lo, win_hi = HIGH_ANGLE_WINDOW
    win_mask = (two_theta >= win_lo) & (two_theta <= win_hi)
    full_lo, full_hi = FULL_MATCH_WINDOW
    full_mask = (two_theta >= full_lo) & (two_theta <= full_hi)

    peak_cols = peak_intensity_columns(peaks)
    peak_two_theta = peaks["two_theta_deg"].to_numpy(dtype=float)
    peak_window_mask = (peak_two_theta >= win_lo) & (peak_two_theta <= win_hi)
    peak_full_mask = (peak_two_theta >= full_lo) & (peak_two_theta <= full_hi)
    I_ref_peaks = peaks[peak_cols[REF_KEY]].to_numpy(dtype=float)

    # Drop trivially-zero / forbidden peaks for relative-error metrics.
    nonzero_ref = I_ref_peaks > 1e-6 * I_ref_peaks.max()

    rows: list[dict] = []
    for model in MODEL_ORDER:
        y_m = pat[model].to_numpy(dtype=float)
        I_m_peaks = peaks[peak_cols[model]].to_numpy(dtype=float)

        # Full profile
        r_full     = pearson_r(y_m, y_ref)
        cos_full   = cosine_sim(y_m, y_ref)
        rp_full    = rp_unweighted(y_m, y_ref)
        rwp_full   = rwp_poisson(y_m, y_ref)
        dg_full    = de_gelder_similarity(y_m, y_ref, two_theta, DG_TRIANGLE_L)
        w1_full    = wasserstein_1d(y_m, y_ref, two_theta)

        # High-Q window on profile
        cos_hi     = cosine_sim(y_m[win_mask], y_ref[win_mask])
        rwp_hi     = rwp_poisson(y_m[win_mask], y_ref[win_mask])

        # Full matching window, requested as 0-180 deg. The underlying profile
        # starts at 10 deg, so this is the full available calculated profile.
        cos_0_180   = cosine_sim(y_m[full_mask], y_ref[full_mask])
        rwp_0_180   = rwp_poisson(y_m[full_mask], y_ref[full_mask])
        dg_0_180    = de_gelder_similarity(
            y_m[full_mask],
            y_ref[full_mask],
            two_theta[full_mask],
            DG_TRIANGLE_L,
        )

        # Bragg-only on integrated peak intensities
        rB_all     = r_bragg(I_m_peaks[nonzero_ref], I_ref_peaks[nonzero_ref])
        mask_hi_pk = peak_window_mask & nonzero_ref
        mask_0_180_pk = peak_full_mask & nonzero_ref
        rB_0_180   = r_bragg(I_m_peaks[mask_0_180_pk], I_ref_peaks[mask_0_180_pk])
        if mask_hi_pk.sum() > 0:
            mare_hi    = mare(I_m_peaks[mask_hi_pk], I_ref_peaks[mask_hi_pk])
            rmse_hi    = rmse_rel(I_m_peaks[mask_hi_pk], I_ref_peaks[mask_hi_pk])
            max_di_hi  = float(np.max(np.abs(
                (I_m_peaks[mask_hi_pk] - I_ref_peaks[mask_hi_pk]) / I_ref_peaks[mask_hi_pk]
            )))
        else:
            mare_hi = rmse_hi = max_di_hi = float("nan")

        rows.append({
            "model": model,
            "label": MODEL_LABELS[model],
            "Pearson_r_full": r_full,
            "cosine_full":     cos_full,
            "cosine_highQ":    cos_hi,
            "cosine_0_180":    cos_0_180,
            "Rp_full":         rp_full,
            "Rwp_full":        rwp_full,
            "Rwp_highQ":       rwp_hi,
            "Rwp_0_180":       rwp_0_180,
            "RB_Bragg":        rB_all,
            "RB_Bragg_0_180":  rB_0_180,
            "MARE_highQ":      mare_hi,
            "RMSE_rel_highQ":  rmse_hi,
            "max_abs_dI_over_I_highQ": max_di_hi,
            "deGelder_S_full": dg_full,
            "deGelder_D_full": 1.0 - dg_full,
            "deGelder_S_0_180": dg_0_180,
            "deGelder_D_0_180": 1.0 - dg_0_180,
            "Wasserstein_full": w1_full,
        })
    out = pd.DataFrame(rows)
    score_pairs = [
        ("Rwp_highQ", "score_Rwp_highQ"),
        ("RB_Bragg", "score_RB_Bragg"),
        ("deGelder_D_full", "score_deGelder_full"),
        ("Rwp_0_180", "score_Rwp_0_180"),
        ("RB_Bragg_0_180", "score_RB_Bragg_0_180"),
        ("deGelder_D_0_180", "score_deGelder_0_180"),
    ]
    for residual_col, score_col in score_pairs:
        baseline = float(out.loc[0, residual_col])
        values = out[residual_col].to_numpy(dtype=float)
        out[score_col] = baseline / np.maximum(values, 1e-30)
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(
    pat: pd.DataFrame,
    df: pd.DataFrame,
    out_base: Path,
    *,
    window_label: str = "70-180°",
    rwp_column: str = "Rwp_highQ",
    rb_column: str = "RB_Bragg",
    dg_column: str = "deGelder_D_full",
    dg_title: str = "full profile, l=1°",
    include_degelder: bool = True,
    panel_size: tuple[float, float] = (3.24, 3.68),
    show_suptitle: bool = True,
    trim: bool = True,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    arial_path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    if arial_path.exists():
        font_manager.fontManager.addfont(str(arial_path))
        arial = font_manager.FontProperties(fname=str(arial_path)).get_name()
    else:
        arial = "Arial"
    plt.rcParams.update({
        "font.family": arial,
        "font.sans-serif": [arial],
        "mathtext.fontset": "custom",
        "mathtext.rm": arial,
        "mathtext.it": f"{arial}:italic",
    })

    focus_metrics = [
        {
            "column": rwp_column,
            "ylabel": r"$R_{\mathrm{wp}}$",
            "title": f"Profile-fitting $R_{{\\mathrm{{wp}}}}$\n{window_label} window",
            "subtitle": "profile search-match analogue",
        },
        {
            "column": rb_column,
            "ylabel": r"$R_B$",
            "title": "Bragg intensity $R_B$\nall nonzero reflections",
            "subtitle": "peak-based intensity term",
        },
        {
            "column": dg_column,
            "ylabel": r"$1-S_{\mathrm{dG}}$",
            "title": f"de Gelder distance\n{dg_title}",
            "subtitle": "weighted cross-correlation",
        },
    ]
    if not include_degelder:
        focus_metrics = focus_metrics[:2]

    panel_w, panel_h = panel_size
    fig, axes = plt.subplots(
        1,
        len(focus_metrics),
        figsize=(panel_w * len(focus_metrics), panel_h),
        sharex=False,
        squeeze=False,
    )
    axes = axes.ravel()
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.set_facecolor("none")
        ax.tick_params(direction="in", top=True, right=True)

    x = np.arange(len(MODEL_ORDER))
    bar_labels = [
        "no\nDWF",
        "emp.\nDebye",
        "elastic\nDebye",
        "QH\nDebye",
        "harm.\nphonon",
    ]

    for ax, spec in zip(axes, focus_metrics):
        vals = df[spec["column"]].to_numpy(dtype=float)
        colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
        bars = ax.bar(
            x, vals, width=0.72,
            color=colors, edgecolor="black", linewidth=0.45,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=8.4)
        ax.set_ylabel(spec["ylabel"], fontsize=10)
        ax.set_title(spec["title"], fontsize=10, fontweight="bold")
        ax.axhline(vals[0], color="0.35", linewidth=0.8, linestyle="--")
        ax.set_yscale("log")
        finite_vals = vals[np.isfinite(vals) & (vals > 0)]
        ax.set_ylim(float(finite_vals.min()) * 0.55,
                    float(finite_vals.max()) * 1.85)
        ax.grid(axis="y", which="both", color="0.88", linewidth=0.6)
        ax.set_axisbelow(True)

        for bar, value in zip(bars, vals):
            if value >= 0.1:
                label = f"{value:.3f}"
            elif value >= 0.01:
                label = f"{value:.4f}"
            else:
                exponent = int(np.floor(np.log10(value)))
                mantissa = value / (10 ** exponent)
                label = f"{mantissa:.2g}e{exponent}"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                label,
                ha="center", va="bottom", fontsize=7.4,
                rotation=90 if value >= 100 else 0,
                color="0.15",
                clip_on=False,
            )

    if show_suptitle:
        fig.suptitle(
            f"XRD search/matching metrics vs. Fischer 1978 ({window_label}; lower is better)",
            fontsize=11.5, fontweight="bold", y=1.03,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    else:
        fig.tight_layout()
    save_kwargs = dict(dpi=330, transparent=True)
    if trim:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(out_base.with_suffix(".png"), **save_kwargs)
    plt.close(fig)

# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def write_csv(df: pd.DataFrame, out_path: Path) -> None:
    cols = [
        "model", "label",
        "Pearson_r_full", "cosine_full", "cosine_highQ", "cosine_0_180",
        "Rp_full", "Rwp_full", "Rwp_highQ", "Rwp_0_180",
        "RB_Bragg", "RB_Bragg_0_180",
        "MARE_highQ", "RMSE_rel_highQ", "max_abs_dI_over_I_highQ",
        "deGelder_S_full", "deGelder_D_full",
        "deGelder_S_0_180", "deGelder_D_0_180",
        "score_Rwp_highQ", "score_RB_Bragg", "score_deGelder_full",
        "score_Rwp_0_180", "score_RB_Bragg_0_180", "score_deGelder_0_180",
        "Wasserstein_full",
    ]
    df.to_csv(out_path, columns=cols, index=False, float_format="%.6g")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_output_dirs()
    log("Loading Sb pattern + peaks CSVs from data/processed/...")
    pat, peaks = load_inputs()
    log(f"  profile rows: {len(pat)};  peak rows: {len(peaks)}")

    log("Computing metrics vs. Fischer 1978 baseline...")
    df = compute_metrics(pat, peaks)
    log("Done. Per-model summary:")
    for _, row in df.iterrows():
        log(f"  {row['label']}: Rwp_full={row['Rwp_full']:.4f}, "
            f"Rwp_hi-Q={row['Rwp_highQ']:.4f}, "
            f"R_B={row['RB_Bragg']:.4f}, "
            f"1-S_dG={row['deGelder_D_full']:.6f}")

    csv_path = PROCESSED / "fig04_panel_c_metrics.csv"
    fig_base = FIGURES_MAIN / "fig04_panel_c_metrics"
    fig_base_0_180 = FIGURES_MAIN / "fig04_panel_c_metrics_0_180"
    fig_base_0_180_two = FIGURES_MAIN / "fig04_panel_c_metrics_0_180_two_metrics"

    log(f"Writing CSV to {csv_path.name}...")
    write_csv(df, csv_path)
    log("Writing 70-180 figure...")
    make_figure(pat, df, fig_base)
    log("Writing 0-180 figure...")
    make_figure(
        pat,
        df,
        fig_base_0_180,
        window_label="0-180°",
        rwp_column="Rwp_0_180",
        rb_column="RB_Bragg_0_180",
        dg_column="deGelder_D_0_180",
        dg_title="0-180° profile, l=1°",
    )
    log("Writing compact all-three-metric 0-180 figure for Fig. 4 panel C...")
    make_figure(
        pat,
        df,
        fig_base_0_180_two,
        window_label="0-180°",
        rwp_column="Rwp_0_180",
        rb_column="RB_Bragg_0_180",
        dg_column="deGelder_D_0_180",
        dg_title="0-180° profile, l=1°",
        include_degelder=True,
        panel_size=(3.24, 3.68),
        show_suptitle=False,
        trim=False,
    )
    log("Done.")


if __name__ == "__main__":
    main()
