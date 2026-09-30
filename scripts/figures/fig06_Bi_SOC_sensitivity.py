#!/usr/bin/env python3
"""Render manuscript Fig 6 (Bi SOC sensitivity).

Bi B_iso(T) under elastic-derived Debye and quasi-harmonic Debye at scalar-
relativistic (noSOC) and scalar+SOC levels, overlaid with the noSOC and SOC
harmonic phonon curves and the empirical digitized reference. The right panel
shows residuals at experiment temperatures. This is the Bi SOC sensitivity
"regime-level vs noise-level" plot discussed around the Bi results section.

Produces:
  fig06_Bi_SOC_sensitivity.png  (ingested by 3_result.tex)

Inputs (all under the flattened CSV store):
  data/processed/Bi_SOC_relativistic_sensitivity_main_curves_wide.csv
  data/processed/Bi_SOC_relativistic_sensitivity_residuals_at_experiment_wide.csv
  data/processed/Bi_SOC_phonon_harmonic_curves.csv
  data/processed/Bi_comparison_experiment_points.csv

Regenerate the scalar CSVs with data/export_figure_wide_csvs.py off the
authoritative ``bt_curves_full.csv``. Regenerate the SOC harmonic-phonon CSV
with data/Bi_SOC_phonon_analysis.py.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Iterable

BASE_DIR = Path(__file__).resolve().parent
# Every path resolves through scripts/paths.py, the single source of truth
# for this package. Nothing outside the repository root is read or written.
sys.path.insert(0, str(BASE_DIR.parent))
from paths import FIGURES_MAIN, PROCESSED, REPO_ROOT, VERIFICATION_LOGS  # noqa: E402

CACHE_DIR = VERIFICATION_LOGS / ".cache"
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


PROJECT_ROOT = REPO_ROOT
WIDE_DIR = PROCESSED

ARIAL_FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
if ARIAL_FONT_PATH.exists():
    font_manager.fontManager.addfont(str(ARIAL_FONT_PATH))
ARIAL_FONT = (
    font_manager.FontProperties(fname=str(ARIAL_FONT_PATH)).get_name()
    if ARIAL_FONT_PATH.exists()
    else "Arial"
)

plt.rcParams.update(
    {
        "font.family": ARIAL_FONT,
        "font.sans-serif": [ARIAL_FONT],
        "mathtext.fontset": "custom",
        "mathtext.rm": ARIAL_FONT,
        "mathtext.it": f"{ARIAL_FONT}:italic",
        "mathtext.bf": f"{ARIAL_FONT}:bold",
        "axes.linewidth": 1.2,
        "axes.labelsize": 11,
        "axes.titlesize": 11.5,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.transparent": True,
    }
)

# Palette aligned with Fig. 2 / Fig. 3 (seaborn "deep" hex values):
#   empirical reference  -> neutral grey  (dashed, secondary role)
#   elastic Debye        -> warm orange
#   quasi-harmonic Debye -> green
#   harmonic phonon      -> blue (same as Fig. 2 hero curve)
# SOC variants reuse the same hue but take a different linestyle.
_C_EMPIRICAL = "#8c8c8c"
_C_HARMONIC  = "#4c72b0"
_C_ELASTIC   = "#dd8452"
_C_QH_DEBYE  = "#55a868"

# (main-curve column, residual column, legend label, color, linestyle)
SOC_METHODS = [
    (
        "B_empirical_digitized_reference_A2",
        None,
        "empirical reference",
        _C_EMPIRICAL,
        (0, (5, 4)),
    ),
    (
        "Bi_noSOC_B_elastic_derived_debye_A2",
        "residual_Bi_noSOC_elastic_derived_debye",
        "elastic Debye",
        _C_ELASTIC,
        "-",
    ),
    (
        "Bi_SOC_B_elastic_derived_debye_A2",
        "residual_Bi_SOC_elastic_derived_debye",
        "elastic Debye (SOC)",
        _C_ELASTIC,
        (0, (5, 4)),
    ),
    (
        "Bi_noSOC_B_quasi_harmonic_debye_A2",
        "residual_Bi_noSOC_quasi_harmonic_debye",
        "quasi-harmonic Debye",
        _C_QH_DEBYE,
        "-",
    ),
    (
        "Bi_SOC_B_quasi_harmonic_debye_A2",
        "residual_Bi_SOC_quasi_harmonic_debye",
        "quasi-harmonic Debye (SOC)",
        _C_QH_DEBYE,
        (0, (5, 3, 1.4, 3)),
    ),
]

# Harmonic phonon curves are loaded separately because the current scalar-SOC
# wide residual CSV predates the SOC APL run.
HARMONIC_METHODS = [
    (
        "bi_main",
        "B_A2__harmonic_phonon_spectrum_(isotropic)",
        "harmonic phonon",
        _C_HARMONIC,
        "-",
    ),
    (
        "soc_apl",
        "B_iso_SOC_A2",
        "harmonic phonon (SOC)",
        _C_HARMONIC,
        (0, (5, 4)),
    ),
]


def parse_float(value: str) -> float:
    if value == "":
        return float("nan")
    return float(value)


def load_csv_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        columns: dict[str, list[float | str]] = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for name, value in row.items():
                if name == "method":
                    columns[name].append(value)
                else:
                    columns[name].append(parse_float(value))

    arrays: dict[str, np.ndarray] = {}
    for name, values in columns.items():
        if name == "method":
            arrays[name] = np.asarray(values, dtype=object)
        else:
            arrays[name] = np.asarray(values, dtype=float)
    return arrays


def finite_xy(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[mask], y_values[mask]


def prepend_origin(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x_values.size == 0:
        return x_values, y_values
    if np.isclose(x_values[0], 0.0) and np.isclose(y_values[0], 0.0):
        return x_values, y_values
    return np.insert(x_values, 0, 0.0), np.insert(y_values, 0, 0.0)


def residual_at_experiment(
    exp_t: np.ndarray,
    exp_b: np.ndarray,
    model_t: np.ndarray,
    model_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_model, y_model = finite_xy(model_t, model_b)
    x_exp, y_exp = finite_xy(exp_t, exp_b)
    if x_model.size == 0 or x_exp.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    inside = (x_exp >= np.nanmin(x_model)) & (x_exp <= np.nanmax(x_model))
    x_exp = x_exp[inside]
    y_exp = y_exp[inside]
    return x_exp, np.interp(x_exp, x_model, y_model) - y_exp


def max_in_x_window(x_values: np.ndarray, y_values: np.ndarray, x_max: float) -> float:
    x_finite, y_finite = finite_xy(x_values, y_values)
    if x_finite.size == 0:
        return float("nan")
    in_window = x_finite <= x_max
    if not np.any(in_window):
        return float(np.nanmax(y_finite))
    return float(np.nanmax(y_finite[in_window]))


def style_box_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.tick_params(which="major", direction="in", top=True, right=True, length=4.5, width=0.9)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.5, width=0.8)
    ax.minorticks_on()
    ax.grid(False)


def apply_arial(fig: plt.Figure) -> None:
    for text in fig.findobj(match=matplotlib.text.Text):
        text.set_fontfamily(ARIAL_FONT)


def save_png(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    apply_arial(fig)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="none", transparent=True)
    print(f"[fig06] wrote {output_path}")


def add_residual_annotations(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: Iterable[np.ndarray],
) -> None:
    y_all = np.concatenate([np.asarray(values, dtype=float) for values in y_values])
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size == 0:
        return

    y_min = min(float(np.min(y_all)), 0.0)
    y_max = max(float(np.max(y_all)), 0.0)
    pad = max((y_max - y_min) * 0.13, 0.08)
    ax.set_ylim(y_min - pad, y_max + pad)

    x_min = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))
    x_span = max(x_max - x_min, 1.0)
    y_low, y_high = ax.get_ylim()
    y_span = y_high - y_low

    ax.text(
        x_min + 0.03 * x_span,
        -0.04 * y_span,
        "experiment baseline",
        ha="left",
        va="top",
        fontsize=9.5,
        color="black",
    )


def plot_fig06(strip_labels: bool = False) -> None:
    curves = load_csv_columns(WIDE_DIR / "Bi_SOC_relativistic_sensitivity_main_curves_wide.csv")
    bi_main = load_csv_columns(WIDE_DIR / "Bi_comparison_main_curves_wide.csv")
    residuals = load_csv_columns(
        WIDE_DIR / "Bi_SOC_relativistic_sensitivity_residuals_at_experiment_wide.csv"
    )
    harmonic_soc = load_csv_columns(WIDE_DIR / "Bi_SOC_phonon_harmonic_curves.csv")
    exp = load_csv_columns(WIDE_DIR / "Bi_comparison_experiment_points.csv")

    fig, (ax_main, ax_resid) = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=False)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.88, wspace=0.14)

    main_x_max = max(float(np.nanmax(curves["T_K"])), float(np.nanmax(exp["T_K"])))
    residual_x_max = float(np.nanmax(residuals["T_K"]))

    handles = []
    residual_series = []
    for curve_column, residual_column, label, color, linestyle in SOC_METHODS:
        x_curve, y_curve = finite_xy(curves["T_K"], curves[curve_column])
        (line,) = ax_main.plot(
            x_curve, y_curve, color=color, linestyle=linestyle, linewidth=2.0, label=label
        )
        handles.append(line)
        if residual_column is not None:
            x_res, y_res = finite_xy(residuals["T_K"], residuals[residual_column])
            x_res, y_res = prepend_origin(x_res, y_res)
            ax_resid.plot(x_res, y_res, color=color, linestyle=linestyle, linewidth=1.7)
            residual_series.append(residuals[residual_column])

    x_exp, y_exp = finite_xy(exp["T_K"], exp["B_exp_A2"])
    for source, curve_column, label, color, linestyle in HARMONIC_METHODS:
        source_table = bi_main if source == "bi_main" else harmonic_soc
        x_curve, y_curve = finite_xy(source_table["T_K"], source_table[curve_column])
        in_main_window = x_curve <= main_x_max
        x_curve = x_curve[in_main_window]
        y_curve = y_curve[in_main_window]
        (line,) = ax_main.plot(
            x_curve,
            y_curve,
            color=color,
            linestyle=linestyle,
            linewidth=2.15,
            label=label,
        )
        handles.append(line)
        x_res, y_res = residual_at_experiment(
            exp["T_K"], exp["B_exp_A2"], source_table["T_K"], source_table[curve_column]
        )
        x_res, y_res = prepend_origin(x_res, y_res)
        ax_resid.plot(x_res, y_res, color=color, linestyle=linestyle, linewidth=1.75)
        residual_series.append(y_res)

    ax_main.scatter(x_exp, y_exp, color="black", s=25, zorder=10, label="experiment")

    scalar_y_max = float(np.nanmax([np.nanmax(curves[m[0]]) for m in SOC_METHODS]))
    harmonic_y_max = float(
        max(
            max_in_x_window(
                bi_main["T_K"],
                bi_main["B_A2__harmonic_phonon_spectrum_(isotropic)"],
                main_x_max,
            ),
            max_in_x_window(harmonic_soc["T_K"], harmonic_soc["B_iso_SOC_A2"], main_x_max),
        )
    )
    y_max = max(float(np.nanmax(y_exp)), scalar_y_max, harmonic_y_max)
    ax_main.set_xlim(0.0, main_x_max)
    ax_main.set_ylim(0.0, y_max * 1.12)
    ax_resid.axhline(0.0, color="black", linestyle="--", linewidth=0.95)
    ax_resid.set_xlim(0.0, residual_x_max)
    add_residual_annotations(ax_resid, residuals["T_K"], residual_series)

    import matplotlib.ticker as ticker

    # Sparse y-axis ticks: 4-5 major ticks with clean steps.
    ax_main.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax_resid.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax_main.yaxis.set_minor_locator(ticker.NullLocator())
    ax_resid.yaxis.set_minor_locator(ticker.NullLocator())
    ax_main.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10], integer=True))
    ax_resid.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10], integer=True))
    ax_main.xaxis.set_minor_locator(ticker.NullLocator())
    ax_resid.xaxis.set_minor_locator(ticker.NullLocator())

    style_box_axis(ax_main)
    style_box_axis(ax_resid)

    if strip_labels:
        for ax in (ax_main, ax_resid):
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    else:
        ax_main.set_title(r"Bi: SOC sensitivity")
        ax_resid.set_title("residuals at experiment T")
        ax_main.set_xlabel("temperature (K)")
        ax_resid.set_xlabel("temperature (K)")
        ax_main.set_ylabel(r"$B_{\mathrm{iso}}(T)$ ($\AA^2$)")
        ax_resid.set_ylabel("model - experiment")
        ax_main.legend(
            handles=handles + [plt.Line2D([], [], marker="o", color="black", linestyle="none", label="experiment")],
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            frameon=False,
            fontsize=8.5,
            handlelength=1.8,
            handletextpad=0.5,
            borderaxespad=0.2,
            labelspacing=0.35,
        )

    suffix = "_no_labels" if strip_labels else ""
    png_path = FIGURES_MAIN / f"fig06_Bi_SOC_sensitivity{suffix}.png"
    apply_arial(fig)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="none", transparent=True)
    plt.close(fig)
    print(f"[fig06] wrote {png_path}")


def main() -> None:
    if not WIDE_DIR.exists():
        raise FileNotFoundError(
            f"Cannot find flattened CSV store at {WIDE_DIR}.\n"
            "Regenerate it by running data/export_figure_wide_csvs.py."
        )
    print(f"[fig06] reading CSVs from {WIDE_DIR}")
    # Normal figure with labels.
    plot_fig06(strip_labels=False)
    # No-labels version (keeps data, spines, ticks; strips all text).
    plot_fig06(strip_labels=True)


if __name__ == "__main__":
    main()
