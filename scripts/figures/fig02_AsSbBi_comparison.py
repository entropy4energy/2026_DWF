#!/usr/bin/env python3
"""Render manuscript Fig 2 (As/Sb/Bi combined B_iso(T) comparisons).

Produces the source figure ``fig02_AsSbBi_comparison.png`` in this folder; the
final manuscript figure (``PICS/fig02``) is composed from it.

Layout is 3-column × 2-row composite: comparison panels (top) + residuals (bottom)
for As, Sb, Bi side-by-side. Arial, seaborn colorblind palette,
experiment-baseline and residual-sign annotations.

Inputs (all under the flattened CSV store):
  data/processed/<material>_comparison_main_curves_wide.csv
  data/processed/<material>_comparison_experiment_points.csv
  data/processed/<material>_comparison_residuals_at_experiment_wide.csv
They are produced by scripts/generators/export_figure_wide_csvs.py off the
authoritative ``bt_curves_full.csv`` / ``bt_at_experiment_temperatures.csv``.

Run from this folder or from anywhere:
    python3 fig02_AsSbBi_comparison.py
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
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns


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

sns.set_theme(
    context="paper",
    style="ticks",
    font=ARIAL_FONT,
    rc={
        "font.family": ARIAL_FONT,
        "font.sans-serif": [ARIAL_FONT],
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
        "mathtext.fontset": "custom",
        "mathtext.rm": ARIAL_FONT,
        "mathtext.it": f"{ARIAL_FONT}:italic",
        "mathtext.bf": f"{ARIAL_FONT}:bold",
        "savefig.transparent": False,
    },
)

# Seaborn "deep" palette: the default seaborn family, widely recommended as the
# most readable/harmonious balance of saturation and contrast for line plots.
# It is also close to color-blind safe. Role assignment keeps the hero curve
# (harmonic phonon) as the strongest hue.
#   empirical reference  -> neutral grey  (dashed, secondary role)
#   elastic Debye        -> warm orange
#   quasi-harmonic Debye -> green
#   harmonic phonon      -> blue (hero curve)
_DEEP = sns.color_palette("deep")
_C_EMPIRICAL = "#8c8c8c"
_C_ELASTIC   = _DEEP[1]
_C_QH_DEBYE  = _DEEP[2]
_C_HARMONIC  = _DEEP[0]

# Experimental uncertainty floor for B_exp(T). Fischer (1978) tabulates
# per-point sigma only for Bi (Table 2): sigma ~ 0.06 A^2 at 293 K and
# ~ 0.09 A^2 at 516 K from the profile-isotropic refinement branch. Following
# the Methods "uncertainty floor", we use the midpoint sigma = 0.075 A^2 as a
# single conservative noise floor applied uniformly across As/Sb/Bi.
_EXP_SIGMA_A2 = 0.075

MAIN_METHODS = [
    (
        "B_A2__empirical_digitized_reference",
        "residual__empirical_digitized_reference",
        "empirical reference",
        _C_EMPIRICAL,
        (0, (5, 4)),
    ),
    (
        "B_A2__elastic-derived_Debye",
        "residual__elastic_derived_debye",
        "elastic Debye",
        _C_ELASTIC,
        "-",
    ),
    (
        "B_A2__quasi-harmonic_Debye",
        "residual__quasi_harmonic_debye",
        "quasi-harmonic Debye",
        _C_QH_DEBYE,
        "-",
    ),
    (
        "B_A2__harmonic_phonon_spectrum_(isotropic)",
        "residual__harmonic_phonon_spectrum_isotropic",
        "harmonic phonon",
        _C_HARMONIC,
        "-",
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
    """Anchor residual curves at (0, 0) for presentation without changing source data."""
    if x_values.size == 0:
        return x_values, y_values
    if np.isclose(x_values[0], 0.0) and np.isclose(y_values[0], 0.0):
        return x_values, y_values
    return np.insert(x_values, 0, 0.0), np.insert(y_values, 0, 0.0)


def style_box_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.tick_params(which="major", direction="in", top=True, right=True, length=4.5, width=0.9)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.5, width=0.8)
    ax.grid(False)


def apply_arial(fig: plt.Figure) -> None:
    for text in fig.findobj(match=matplotlib.text.Text):
        text.set_fontfamily(ARIAL_FONT)


def save_png(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    apply_arial(fig)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="none", transparent=True)
    plt.close(fig)
    print(f"[fig02] wrote {output_path}")


def add_residual_annotations(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: Iterable[np.ndarray],
) -> None:
    set_residual_ylim(ax, y_values)

    y_low, y_high = ax.get_ylim()
    y_span = y_high - y_low
    x_min = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))
    x_span = max(x_max - x_min, 1.0)

    baseline_y = 0.0 - 0.08 * y_span
    ax.text(
        x_min + 0.03 * x_span,
        baseline_y,
        "experiment baseline",
        ha="left",
        va="top",
        fontsize=9.5,
        color="black",
    )

    y_all = np.concatenate([np.asarray(values, dtype=float) for values in y_values])
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size == 0:
        return

    y_min = min(float(np.min(y_all)), 0.0)
    y_max = max(float(np.max(y_all)), 0.0)
    arrow_y = y_high - 0.15 * y_span if y_max > abs(y_min) else y_low + 0.18 * y_span
    ax.annotate(
        "",
        xy=(x_min + 0.78 * x_span, arrow_y),
        xytext=(x_min + 0.20 * x_span, arrow_y),
        arrowprops={"arrowstyle": "->", "color": _C_HARMONIC, "linewidth": 1.0},
    )
    ax.text(
        x_min + 0.49 * x_span,
        arrow_y + 0.015 * y_span,
        "residual +",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="black",
    )


def set_residual_ylim(ax: plt.Axes, y_values: Iterable[np.ndarray]) -> None:
    y_all = np.concatenate([np.asarray(values, dtype=float) for values in y_values])
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size == 0:
        return

    y_min = min(float(np.min(y_all)), 0.0)
    y_max = max(float(np.max(y_all)), 0.0)
    pad = max((y_max - y_min) * 0.13, 0.08)
    ax.set_ylim(y_min - pad, y_max + pad)


def comparison_layout() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    fig, (ax_main, ax_resid) = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=False)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.88, wspace=0.14)
    return fig, ax_main, ax_resid


def plot_material_axes(
    material: str,
    ax_main: plt.Axes,
    ax_resid: plt.Axes,
    *,
    compact: bool = False,
    include_legend: bool = True,
    show_ylabels: bool = True,
    annotate_residuals: bool = True,
) -> tuple[list[plt.Line2D], tuple]:
    curves = load_csv_columns(WIDE_DIR / f"{material}_comparison_main_curves_wide.csv")
    exp = load_csv_columns(WIDE_DIR / f"{material}_comparison_experiment_points.csv")
    residuals = load_csv_columns(WIDE_DIR / f"{material}_comparison_residuals_at_experiment_wide.csv")

    main_x_max = max(float(np.nanmax(curves["T_K"])), float(np.nanmax(exp["T_K"])))
    residual_x_max = float(np.nanmax(residuals["T_K"]))
    shared_x_max = residual_x_max if compact else max(main_x_max, residual_x_max)

    handles = []
    residual_series = []
    for curve_column, residual_column, label, color, linestyle in MAIN_METHODS:
        x_curve, y_curve = finite_xy(curves["T_K"], curves[curve_column])
        (line,) = ax_main.plot(
            x_curve,
            y_curve,
            color=color,
            linestyle=linestyle,
            linewidth=1.65 if compact else 2.2,
            label=label,
        )
        handles.append(line)

        x_res, y_res = finite_xy(residuals["T_K"], residuals[residual_column])
        x_res, y_res = prepend_origin(x_res, y_res)
        ax_resid.plot(
            x_res,
            y_res,
            color=color,
            linestyle=linestyle,
            linewidth=1.35 if compact else 1.9,
        )
        residual_series.append(residuals[residual_column])

    x_exp, y_exp = finite_xy(exp["T_K"], exp["B_exp_A2"])
    order = np.argsort(x_exp)
    x_exp_sorted = x_exp[order]
    y_exp_sorted = y_exp[order]
    # Uncertainty band: +/- sigma floor (Bi-derived, Methods) drawn as a light
    # shaded ribbon connecting the experimental points. This visualizes the
    # per-point uncertainty without introducing any fitted curve.
    band = ax_main.fill_between(
        x_exp_sorted,
        y_exp_sorted - _EXP_SIGMA_A2,
        y_exp_sorted + _EXP_SIGMA_A2,
        color="black",
        alpha=0.16,
        linewidth=0.0,
        zorder=8,
        label=r"experiment $\pm\sigma$",
    )
    experiment_scatter = ax_main.scatter(
        x_exp,
        y_exp,
        color="black",
        s=16 if compact else 25,
        zorder=10,
        label="experiment",
    )
    # Composite legend handle: a grey patch (uncertainty band) with the black
    # marker overlaid, displayed as a single legend entry "experiment +/- sigma".
    experiment_handle = (band, experiment_scatter)

    y_max = max(
        float(np.nanmax(y_exp)),
        float(np.nanmax([np.nanmax(curves[m[0]]) for m in MAIN_METHODS])),
    )
    ax_main.set_xlim(0.0, shared_x_max if compact else main_x_max)
    ax_main.set_ylim(0.0, y_max * 1.14)
    ax_resid.axhline(0.0, color="black", linestyle="--", linewidth=0.85 if compact else 0.95)
    ax_resid.set_xlim(0.0, shared_x_max if compact else residual_x_max)
    if annotate_residuals:
        add_residual_annotations(ax_resid, residuals["T_K"], residual_series)
    else:
        set_residual_ylim(ax_resid, residual_series)

    if compact:
        ax_main.set_title(material, fontweight="bold", pad=4, fontsize=9.0)
        ax_main.set_xlabel("")
        ax_main.tick_params(labelbottom=False)
    else:
        ax_main.set_title(f"{material}: $B_{{\\mathrm{{iso}}}}(T)$ comparison")
        ax_resid.set_title("residuals at experiment T")
        ax_main.set_xlabel("temperature (K)")
    ax_resid.set_xlabel("temperature (K)")
    if show_ylabels:
        ax_main.set_ylabel(r"$B_{\mathrm{iso}}(T)$ ($\AA^2$)")
        ax_resid.set_ylabel("model - experiment", labelpad=7 if compact else 10)
    else:
        ax_main.set_ylabel("")
        ax_resid.set_ylabel("")

    import matplotlib.ticker as ticker
    ax_main.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax_resid.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    # Sparse ticks: let MaxNLocator pick 3-4 clean integer positions on both axes,
    # with "nice" steps so labels stay readable at the small compact size.
    ax_main.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10], integer=True))
    ax_resid.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10], integer=True))
    ax_main.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))
    ax_resid.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))
    ax_main.xaxis.set_minor_locator(ticker.NullLocator())
    ax_resid.xaxis.set_minor_locator(ticker.NullLocator())
    ax_main.yaxis.set_minor_locator(ticker.NullLocator())
    ax_resid.yaxis.set_minor_locator(ticker.NullLocator())
    if material == "Sb" and not compact:
        ax_main.text(
            285,
            1.34,
            "harmonic phonon curve tracks experiment",
            color=_C_HARMONIC,
            fontsize=7.4,
            rotation=29,
            ha="center",
            va="center",
        )
    if include_legend:
        ax_main.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            frameon=False,
            fontsize=9.0,
            handlelength=1.8,
            handletextpad=0.5,
            borderaxespad=0.2,
            labelspacing=0.35,
        )

    style_box_axis(ax_main)
    style_box_axis(ax_resid)
    if compact:
        ax_main.tick_params(labelsize=7.8, length=3.6, width=0.8)
        ax_resid.tick_params(labelsize=7.2, length=3.0, width=0.75)
        ax_resid.xaxis.label.set_size(8.2)
        ax_main.yaxis.label.set_size(8.5)
        ax_resid.yaxis.label.set_size(8.0)
    return handles, experiment_handle


def make_standalone_legend_handles() -> list:
    """Fresh legend artists (not tied to any prior figure)."""
    handles: list = []
    for _curve, _resid, _label, color, linestyle in MAIN_METHODS:
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
            )
        )
    band_patch = Patch(facecolor="black", edgecolor="none", alpha=0.16)
    marker = Line2D(
        [],
        [],
        linestyle="None",
        marker="o",
        color="black",
        markersize=6,
    )
    handles.append((band_patch, marker))
    return handles


def plot_comparison_figure(material: str, output_name: str) -> None:
    fig, ax_main, ax_resid = comparison_layout()
    plot_material_axes(material, ax_main, ax_resid)
    save_png(fig, FIGURES_MAIN / output_name)


def plot_combined_stacked_figure(output_name: str, *, strip_labels: bool = False) -> list:
    """Build the 3-column × 2-row composite.

    strip_labels=True  → remove all axis labels, tick labels, x-titles, and
                         the figure legend; keep (a)(b)(c) annotations only.
    Returns the collected legend handles list for downstream use.
    """
    fig = plt.figure(figsize=(7.1, 5.05))
    outer = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 0.4],
        left=0,
        right=0.995,
        bottom=0.092,
        top=0.82,
        wspace=0.12,
        hspace=0.05,
    )

    # Panel letters are added when the figure is assembled in PICS.pptx, not
    # baked in here, so the layout can be rearranged without re-rendering.
    legend_handles = None
    for column, material in enumerate(("As", "Sb", "Bi")):
        ax_main = fig.add_subplot(outer[0, column])
        ax_resid = fig.add_subplot(outer[1, column], sharex=ax_main)
        handles, experiment_handle = plot_material_axes(
            material,
            ax_main,
            ax_resid,
            compact=True,
            include_legend=False,
            show_ylabels=(column == 0) and not strip_labels,
            annotate_residuals=False,
        )
        if strip_labels:
            # Remove all axis labels and tick labels; keep spines and ticks.
            for ax in (ax_main, ax_resid):
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            # Also remove the bold material title above the main panel.
            ax_main.set_title("")
        if legend_handles is None:
            legend_handles = handles + [experiment_handle]

    if legend_handles is not None and not strip_labels:
        # Align legend width to the combined bounding box of the 3 top panels.
        fig.canvas.draw()
        top_axes = [ax for ax in fig.axes if ax.get_subplotspec().rowspan.start == 0]
        renderer = fig.canvas.get_renderer()
        panel_bboxes = [ax.get_window_extent(renderer) for ax in top_axes]
        panel_width_px = max(b.x1 for b in panel_bboxes) - min(b.x0 for b in panel_bboxes)
        panel_center_x_frac = (
            min(b.x0 for b in panel_bboxes) + max(b.x1 for b in panel_bboxes)
        ) / 2.0 / fig.bbox.width

        legend_labels = [method[2] for method in MAIN_METHODS] + [r"experiment $\pm\sigma$"]
        base_fontsize = 10.0
        leg = fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="upper center",
            bbox_to_anchor=(panel_center_x_frac, 0.975),
            bbox_transform=fig.transFigure,
            ncol=5,
            frameon=False,
            fontsize=base_fontsize,
            handlelength=2.0,
            handletextpad=0.5,
            columnspacing=1.0,
            borderaxespad=0.0,
            handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)},
        )
        current_fontsize = base_fontsize
        for _ in range(4):
            fig.canvas.draw()
            leg_bbox_px = leg.get_window_extent(fig.canvas.get_renderer())
            if leg_bbox_px.width <= 0:
                break
            ratio = panel_width_px / leg_bbox_px.width
            if abs(ratio - 1.0) < 0.005:
                break
            current_fontsize *= ratio
            for txt in leg.get_texts():
                txt.set_fontsize(current_fontsize)
    save_png(fig, FIGURES_MAIN / output_name)
    return legend_handles or []


def plot_legend_only_figure(output_name: str) -> None:
    """Save a standalone horizontal legend strip (transparent background)."""
    legend_labels = [method[2] for method in MAIN_METHODS] + [r"experiment $\pm\sigma$"]
    legend_handles = make_standalone_legend_handles()

    # Minimal invisible axes just to host the figure legend.
    fig, ax = plt.subplots(figsize=(7.1, 0.45))
    ax.set_visible(False)

    leg = fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="center",
        ncol=5,
        frameon=False,
        fontsize=10.0,
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=1.0,
        borderaxespad=0.0,
        handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)},
    )
    apply_arial(fig)
    (FIGURES_MAIN / output_name).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES_MAIN / output_name,
        dpi=300,
        bbox_inches="tight",
        facecolor="none",
        transparent=True,
    )
    plt.close(fig)
    print(f"[fig02] wrote {BASE_DIR / output_name}")


def main() -> None:
    if not WIDE_DIR.exists():
        raise FileNotFoundError(
            f"Cannot find flattened CSV store at {WIDE_DIR}.\n"
            "Regenerate it by running data/export_figure_wide_csvs.py."
        )
    print(f"[fig02] reading CSVs from {WIDE_DIR}")

    # 1. Normal combined figure (transparent background).
    legend_handles = plot_combined_stacked_figure("fig02_AsSbBi_comparison.png")

    # 2. Combined figure without axis labels / tick labels (keeps (a)(b)(c)).
    plot_combined_stacked_figure("fig02_AsSbBi_no_labels.png", strip_labels=True)

    # 3. Standalone legend strip.
    plot_legend_only_figure("fig02_legend_only.png")


if __name__ == "__main__":
    main()
