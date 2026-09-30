#!/usr/bin/env python3
"""Render manuscript Fig 5 (Bi scalar-relativistic phonon-DOS diagnostic) and
its CSV tables."""

from __future__ import annotations

import csv
import lzma
import os
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
# Every path resolves through scripts/paths.py, the single source of truth for
# this package. Nothing outside the repository root is read or written.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import FIGURES_MAIN, PROCESSED, REPO_ROOT, VERIFICATION_LOGS, calc_dir  # noqa: E402

PROJECT_ROOT = REPO_ROOT
PHDOS_PATH = calc_dir("Bi") / "PHDOSCAR.xz"
# Tables go to data/processed/, renders to figures/main/.
CSV_DIR = PROCESSED
FIG_DIR = FIGURES_MAIN
CACHE_DIR = VERIFICATION_LOGS / ".cache"

THZ_PER_EV = 241.7989242084918
LEVERAGE_CUTOFF_THZ = 0.10
WINDOWS = [
    ("0-1", 0.0, 1.0),
    ("1-2", 1.0, 2.0),
    ("2-4", 2.0, 4.0),
    (">4", 4.0, None),
]


def setup_matplotlib():
    (CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
    os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))

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

    plt.rcParams.update(
        {
            "font.family": arial,
            "font.sans-serif": [arial],
            "mathtext.fontset": "custom",
            "mathtext.rm": arial,
            "mathtext.it": f"{arial}:italic",
            "mathtext.bf": f"{arial}:bold",
            "axes.linewidth": 1.0,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.transparent": True,
        }
    )
    return plt


def read_phdos(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float]] = []
    with lzma.open(path, "rt", encoding="utf-8") as infile:
        for idx, raw_line in enumerate(infile):
            if idx < 6:
                continue
            parts = raw_line.split()
            if len(parts) < 2:
                continue
            rows.append((float(parts[0]) * THZ_PER_EV, float(parts[1])))

    if not rows:
        raise ValueError(f"No phonon-DOS rows found in {path}")

    data = np.asarray(rows, dtype=float)
    return data[:, 0], data[:, 1]


def integrate_window(x_values: np.ndarray, y_values: np.ndarray, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    mask = (x_values >= lo) & (x_values <= hi)
    xs = x_values[mask]
    ys = y_values[mask]
    if xs.size == 0:
        xs = np.asarray([lo, hi], dtype=float)
        ys = np.interp(xs, x_values, y_values)
    else:
        if xs[0] > lo:
            xs = np.insert(xs, 0, lo)
            ys = np.insert(ys, 0, np.interp(lo, x_values, y_values))
        if xs[-1] < hi:
            xs = np.append(xs, hi)
            ys = np.append(ys, np.interp(hi, x_values, y_values))
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trapz(ys, xs))


def first_prominent_peak(freq: np.ndarray, dos: np.ndarray) -> float:
    if freq.size < 3:
        return float(freq[int(np.argmax(dos))])
    local_maxima = np.where((dos[1:-1] >= dos[:-2]) & (dos[1:-1] >= dos[2:]))[0] + 1
    threshold = 0.15 * float(np.max(dos))
    candidates = [idx for idx in local_maxima if freq[idx] >= 0.5 and dos[idx] >= threshold]
    if candidates:
        return float(freq[candidates[0]])
    if local_maxima.size:
        return float(freq[local_maxima[0]])
    return float(freq[int(np.argmax(dos))])


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_number(value: float) -> str:
    return f"{value:.6g}"


def main() -> None:
    freq_all, dos_all = read_phdos(PHDOS_PATH)
    max_dos = float(np.max(dos_all))
    imaginary_mask = (freq_all < -1.0e-3) & (dos_all > 1.0e-6 * max_dos)
    has_imaginary = bool(np.any(imaginary_mask))

    positive_mask = freq_all > 0.0
    freq = freq_all[positive_mask]
    dos = dos_all[positive_mask]
    freq_max = float(freq[-1])

    total_dos = integrate_window(freq, dos, 0.0, freq_max)
    leverage_mask = freq >= LEVERAGE_CUTOFF_THZ
    leverage_freq = freq[leverage_mask]
    leverage_density = dos[leverage_mask] / (leverage_freq**2)
    total_leverage = integrate_window(leverage_freq, leverage_density, LEVERAGE_CUTOFF_THZ, freq_max)

    window_rows: list[dict[str, object]] = []
    for label, lo, hi_raw in WINDOWS:
        hi = freq_max if hi_raw is None else min(hi_raw, freq_max)
        if hi <= lo:
            dos_integral = 0.0
            leverage_integral = 0.0
        else:
            dos_integral = integrate_window(freq, dos, lo, hi)
            lev_lo = max(lo, LEVERAGE_CUTOFF_THZ)
            leverage_integral = (
                integrate_window(leverage_freq, leverage_density, lev_lo, hi)
                if hi > LEVERAGE_CUTOFF_THZ
                else 0.0
            )
        window_rows.append(
            {
                "frequency_window_THz": label,
                "dos_fraction": format_number(dos_integral / total_dos if total_dos else 0.0),
                "classical_leverage_proxy_fraction": format_number(
                    leverage_integral / total_leverage if total_leverage else 0.0
                ),
            }
        )

    dos_rows = [
        {
            "frequency_THz": format_number(freq_value),
            "dos_raw": format_number(dos_value),
            "dos_normalized_to_max": format_number(dos_value / max_dos),
            "classical_leverage_proxy_g_over_omega2": format_number(
                dos_value / (freq_value**2) if freq_value >= LEVERAGE_CUTOFF_THZ else 0.0
            ),
        }
        for freq_value, dos_value in zip(freq, dos)
    ]

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        CSV_DIR / "Bi_noSOC_phonon_dos.csv",
        dos_rows,
        [
            "frequency_THz",
            "dos_raw",
            "dos_normalized_to_max",
            "classical_leverage_proxy_g_over_omega2",
        ],
    )
    write_csv(
        CSV_DIR / "Bi_noSOC_low_frequency_leverage.csv",
        window_rows,
        [
            "frequency_window_THz",
            "dos_fraction",
            "classical_leverage_proxy_fraction",
        ],
    )

    first_peak = first_prominent_peak(freq, dos)
    sub2_dos = sum(float(row["dos_fraction"]) for row in window_rows[:2])
    sub2_leverage = sum(float(row["classical_leverage_proxy_fraction"]) for row in window_rows[:2])

    plt = setup_matplotlib()

    def build_figure(strip_labels: bool = False):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1.35, 1.0]})
        ax_dos, ax_bar = axes

        norm_dos = dos / max_dos
        ax_dos.plot(freq, norm_dos, color="#1d4ed8", lw=1.9)
        ax_dos.fill_between(freq, 0, norm_dos, where=freq <= 1.0, color="#f59e0b", alpha=0.30, lw=0)
        ax_dos.fill_between(freq, 0, norm_dos, where=(freq > 1.0) & (freq <= 2.0), color="#10b981", alpha=0.26, lw=0)
        ax_dos.fill_between(freq, 0, norm_dos, where=(freq > 2.0) & (freq <= 4.0), color="#64748b", alpha=0.22, lw=0)
        ax_dos.axvline(1.0, color="0.35", ls="--", lw=0.8)
        ax_dos.axvline(2.0, color="0.35", ls="--", lw=0.8)
        if not strip_labels:
            ax_dos.text(
                0.04,
                0.95,
                "no significant\nimaginary DOS",
                transform=ax_dos.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
            )
        ax_dos.set_xlim(0.0, max(4.5, freq_max))
        ax_dos.set_ylim(0.0, 1.08)

        colors = ["#f59e0b", "#10b981", "#64748b", "#94a3b8"]
        x_positions = np.arange(len(WINDOWS))
        dos_percent = np.asarray([float(row["dos_fraction"]) * 100.0 for row in window_rows])
        leverage_percent = np.asarray(
            [float(row["classical_leverage_proxy_fraction"]) * 100.0 for row in window_rows]
        )
        bar_width = 0.36
        ax_bar.bar(
            x_positions - bar_width / 2,
            dos_percent,
            width=bar_width,
            color=colors,
            alpha=0.72,
            label="integrated DOS",
        )
        ax_bar.bar(
            x_positions + bar_width / 2,
            leverage_percent,
            width=bar_width,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            hatch="///",
            alpha=0.72,
            label=r"$g(f)/f^2$ fraction",
        )
        ax_bar.set_xticks(x_positions)
        ax_bar.set_ylim(0.0, 70.0)

        import matplotlib.ticker as ticker

        ax_dos.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax_dos.yaxis.set_minor_locator(ticker.NullLocator())
        ax_bar.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax_bar.yaxis.set_minor_locator(ticker.NullLocator())

        for ax in axes:
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(1.0)
            ax.tick_params(which="major", direction="in", top=True, right=True, length=4.0, width=0.8)
            ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.3, width=0.7)
            ax.grid(False)

        if strip_labels:
            for ax in axes:
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_title("")
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            ax_bar.set_xticklabels([])
        else:
            ax_dos.set_xlabel("Frequency (THz)")
            ax_dos.set_ylabel("Bi phonon DOS (normalized)")
            ax_dos.set_title("Bi phonon DOS")
            ax_bar.set_xticklabels([row["frequency_window_THz"] for row in window_rows])
            ax_bar.set_ylabel("Window share (%)")
            ax_bar.set_xlabel("Frequency window (THz)")
            ax_bar.set_title("Inverse-square-weighted DOS")
            ax_bar.legend(frameon=False, loc="upper right", handlelength=1.4)
            #ax_bar.text(
            #    0.03,
            #    0.95,
            #    f"<2 THz: {sub2_dos * 100:.1f}% int. DOS\n{sub2_leverage * 100:.1f}% leverage proxy",
            #    transform=ax_bar.transAxes,
            #    ha="left",
            #    va="top",
            #    fontsize=7.4,
            #)

        fig.tight_layout(w_pad=1.8)
        return fig

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Normal figure with labels.
    png_path = FIG_DIR / "fig05_Bi_phonon_dos.png"
    fig_normal = build_figure(strip_labels=False)
    fig_normal.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="none", transparent=True)
    plt.close(fig_normal)

    # No-labels version (keeps data and spines, strips all text).
    no_labels_path = FIG_DIR / "fig05_Bi_phonon_dos_no_labels.png"
    fig_stripped = build_figure(strip_labels=True)
    fig_stripped.savefig(no_labels_path, dpi=300, bbox_inches="tight", facecolor="none", transparent=True)
    plt.close(fig_stripped)

    print(f"Wrote {png_path}")
    print(f"Wrote {CSV_DIR / 'Bi_noSOC_low_frequency_leverage.csv'}")
    print(f"First prominent DOS peak: {first_peak:.3f} THz")
    print(f"Integrated DOS weight below 2 THz: {sub2_dos * 100:.1f}%")
    print(f"Inverse-square-weighted DOS fraction below 2 THz: {sub2_leverage * 100:.1f}%")
    print(f"Significant imaginary DOS: {'yes' if has_imaginary else 'no'}")


if __name__ == "__main__":
    main()
