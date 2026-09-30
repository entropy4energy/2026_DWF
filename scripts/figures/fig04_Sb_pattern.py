#!/usr/bin/env python3
"""Sb 294 K Cu-Kalpha simulated powder pattern under four DWF treatments.

The A7 Sb structure is used as the most-discriminating A7 test case (harmonic phonon B(294 K) matches
Fischer 1978 experimental B within 0.04 A^2, while the three scalar Debye
treatments underpredict by 0.4-0.6 A^2). The script produces:

(1) bt4 comparison: I(2theta) for each of {no DWF, 4 model DWFs, experimental}
(2) Delta I / I residual panel vs the experimental DWF baseline
(3) Stick pattern with (hkl) labels as a companion SI figure
(4) CSV dump of all per-peak data for traceability

Sources
-------
Structure: data/dft_raw/Sb/CONTCAR.static.xz
B values:  bt_curves_full.csv (linearly interpolated to T=294 K)
B_exp:     bt_at_experiment_temperatures.csv (T=294 K, Fischer 1978 anchor)

Geometry: Cu K-alpha_1 lambda = 1.5406 A (single-wavelength; Ka2 not included);
          2theta in [10, 100] deg; pseudo-Voigt profiles FWHM 0.10 deg; Lorentz-
          polarization factor for unpolarized Cu tube; atomic form factor for Sb
          using the four-Gaussian + c tabulation (Waasmaier & Kirfel 1995;
          refitted values commonly tabulated in International Tables Vol. C).
"""
from __future__ import annotations

import csv
import lzma
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent


# Every path resolves through scripts/paths.py, the single source of truth for
# this package. Nothing outside the repository root is read or written.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import FIGURES_MAIN, PROCESSED, calc_dir  # noqa: E402

_CSV_DIR = PROCESSED


CONTCAR_XZ = calc_dir("Sb") / "CONTCAR.static.xz"
BT_CURVES_CSV = PROCESSED / "bt_curves_full.csv"
BT_EXP_CSV = PROCESSED / "bt_at_experiment_temperatures.csv"

# Tables go to data/processed/, renders to figures/main/.
CSV_DIR = PROCESSED
FIG_DIR = FIGURES_MAIN

LAMBDA_CU_KA1 = 1.540598  # Angstrom; Cu K-alpha_1 CODATA
TWO_THETA_MIN = 10.0
TWO_THETA_MAX = 180.0
TWO_THETA_STEP = 0.02  # deg
PV_FWHM_DEG = 0.10    # pseudo-Voigt FWHM, instrumental + specimen broadening lumped
PV_ETA = 0.5          # pseudo-Voigt mixing (0=Gaussian, 1=Lorentzian)

# Window used for the main-panel zoom (where DWF attenuation dominates the
# intensity contrast) and for the high-angle highlight on the dI/I panel.
HIGH_ANGLE_WINDOW = (70.0, 180.0)  # deg

SB_Z = 51  # atomic number

# Waasmaier-Kirfel 1995 atomic scattering factor parameters for Sb (neutral atom).
# f0(s) = sum_i a_i exp(-b_i s^2) + c, s = sin(theta)/lambda [A^{-1}].
# Reference: D. Waasmaier, A. Kirfel, Acta Cryst. A51, 416 (1995), Table 1.
SB_WK_A = np.array([3.564709, 6.734777, 19.33200, 10.30818, 11.65350])
SB_WK_B = np.array([0.504098, 14.55389, 1.209340, 51.02182, 0.007952])
SB_WK_C = 0.0

# Dispersion corrections for Cu K-alpha (International Tables Vol. C, Table 4.2.6.8).
# Sb at Cu K-alpha: f' = -0.591, f'' = 5.484 (standard values).
# Small for Sb at Cu energy; included for completeness.
SB_F_PRIME = -0.591
SB_F_DPRIME = 5.484


def log(msg: str) -> None:
    print(f"[pc4] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Structure loader (VASP CONTCAR format, Sb A7 primitive hexagonal-setting)
# ---------------------------------------------------------------------------

def read_contcar(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (lattice 3x3 in Angstrom, frac_coords Nx3, species list)."""
    with lzma.open(path, "rt") as fh:
        lines = fh.readlines()
    scale = float(lines[1].strip())
    lattice = np.array(
        [[float(x) for x in lines[2 + i].split()] for i in range(3)]
    )
    if scale < 0:  # VASP scale = -|V| means volume in A^3
        current_vol = abs(np.linalg.det(lattice))
        target_vol = abs(scale)
        lattice *= (target_vol / current_vol) ** (1.0 / 3.0)
    else:
        lattice *= scale
    species_labels = lines[5].split()
    counts = [int(n) for n in lines[6].split()]
    species = []
    for label, n in zip(species_labels, counts):
        species.extend([label] * n)
    coord_mode = lines[7].strip().lower()
    if not coord_mode.startswith("d"):
        raise ValueError(f"Expected Direct coordinates, got: {coord_mode}")
    frac = np.array(
        [[float(x) for x in lines[8 + i].split()[:3]] for i in range(sum(counts))]
    )
    return lattice, frac, species


def read_b_values() -> tuple[dict[str, float], float, float]:
    """Return (model_B_dict, B_exp_at_294K, T_exp_K).

    Model B values are linearly interpolated to T = T_exp (Fischer 1978 anchor,
    294 K) from the bracketing tabulated rows of bt_curves_full.csv. The
    experimental B is taken directly from the Fischer 294 K row.
    """
    T_TARGET = 294.0
    by_method: dict[str, list[tuple[float, float]]] = {}
    with open(BT_CURVES_CSV) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["material"] == "Sb" and row["soc_status"] == "noSOC":
                if row["B_A2"].strip().upper() in {"NA", "NAN", ""}:
                    continue
                by_method.setdefault(row["method"], []).append(
                    (float(row["T_K"]), float(row["B_A2"]))
                )
    want = {
        "empirical digitized reference",
        "elastic-derived Debye",
        "quasi-harmonic Debye",
        "harmonic phonon spectrum (isotropic)",
    }
    missing = want - set(by_method)
    if missing:
        raise RuntimeError(f"Missing Sb B(T) rows for: {missing}")
    model_B: dict[str, float] = {}
    for method in want:
        pts = sorted(by_method[method])
        T_arr = np.array([t for t, _ in pts])
        B_arr = np.array([b for _, b in pts])
        if T_TARGET < T_arr.min() or T_TARGET > T_arr.max():
            raise RuntimeError(
                f"T={T_TARGET} K outside tabulated range for method '{method}'"
            )
        model_B[method] = float(np.interp(T_TARGET, T_arr, B_arr))
    B_exp = None
    T_exp = None
    with open(BT_EXP_CSV) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["material"] == "Sb" and row["soc_status"] == "noSOC":
                T = float(row["T_K"])
                if abs(T - T_TARGET) < 1.0:
                    B_exp = float(row["B_exp_A2"])
                    T_exp = T
                    break
    if B_exp is None:
        raise RuntimeError("Missing Fischer Sb B_exp @ T=294 K row")
    return model_B, B_exp, T_exp


# ---------------------------------------------------------------------------
# Powder diffraction pattern (crystal-symmetry-naive HKL enumeration)
# ---------------------------------------------------------------------------

def sb_form_factor(s: np.ndarray) -> np.ndarray:
    """f0(s) from Waasmaier-Kirfel plus dispersion corrections. Returns complex array."""
    s2 = s * s
    f0 = np.sum(
        SB_WK_A[:, None] * np.exp(-SB_WK_B[:, None] * s2[None, :]),
        axis=0,
    ) + SB_WK_C
    return f0 + SB_F_PRIME + 1j * SB_F_DPRIME


def index_bounds(lattice: np.ndarray, lambda_xr: float) -> list[int]:
    """Complete per-axis Miller-index bounds implied by the Ewald limit.

    Since h_i = G . a_i / 2 pi, the limit |G| <= 2 pi / d_min = 4 pi / lambda
    gives |h_i| <= 2 |a_i| / lambda. A fixed cube is *not* equivalent: at Cu
    K-alpha_1 the A7 c axis (11.47 A) needs |l| <= 14, so the +/-10 cube used
    before dropped 56 non-extinct reflections carrying about 4.6% of the
    integrated intensity, the strongest of them at 2theta = 148.6 deg with 4.1%
    of the tallest peak. The per-reflection dI/I maxima of panel (c) are
    unaffected -- they sit on the highest-s reflection, which the cube retained
    -- but the profiles of panel (a) and the whole-pattern metrics of panel (b)
    are not.
    """
    return [int(np.floor(2.0 * float(np.linalg.norm(lattice[i])) / lambda_xr)) + 1
            for i in range(3)]


def enumerate_hkl(
    lattice: np.ndarray, lambda_xr: float, two_theta_max_deg: float,
    bounds: list[int] | None = None
) -> list[tuple[int, int, int, float, float]]:
    """Return [(h, k, l, d_spacing, two_theta_deg), ...] inside the 2theta window.

    Uses reciprocal lattice from the real-space lattice. Enumerates the complete
    Ewald-limited index range (see `index_bounds`) and keeps those reflections
    with 2theta <= cutoff.
    """
    recip = 2 * np.pi * np.linalg.inv(lattice).T  # rows: b1,b2,b3 (A^-1)
    k_max_mag = 2.0 * np.sin(np.radians(two_theta_max_deg / 2.0)) / lambda_xr
    if bounds is None:
        bounds = index_bounds(lattice, lambda_xr)
    hkls: list[tuple[int, int, int, float, float]] = []
    for h in range(-bounds[0], bounds[0] + 1):
        for k in range(-bounds[1], bounds[1] + 1):
            for l in range(-bounds[2], bounds[2] + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                G = h * recip[0] + k * recip[1] + l * recip[2]
                q = np.linalg.norm(G) / (2 * np.pi)  # q = 1/d (A^-1)
                if q > k_max_mag:
                    continue
                d = 1.0 / q
                sin_theta = lambda_xr / (2.0 * d)
                if sin_theta >= 1.0:
                    continue
                two_theta = 2.0 * np.degrees(np.arcsin(sin_theta))
                if two_theta < 2.0 or two_theta > two_theta_max_deg:
                    continue
                hkls.append((h, k, l, d, two_theta))
    return hkls


def group_equivalent(
    hkls: list[tuple[int, int, int, float, float]],
    d_tol: float = 1e-4,
) -> list[tuple[list[tuple[int, int, int]], float, float, int]]:
    """Group reflections sharing the same d-spacing as one multiplicity class.

    Returns [(list_of_hkl, d, two_theta, multiplicity), ...] sorted by d desc
    (i.e. 2theta ascending).
    """
    hkls_sorted = sorted(hkls, key=lambda x: -x[3])
    groups: list[tuple[list[tuple[int, int, int]], float, float, int]] = []
    used = [False] * len(hkls_sorted)
    for i, (h, k, l, d, tt) in enumerate(hkls_sorted):
        if used[i]:
            continue
        bucket = [(h, k, l)]
        used[i] = True
        for j in range(i + 1, len(hkls_sorted)):
            if used[j]:
                continue
            if abs(hkls_sorted[j][3] - d) < d_tol:
                bucket.append(hkls_sorted[j][:3])
                used[j] = True
        groups.append((bucket, d, tt, len(bucket)))
    return groups


def structure_factor(
    hkls_group: list[tuple[int, int, int]],
    frac: np.ndarray,
    f_atom: complex,
) -> complex:
    """Structure factor F for the first reflection in `hkls_group`.

    Because all atoms in the A7 Sb primitive cell are the same species, the
    structure factor reduces to f_atom * sum_n exp(2pi i (h x_n + k y_n + l z_n)).
    Reflections grouped by a common d-spacing are not necessarily symmetry-
    equivalent (the A7 cell has accidental d-coincidences), so callers must sum
    |F|^2 over every group member rather than scaling one representative.
    """
    h, k, l = hkls_group[0]
    phase = np.exp(2j * np.pi * (h * frac[:, 0] + k * frac[:, 1] + l * frac[:, 2]))
    return f_atom * np.sum(phase)


def lp_factor(two_theta_deg: np.ndarray) -> np.ndarray:
    """Lorentz-polarization for an unpolarized lab X-ray source (no monochromator)."""
    tt = np.radians(two_theta_deg)
    theta = tt / 2.0
    # Lorentz: 1 / (sin(theta)^2 cos(theta)); polarization: (1 + cos^2(2theta)) / 2.
    L = 1.0 / (np.sin(theta) ** 2 * np.cos(theta))
    P = (1.0 + np.cos(tt) ** 2) / 2.0
    return L * P


def pseudo_voigt(
    x: np.ndarray, center: float, fwhm: float, eta: float
) -> np.ndarray:
    """Normalized pseudo-Voigt (area = 1) centered at `center`."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = fwhm / 2.0
    G = np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    L = (gamma / np.pi) / ((x - center) ** 2 + gamma ** 2)
    return eta * L + (1.0 - eta) * G


def compute_peak_intensities(
    lattice: np.ndarray,
    frac: np.ndarray,
) -> list[dict]:
    """Return a list of peak records; intensity is the static (B=0) intensity
    at the Bragg angle including multiplicity, |F|^2, Lorentz-polarization.

    Each record contains: hkl_rep, multiplicity, d_A, two_theta_deg, s_invA,
    I_static (pre-DWF).
    """
    hkls = enumerate_hkl(lattice, LAMBDA_CU_KA1, TWO_THETA_MAX)
    groups = group_equivalent(hkls)
    two_thetas = np.array([g[2] for g in groups])
    lp = lp_factor(two_thetas)
    peaks: list[dict] = []
    for (reps, d, tt, mult), lp_i in zip(groups, lp):
        s = 0.5 / d  # s = sin(theta)/lambda = 1 / (2 d)
        f_atom = complex(sb_form_factor(np.array([s]))[0])
        # Reflections sharing this d-spacing may be accidentally (not
        # symmetry-) degenerate, so |F|^2 differs between them. Sum |F|^2 over
        # every group member instead of scaling one representative by `mult`.
        sum_F2 = sum(
            (lambda F: F.real ** 2 + F.imag ** 2)(structure_factor([hkl], frac, f_atom))
            for hkl in reps
        )
        I_static = sum_F2 * lp_i
        peaks.append(
            {
                "hkl_rep": reps[0],
                "multiplicity": mult,
                "d_A": d,
                "two_theta_deg": tt,
                "s_invA": s,
                "I_static": float(I_static),
            }
        )
    peaks.sort(key=lambda p: p["two_theta_deg"])
    return peaks


def apply_dwf(peaks: list[dict], B_A2: float) -> np.ndarray:
    """Return per-peak integrated intensities after DWF attenuation exp(-2 B s^2)."""
    out = np.zeros(len(peaks))
    for i, p in enumerate(peaks):
        out[i] = p["I_static"] * np.exp(-2.0 * B_A2 * p["s_invA"] ** 2)
    return out


def convolve_profile(
    peaks: list[dict],
    peak_int: np.ndarray,
    two_theta_grid: np.ndarray,
    fwhm: float = PV_FWHM_DEG,
    eta: float = PV_ETA,
) -> np.ndarray:
    """Return I(2theta) on `two_theta_grid` using pseudo-Voigt profiles per peak."""
    pattern = np.zeros_like(two_theta_grid)
    for p, I in zip(peaks, peak_int):
        pattern += I * pseudo_voigt(two_theta_grid, p["two_theta_deg"], fwhm, eta)
    return pattern


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_peak_csv(
    peaks: list[dict],
    B_values: dict[str, float],
    out_path: Path,
) -> None:
    """Per-peak dump with hkl, 2theta, d, |F|^2, and DWF-attenuated intensities."""
    method_order = [
        "no DWF (static)",
        "empirical digitized reference",
        "elastic-derived Debye",
        "quasi-harmonic Debye",
        "harmonic phonon spectrum (isotropic)",
        "experimental (Fischer 1978)",
    ]
    attenuated = {m: apply_dwf(peaks, B_values[m]) for m in method_order}
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        header = [
            "h", "k", "l",
            "multiplicity",
            "d_A",
            "two_theta_deg",
            "s_invA",
            "I_static_prefactor",
        ] + [f"I_{m.replace(' ', '_').replace('(', '').replace(')', '')}_B={B_values[m]:.4f}"
             for m in method_order]
        w.writerow(header)
        for i, p in enumerate(peaks):
            row = [
                p["hkl_rep"][0], p["hkl_rep"][1], p["hkl_rep"][2],
                p["multiplicity"],
                f"{p['d_A']:.6f}",
                f"{p['two_theta_deg']:.4f}",
                f"{p['s_invA']:.6f}",
                f"{p['I_static']:.4f}",
            ] + [f"{attenuated[m][i]:.4f}" for m in method_order]
            w.writerow(row)


def write_pattern_csv(
    two_theta: np.ndarray,
    patterns: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    methods = list(patterns.keys())
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["two_theta_deg"] + methods)
        for i, tt in enumerate(two_theta):
            w.writerow([f"{tt:.4f}"] + [f"{patterns[m][i]:.6e}" for m in methods])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "empirical digitized reference":         "#8c8c8c",
    "elastic-derived Debye":                 "#DD8452",
    "quasi-harmonic Debye":                  "#55A868",
    "harmonic phonon spectrum (isotropic)":  "#4C72B0",
    "experimental (Fischer 1978)":           "black",
    "no DWF (static)":                       "0.5",
}

METHOD_LABELS = {
    "empirical digitized reference":         r"empirical Debye-function ($B=0.686$)",
    "elastic-derived Debye":                 r"elastic-derived Debye ($B=0.658$)",
    "quasi-harmonic Debye":                  r"quasi-harmonic Debye ($B=0.478$)",
    "harmonic phonon spectrum (isotropic)":  r"harmonic phonon, isotropic ($B=1.061$)",
    "experimental (Fischer 1978)":           r"experimental Fischer 1978 ($B=1.097$)",
    "no DWF (static)":                       r"no DWF, $B=0$ (static)",
}


def plot_main_figure(
    two_theta: np.ndarray,
    patterns: dict[str, np.ndarray],
    B_values: dict[str, float],
    peaks: list[dict],
    out_base: Path,
) -> dict[str, float]:
    """Single-panel figure: per-peak dI/I vs. Fischer experiment.

    Saves two variants:
      <out_base>.png         — full version with title, axis labels, legend
      <out_base>_nolabel.png — data-only version (no text of any kind)

    Full 2theta range (10-180 deg); HIGH_ANGLE_WINDOW highlighted in gold.
    Returns per-method max |dI/I| inside HIGH_ANGLE_WINDOW.
    """
    import matplotlib.pyplot as plt

    win_lo, win_hi = HIGH_ANGLE_WINDOW

    peak_tt = np.array([p["two_theta_deg"] for p in peaks])
    peak_s  = np.array([p["s_invA"]       for p in peaks])
    window_peak_mask = (peak_tt >= win_lo) & (peak_tt <= win_hi)
    max_abs_di_over_i: dict[str, float] = {}

    method_order = [
        "no DWF (static)",
        "empirical digitized reference",
        "elastic-derived Debye",
        "quasi-harmonic Debye",
        "harmonic phonon spectrum (isotropic)",
    ]

    # Pre-compute per-method dI/I arrays once.
    # Append an analytical endpoint at 2theta=180 (s = 1/lambda) so curves touch right edge.
    s_at_180 = 1.0 / LAMBDA_CU_KA1
    peak_tt_ext = np.append(peak_tt, 180.0)
    peak_s_ext  = np.append(peak_s,  s_at_180)
    window_peak_mask_ext = np.append(window_peak_mask, False)

    di_arrays: dict[str, np.ndarray] = {}
    for m in method_order:
        dB = B_values[m] - B_values["experimental (Fischer 1978)"]
        di_arrays[m] = 100.0 * (np.exp(-2.0 * dB * peak_s_ext ** 2) - 1.0)
        max_abs_di_over_i[m] = (
            float(np.max(np.abs(di_arrays[m][window_peak_mask_ext])))
            if np.any(window_peak_mask_ext) else float("nan")
        )

    y_data_max = max(float(np.max(v)) for v in di_arrays.values())

    def _draw(ax: "plt.Axes", labeled: bool) -> None:
        """Plot data onto ax. If labeled=False, suppress all text."""
        import matplotlib.ticker as ticker

        for m in method_order:
            ls = "--" if m == "no DWF (static)" else "-"
            lw = 2.0 if m == "harmonic phonon spectrum (isotropic)" else 1.3
            ax.plot(
                peak_tt_ext, di_arrays[m],
                color=METHOD_COLORS[m], linestyle=ls, linewidth=lw, alpha=0.92,
                marker="o", markersize=3.5,
                label=METHOD_LABELS[m] if labeled else None,
            )
        ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
        ax.set_xlim(TWO_THETA_MIN, TWO_THETA_MAX)
        ax.set_ylim(bottom=-20)
        ax.grid(False)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        # suppress -20 tick label
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: "" if int(round(x)) == -20 else f"{int(round(x))}")
        )

        if labeled:
            #ax.text(
            #    (win_lo + win_hi) / 2.0, y_data_max * 1.01,
            #    fr"highlighted: $2\theta \in [{win_lo:.0f}^{{\circ}}\!,\,{win_hi:.0f}^{{\circ}}]$",
            #    ha="center", va="bottom", fontsize=8.5, color="goldenrod",
            #)
            ax.set_xlabel(
                r"$2\theta$ (deg)",
                fontsize=10,
            )
            ax.set_ylabel(r"$\Delta I / I$ vs. experiment (%)", fontsize=10)
            #ax.set_title(
            #    r"Sb 300 K: intensity deviation from Fischer 1978 experiment"
            #    "\nper reflection, four DWF treatments vs. static ($B=0$)",
            #    fontsize=10, fontweight="bold",
            #)
            ax.legend(
                fontsize=8.0, loc="upper left", bbox_to_anchor=(0.01, 0.99),
                frameon=False, ncol=1, handlelength=2.2, borderaxespad=0.3,
            )
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(labelbottom=False, labelleft=False)

    # --- labeled version ---
    # Trial layout requested by the user: swap panel A/B sizes, so panel A now
    # uses the former compact panel-B footprint.
    fig, ax = plt.subplots(figsize=(5.04, 3.36))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    fig.subplots_adjust(left=0.17, right=0.955, top=0.965, bottom=0.18)
    _draw(ax, labeled=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=330, transparent=True)
    plt.close(fig)

    # --- no-label version ---
    fig2, ax2 = plt.subplots(figsize=(5.04, 3.36))
    fig2.patch.set_alpha(0.0)
    ax2.set_facecolor("none")
    fig2.subplots_adjust(left=0.035, right=0.995, top=0.985, bottom=0.055)
    _draw(ax2, labeled=False)
    nolabel_base = out_base.parent / (out_base.name + "_nolabel")
    fig2.savefig(nolabel_base.with_suffix(".png"), dpi=330, transparent=True)
    plt.close(fig2)

    return max_abs_di_over_i


def plot_stick_figure_si(
    peaks: list[dict],
    B_values: dict[str, float],
    out_base: Path,
) -> None:
    """Stick pattern + (hkl) labels (SI companion)."""
    import matplotlib.pyplot as plt

    method_order = [
        "experimental (Fischer 1978)",
        "harmonic phonon spectrum (isotropic)",
        "empirical digitized reference",
        "elastic-derived Debye",
        "quasi-harmonic Debye",
    ]
    n = len(method_order)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 1.3 * n + 0.5), sharex=True)

    # Compute max per-method intensity across all peaks for per-panel scaling.
    peak_ints: dict[str, np.ndarray] = {m: apply_dwf(peaks, B_values[m]) for m in method_order}
    abs_max = max(np.max(v) for v in peak_ints.values())

    # Global top-20 strongest peaks for labeling, to keep clutter manageable.
    strongest = np.argsort(-peak_ints["harmonic phonon spectrum (isotropic)"])[:20]
    strongest_set = set(int(i) for i in strongest)

    for ax, m in zip(axes, method_order):
        ints = peak_ints[m]
        for i, p in enumerate(peaks):
            y = ints[i]
            ax.vlines(p["two_theta_deg"], 0, 100.0 * y / abs_max, color=METHOD_COLORS[m], linewidth=1.0)
            if i in strongest_set and m == "harmonic phonon spectrum (isotropic)":
                h, k, l = p["hkl_rep"]
                label = f"({h}{k}{l})".replace("-", r"$\bar{1}$")
                ax.annotate(
                    label,
                    xy=(p["two_theta_deg"], 100.0 * y / abs_max),
                    xytext=(0, 3), textcoords="offset points",
                    fontsize=6, ha="center", color="0.2", rotation=0,
                )
        ax.set_xlim(TWO_THETA_MIN, TWO_THETA_MAX)
        ax.set_ylim(0, 105)
        ax.set_ylabel("I (%)", fontsize=8)
        ax.grid(True, axis="x", alpha=0.15)
        ax.text(
            0.99, 0.85, METHOD_LABELS[m],
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            color=METHOD_COLORS[m],
        )

    axes[-1].set_xlabel(r"$2\theta$ (deg), Cu $K_{\alpha 1}$, $\lambda=1.5406$ $\mathrm{\AA}$", fontsize=10)
    fig.suptitle(
        r"Sb stick pattern at 294 K: per-reflection intensities under each DWF treatment",
        fontsize=10, fontweight="bold",
    )
    fig.subplots_adjust(hspace=0.12, top=0.93)
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight", transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    from matplotlib import font_manager

    arial_path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    if arial_path.exists():
        font_manager.fontManager.addfont(str(arial_path))
        arial = font_manager.FontProperties(fname=str(arial_path)).get_name()
    else:
        arial = "Arial"

    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": arial,
        "font.sans-serif": [arial],
        "mathtext.fontset": "custom",
        "mathtext.rm": arial,
        "mathtext.it": f"{arial}:italic",
        "mathtext.bf": f"{arial}:bold",
    })

    log("Reading Sb CONTCAR...")
    lattice, frac, species = read_contcar(CONTCAR_XZ)
    assert set(species) == {"Sb"}, f"Unexpected species: {set(species)}"
    log(f"  Lattice vol = {abs(np.linalg.det(lattice)):.4f} A^3; N_atoms = {len(species)}")

    log("Reading B values (4 models interpolated to T=294 K + Fischer experimental @ 294 K)...")
    model_B, B_exp, T_exp = read_b_values()
    B_values = dict(model_B)
    B_values["no DWF (static)"] = 0.0
    B_values["experimental (Fischer 1978)"] = B_exp
    for k, v in B_values.items():
        log(f"  B[{k}] = {v:.4f} A^2")
    log(f"  Fischer T_exp = {T_exp:.1f} K; model T = 294 K (linear interp on bt_curves_full)")

    log("Enumerating hkl in [10, 100] deg 2theta...")
    peaks = compute_peak_intensities(lattice, frac)
    log(f"  {len(peaks)} unique-d reflection groups")

    log("Convolving to pseudo-Voigt pattern grid...")
    two_theta = np.arange(TWO_THETA_MIN, TWO_THETA_MAX + TWO_THETA_STEP, TWO_THETA_STEP)
    method_order = [
        "no DWF (static)",
        "empirical digitized reference",
        "elastic-derived Debye",
        "quasi-harmonic Debye",
        "harmonic phonon spectrum (isotropic)",
        "experimental (Fischer 1978)",
    ]
    patterns: dict[str, np.ndarray] = {}
    for m in method_order:
        peak_int = apply_dwf(peaks, B_values[m])
        patterns[m] = convolve_profile(peaks, peak_int, two_theta)

    log("Writing CSVs...")
    write_peak_csv(peaks, B_values, CSV_DIR / "Sb_pc4_peaks_table.csv")
    write_pattern_csv(two_theta, patterns, CSV_DIR / "Sb_pc4_pattern_curves.csv")

    log("Plotting main figure (pseudo-Voigt)...")
    main_base = FIG_DIR / "fig04_panel_a_pattern"
    max_abs_di = plot_main_figure(two_theta, patterns, B_values, peaks, main_base)

    log("Plotting stick figure (SI)...")
    stick_base = FIG_DIR / "fig04_panel_a_pattern_stick_SI"
    plot_stick_figure_si(peaks, B_values, stick_base)

    log("Summary of max |dI/I| in window %s:" % (HIGH_ANGLE_WINDOW,))
    for m, v in max_abs_di.items():
        log(f"  {m}: {v:.2f}%")
    log("Done.")


if __name__ == "__main__":
    main()
