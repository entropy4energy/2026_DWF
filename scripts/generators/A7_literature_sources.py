#!/usr/bin/env python3
"""Build data/literature/A7_literature_B_sources.csv: the literature provenance
for every As / Sb / Bi Debye-Waller number used by the manuscript.

Why this file exists
--------------------
The A7 experimental anchors and the "empirical" reference curve enter the
pipeline through scripts/vendor/dwf_plot_config.py as bare numeric literals with
no citation attached. This script attaches the citation, records which figure or
table each number came from, flags the points that are physically impossible or
inconsistent with their own source curve, and writes an independent cross-check
against the one table in the source paper that tabulates a Debye-Waller factor.

It mirrors cF8_literature_sources.py (C / Si / Ge) column for column, so the two
archives can be concatenated.

Source
------
    P Fischer, I Sosnowska and M Szymanski,
    "Debye-Waller factor and thermal expansion of arsenic, antimony and
    bismuth", J. Phys. C: Solid State Phys. 11 (1978) 1043.
    Received 8 August 1977, in final form 21 November 1977.
    Institut fuer Reaktortechnik ETHZ, EIR, Wuerenlingen, Switzerland;
    Institute of Experimental Physics, University of Warsaw, Poland.

Fischer et al. do NOT tabulate B(T). Their text reads: "The experimentally
determined temperature factors B of As, Sb, and Bi are shown in figure 3
together with theoretical predictions based on various models of lattice
dynamics." Table 1 gives the hexagonal positional parameter z, Table 3 gives
Debye temperatures from elastic constants, and Table 2 gives Bi temperature
factors at 293 K and 516 K only. All B(T) values used here were therefore read
off figure 3 by hand, and Table 2 is the only independent check available.

What figure 3 contains (from its caption)
-----------------------------------------
    filled circle  results from neutron diffraction        <- the anchors used here
    open circle    results from x-ray diffraction
                     As: Schiferl and Barrett 1969
                     Sb: Cucka and Barrett 1962
                     Bi: Barrett et al 1963
    triangle/square  Mossbauer measurements, Sb only
                     Avenarius and Kuzmin 1975; Avenarius et al 1970
    curve 1        calculation based on the Debye model    <- the "empirical" curve
    curve 2        Born-von Karman model
    curve 3        from experimentally determined g(omega)

Because four different point styles and three different curves share one panel,
a hand reading can pick up the wrong series. The flags below record where that
appears to have happened.

Usage
-----
    python3 scripts/generators/A7_literature_sources.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

BASE_DIR = Path(__file__).resolve().parent
# Every path resolves through scripts/paths.py, the single source of truth
# for this package. Nothing outside the repository root is read or written.
sys.path.insert(0, str(BASE_DIR.parent))
from paths import LITERATURE, VENDOR  # noqa: E402

OUT = LITERATURE / "A7_literature_B_sources.csv"

H_SI = 6.62607015e-34
KB_SI = 1.380649e-23
AMU_TO_KG = 1.66053906660e-27

SOURCE_KEY = "Fischer1978"
SOURCE_LABEL = "Fischer, Sosnowska and Szymanski 1978 (J Phys C 11 1043)"

# Fischer et al. 1978, Table 2 -- isotropic and anisotropic temperature factors
# of Bi. Three refinements at each temperature:
#   (a) profile analysis
#   (b) integrated intensities, isotropic refinement
#   (c) integrated intensities, anisotropic refinement
# Values read directly from the printed table; the parenthesised digit is the
# standard deviation on the last quoted decimal.
BI_TABLE2 = [
    # T_K, refinement, B_isotr, sigma, R_percent
    (293.0, "a", 1.02, 0.06, 3.1),
    (293.0, "b", 1.00, 0.20, 3.2),
    (293.0, "c", 0.97, 0.09, 2.4),
    (516.0, "a", 1.34, 0.09, 6.9),
    (516.0, "b", 1.20, 0.30, 7.1),
    (516.0, "c", 1.10, 0.20, 4.2),
]

# Points that are inconsistent with their own source series. See the module
# docstring. The Sb clamp is sanitize_curve_points() in
# rebuild_dwf_revision_package.py, which maps any T in (-10, 0) to exactly 0 K.
#   artifact -> physically impossible as read (B must rise with T)
#   suspect  -> inconsistent with the single-Debye curve the rest of the series
#               defines; most likely read off curve 2/3 or off the experimental
#               points, which all lie above curve 1 in that temperature range
FLAGGED = {
    ("Sb", "empirical", -1.4285714): (
        "artifact",
        "negative temperature as read; clamped to T = 0 K in the analysis, so "
        "Sb_comparison_main_curves_wide.csv carries B(0 K) = 0.27142857 A^2 "
        "above B(10 K) = 0.17678498 A^2",
    ),
    ("As", "empirical", 2.9398968): (
        "artifact",
        "B = 0.27746586 A^2 exceeds the 36.179334 K value 0.18471171 A^2; "
        "positive T so not clamped, retained as read",
    ),
    ("As", "empirical", 120.098039): (
        "suspect",
        "28.1% above the single-Debye curve that the other 10 points of this "
        "series define to within 2.8% (Theta_D = 230.9 K fitted on T > 200 K); "
        "curve 2, curve 3 and the neutron points all lie above curve 1 here",
    ),
    ("As", "empirical", 159.183987): (
        "suspect",
        "17.3% above the same fitted single-Debye curve; see the 120.098039 K row",
    ),
}


def debye_b_factor(temperature: float, theta_d: float, mass_amu: float) -> float:
    """Isotropic Debye B(T) in A^2, matching rebuild_dwf_revision_package.py."""
    prefactor = 6.0 * H_SI**2 / (mass_amu * AMU_TO_KG * KB_SI * theta_d)
    if temperature <= 0.0:
        return float(prefactor * 0.25 * 1.0e20)
    x = theta_d / temperature
    integral = quad(lambda t: t / (np.exp(t) - 1.0), 0.0, x, limit=200)[0]
    return float(prefactor * (0.25 + (temperature / theta_d) ** 2 * integral) * 1.0e20)


def fit_theta_d(temps, values, mass_amu, t_min):
    """Debye temperature reproducing the mean of a digitized curve above t_min."""
    temps = np.asarray(temps, float)
    values = np.asarray(values, float)
    keep = temps > t_min
    t, b = temps[keep], values[keep]

    def residual(theta):
        return float(np.mean([debye_b_factor(x, theta, mass_amu) - y
                              for x, y in zip(t, b)]))

    theta = brentq(residual, 20.0, 2000.0)
    pred = np.array([debye_b_factor(x, theta, mass_amu) for x in t])
    rel = (pred - b) / b
    return theta, float(np.max(np.abs(rel))), float(np.sqrt(np.mean(rel**2)))


HEADER = """\
# Literature provenance for the A7 (As, Sb, Bi) isotropic Debye-Waller factors
# used in this repository.
# Column layout is identical to cF8_literature_B_sources.csv so the two archives
# can be concatenated.
#
# Single source for every B(T) value below:
#   P Fischer, I Sosnowska and M Szymanski, "Debye-Waller factor and thermal
#   expansion of arsenic, antimony and bismuth", J. Phys. C: Solid State Phys.
#   11 (1978) 1043.
#
# The paper does not tabulate B(T); it presents it in figure 3. Every value with
# source_type = digitized_figure_manual_* was therefore READ OFF FIGURE 3 BY
# HAND by the authors of the present work. No digitizer software was used and no
# reading uncertainty was recorded at the time, so B_unc_A2 is empty for those
# rows. Table 2 of the same paper tabulates Bi at 293 K and 516 K and is the
# only independent check available; those rows carry the published sigma.
#
# source_type:
#   digitized_figure_manual_experiment
#       hand-read from the filled circles of figure 3 (neutron diffraction,
#       Fischer et al.'s own measurement). Used as the experimental anchors.
#   digitized_figure_manual_model_curve
#       hand-read from curve 1 of figure 3, the Debye-model calculation. This is
#       the series the manuscript benchmarks as "empirical digitized reference".
#       It is a MODEL curve from the source paper, not a measurement.
#   diffraction_refinement
#       tabulated in Table 2 of the same paper (Bi only).
#
# flag:
#   ""         clean point
#   artifact   physically impossible as read (B must rise with T); retained
#              verbatim for archival fidelity
#   suspect    inconsistent with the single-Debye curve the rest of its own
#              series defines; most likely read off the wrong curve or the wrong
#              point style in a panel that carries three curves and four point
#              styles
# Flagged rows are archived, not deleted; none of the published numbers were
# changed.
#
"""


def main() -> None:
    namespace = {}
    exec(compile((VENDOR / "dwf_plot_config.py").read_text(), "dwf_plot_config.py", "exec"),
         namespace)
    elements = namespace["ELEMENTS"]

    rows = []
    fit_report = {}

    for material in ("As", "Sb", "Bi"):
        config = elements[material]
        mass = float(config["mass_amu"])

        # --- experimental anchors: filled circles of figure 3 ---
        for temperature, b_value in zip(config["exp_temps"], config["exp_b"]):
            flag, note = FLAGGED.get((material, "experiment", temperature), ("", ""))
            rows.append({
                "material": material,
                "T_K": f"{temperature:.6g}",
                "B_A2": f"{b_value:.6g}",
                "B_unc_A2": "",
                "source_key": SOURCE_KEY,
                "source_label": SOURCE_LABEL,
                "source_type": "digitized_figure_manual_experiment",
                "page_table": "figure 3 (filled circles, neutron diffraction)",
                "quantity": "B_iso as plotted",
                "conversion": "none (direct B)",
                "flag": flag,
                "notes": note or "hand-read from figure 3; no reading uncertainty recorded",
            })

        # --- the "empirical" reference curve: curve 1 of figure 3 ---
        theta, max_dev, rms_dev = fit_theta_d(
            config["empirical_temps"], config["empirical_b"], mass, t_min=200.0)
        fit_report[material] = (theta, max_dev, rms_dev)
        curve_note = (f"hand-read from figure 3 curve 1 (Debye model); a single "
                      f"Debye temperature Theta_D = {theta:.1f} K reproduces the "
                      f"T > 200 K part of this series to "
                      f"{max_dev * 100:.1f}% max / {rms_dev * 100:.1f}% RMS, "
                      f"which is what identifies it as curve 1")
        for temperature, b_value in zip(config["empirical_temps"], config["empirical_b"]):
            flag, note = FLAGGED.get((material, "empirical", temperature), ("", ""))
            rows.append({
                "material": material,
                "T_K": f"{temperature:.6g}",
                "B_A2": f"{b_value:.6g}",
                "B_unc_A2": "",
                "source_key": SOURCE_KEY,
                "source_label": SOURCE_LABEL,
                "source_type": "digitized_figure_manual_model_curve",
                "page_table": "figure 3 (curve 1, Debye model)",
                "quantity": "B_iso as plotted",
                "conversion": "none (direct B)",
                "flag": flag,
                "notes": note or curve_note,
            })

    # --- Bi Table 2: the one independent cross-check ---
    bi_exp = dict(zip(elements["Bi"]["exp_temps"], elements["Bi"]["exp_b"]))
    for temperature, refinement, b_value, sigma, r_percent in BI_TABLE2:
        nearest_t = min(bi_exp, key=lambda t: abs(t - temperature))
        nearest_b = bi_exp[nearest_t]
        delta = nearest_b - b_value
        within = "within" if abs(delta) <= sigma else "OUTSIDE"
        rows.append({
            "material": "Bi",
            "T_K": f"{temperature:.6g}",
            "B_A2": f"{b_value:.6g}",
            "B_unc_A2": f"{sigma:.6g}",
            "source_key": SOURCE_KEY,
            "source_label": SOURCE_LABEL,
            "source_type": "diffraction_refinement",
            "page_table": f"Table 2 ({refinement})",
            "quantity": f"B_isotr, R = {r_percent:.1f}%",
            "conversion": "none (direct B)",
            "flag": "",
            "notes": (f"refinement ({refinement}); cross-check on the hand reading: "
                      f"the digitized anchor at {nearest_t:.6g} K is "
                      f"{nearest_b:.6g} A^2, i.e. {delta:+.4f} A^2 versus this "
                      f"tabulated value -- {within} the published sigma of "
                      f"{sigma:.2f} A^2"),
        })

    fieldnames = ["material", "T_K", "B_A2", "B_unc_A2", "source_key",
                  "source_label", "source_type", "page_table", "quantity",
                  "conversion", "flag", "notes"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        handle.write(HEADER)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # ---------------- console report ----------------
    print("=== curve identification: is the 'empirical' series a single Debye curve? ===")
    print(f"  {'':4s} {'fitted Theta_D (K)':>19s} {'max dev':>9s} {'RMS dev':>9s}   (fit on T > 200 K)")
    for material, (theta, max_dev, rms_dev) in fit_report.items():
        print(f"  {material:4s} {theta:19.1f} {max_dev * 100:8.2f}% {rms_dev * 100:8.2f}%")
    print("  A single Debye temperature reproducing the series to a few percent is\n"
          "  consistent with curve 1 of figure 3 (the Debye-model curve).")

    print("\n=== cross-check against Fischer Table 2 (the only tabulated B) ===")
    for temperature, refinement, b_value, sigma, _r in BI_TABLE2:
        nearest_t = min(bi_exp, key=lambda t: abs(t - temperature))
        nearest_b = bi_exp[nearest_t]
        delta = nearest_b - b_value
        verdict = "within sigma" if abs(delta) <= sigma else "OUTSIDE sigma"
        print(f"  Bi {temperature:5.0f} K ({refinement}): table {b_value:.2f}({sigma:.2f}) "
              f"vs hand-read {nearest_b:.5f} at {nearest_t:.4f} K "
              f"-> {delta:+.4f} A^2, {verdict}")

    print("\n=== flagged rows ===")
    flagged = [r for r in rows if r["flag"]]
    for row in flagged:
        print(f"  {row['material']:6s} T = {row['T_K']:>12s} K  B = {row['B_A2']:>10s}  "
              f"[{row['flag']}]")
    print(f"  {len(flagged)} flagged of {len(rows)} rows.")

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
