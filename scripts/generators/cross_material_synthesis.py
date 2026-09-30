#!/usr/bin/env python3
"""Cross-material synthesis for the Debye-Waller benchmark.

Pulls together the seven calculated entries:
  - A7 series (Fischer 1978 neutron):  As, Sb, Bi (noSOC), Bi (SOC)
  - cF8 controls (single-T diffraction): C, Si, Ge

Goal: compare the A7-calibrated regime guide with non-A7 controls and assess
whether the Bi SOC harmonic phonon calculation follows the same harmonic
descriptor trends. Generalization beyond the tested systems requires validation.

Output:
  data/processed/cross_material_descriptors_table.csv
    -- one row per material with V_atom, G_VRH, mean_omega, sub-2THz,
       sub-1THz, leverage, leverage_below_2THz, plus the regime label
       suggested by the descriptor pair.

  data/processed/cross_material_room_temperature_residuals.csv
    -- B_iso(T_room) for every material x every model, plus the
       experimental anchor and the fractional deviation (model/exp - 1).

  data/processed/cross_material_synthesis_summary.md
    -- compact prose interpretation of the table, written as a memo to
       go alongside any future revision -- NOT to be inserted into the
       manuscript without the author's review.

The script reads only the CSV products written by the per-material
analysis scripts; it does not re-parse VASP/AFLOW outputs.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# Every path resolves through scripts/paths.py, the single source of truth for
# this package. Nothing outside the repository root is read or written.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import LITERATURE, PROCESSED, VERIFICATION_LOGS  # noqa: E402
CSV_DIR = PROCESSED
OUT_TABLE = CSV_DIR / "cross_material_descriptors_table.csv"
OUT_RESIDUALS = CSV_DIR / "cross_material_room_temperature_residuals.csv"
OUT_MEMO = CSV_DIR / "cross_material_synthesis_summary.md"


def load_a7_descriptors() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with (CSV_DIR / "descriptor_table.csv").open() as fh:
        for row in csv.DictReader(fh):
            mat = row["material"]
            out[mat] = row
    return out


def load_kv_descriptor(name: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if name == "Bi_SOC_phonon":
        path = CSV_DIR / "Bi_SOC_phonon_descriptors.csv"
    else:
        path = CSV_DIR / f"{name}_phonon_descriptors.csv"
    with path.open() as fh:
        for row in csv.reader(fh):
            if len(row) < 2 or row[0] == "descriptor":
                continue
            try:
                # the SOC descriptor file has [name, noSOC, SOC, delta_pct]
                if name == "Bi_SOC_phonon" and len(row) >= 3:
                    out[row[0]] = float(row[2])
                else:
                    out[row[0]] = float(row[1])
            except ValueError:
                continue
    return out


def load_curves(name: str) -> dict[str, np.ndarray]:
    """Load harmonic + scalar Debye B(T) curves written by the per-material script."""
    if name == "Ge":
        path = CSV_DIR / "Ge_phonon_B_curves.csv"
    elif name == "Si":
        path = CSV_DIR / "Si_phonon_B_curves.csv"
    elif name == "C":
        path = CSV_DIR / "C_phonon_B_curves.csv"
    else:
        raise KeyError(name)
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    return {
        "T": arr["T_K"],
        "B_harm": arr["B_iso_harmonic_A2"],
        "B_emp": arr[f"B_emp_{int(load_kv_descriptor(name)['theta_D_empirical_lit_K'])}K_A2"],
        "B_AEL": arr["B_AEL_A2"],
        "B_AGL": arr["B_AGL_A2"],
    }


REFERENCE_SOURCE_TYPES = {
    "diffraction_refinement",
    "cbed_direct",
    "pdos_derived",
    "thetaM_derived",
}


def load_cf8_literature_median(material: str, lo: float = 290.0,
                               hi: float = 300.0) -> float:
    """Return the source-balanced literature median used by SI Fig. S1."""
    source_medians = load_cf8_literature_source_medians(material, lo, hi)
    return float(np.median(list(source_medians.values())))


def load_cf8_literature_source_medians(material: str, lo: float = 290.0,
                                       hi: float = 300.0) -> dict[str, float]:
    """Per-source medians in [lo, hi] K, the inputs to the median above."""
    by_source: dict[str, list[float]] = {}
    path = LITERATURE / "cF8_literature_B_sources.csv"
    with path.open(newline="") as fh:
        reader = csv.DictReader(line for line in fh if not line.startswith("#"))
        for row in reader:
            if row["material"] != material:
                continue
            if row.get("source_type", "") not in REFERENCE_SOURCE_TYPES:
                continue
            if row.get("flag", "").strip() == "artifact":
                continue
            try:
                temperature = float(row["T_K"])
                value = float(row["B_A2"])
            except ValueError:
                continue
            if lo <= temperature <= hi:
                by_source.setdefault(row["source_key"], []).append(value)
    if not by_source:
        raise ValueError(f"No cF8 literature reference for {material} in {lo:g}-{hi:g} K")
    return {key: float(np.median(values)) for key, values in by_source.items()}


def load_a7_room_temperature_BT() -> dict[str, dict[str, float]]:
    """Pull the A7 model curves and Fischer experimental points so we can
    quote a single representative T (294 K for Sb to match the Fig. 4
    diagnostic; 293-300 K for As/Bi/Bi-SOC). The wide CSVs already encode
    the model output."""
    out: dict[str, dict[str, float]] = {}
    EMP = "B_A2__empirical_digitized_reference"
    AEL = "B_A2__elasticderived_Debye"
    AGL = "B_A2__quasiharmonic_Debye"
    HARM = "B_A2__harmonic_phonon_spectrum_isotropic"
    a7_specs = {
        "As":  ("As_comparison_main_curves_wide.csv", EMP, AEL, AGL, HARM,
                "As_comparison_experiment_points.csv", "B_exp_A2", 293.0),
        "Sb":  ("Sb_comparison_main_curves_wide.csv", EMP, AEL, AGL, HARM,
                "Sb_comparison_experiment_points.csv", "B_exp_A2", 294.0),
        "Bi":  ("Bi_comparison_main_curves_wide.csv", EMP, AEL, AGL, HARM,
                "Bi_comparison_experiment_points.csv", "B_exp_A2", 291.9),
    }
    for mat, (curves_csv, c_emp, c_ael, c_agl, c_harm,
              exp_csv, exp_col, T_target) in a7_specs.items():
        arr = np.genfromtxt(CSV_DIR / curves_csv, delimiter=",", names=True,
                            dtype=float, encoding="utf-8")
        T = arr["T_K"]
        Be = float(np.interp(T_target, T, arr[c_emp]))
        Ba = float(np.interp(T_target, T, arr[c_ael]))
        Bq = float(np.interp(T_target, T, arr[c_agl]))
        Bh = float(np.interp(T_target, T, arr[c_harm]))
        # exp anchor at T closest to T_target
        exp_arr = np.genfromtxt(CSV_DIR / exp_csv, delimiter=",", names=True,
                                dtype=float, encoding="utf-8")
        i_close = int(np.argmin(np.abs(exp_arr["T_K"] - T_target)))
        T_exp = float(exp_arr["T_K"][i_close])
        B_exp = float(exp_arr[exp_col][i_close])
        out[mat] = {
            "T_target": T_target, "T_exp": T_exp, "B_exp": B_exp,
            "B_emp": Be, "B_AEL": Ba, "B_AGL": Bq, "B_harm": Bh,
        }

    # Bi SOC harmonic phonon (newly computed) + Bi SOC scalar Debye
    soc_curves = np.genfromtxt(
        CSV_DIR / "Bi_SOC_phonon_harmonic_curves.csv",
        delimiter=",", names=True, dtype=float, encoding="utf-8",
    )
    soc_scalar = np.genfromtxt(
        CSV_DIR / "Bi_SOC_relativistic_sensitivity_main_curves_wide.csv",
        delimiter=",", names=True, dtype=float, encoding="utf-8",
    )
    T_target = 291.9
    bi_exp = np.genfromtxt(
        CSV_DIR / "Bi_comparison_experiment_points.csv",
        delimiter=",", names=True, dtype=float, encoding="utf-8",
    )
    i_close = int(np.argmin(np.abs(bi_exp["T_K"] - T_target)))
    out["Bi_SOC"] = {
        "T_target": T_target,
        "T_exp": float(bi_exp["T_K"][i_close]),
        "B_exp": float(bi_exp["B_exp_A2"][i_close]),
        "B_emp": float("nan"),  # SOC empirical Debye reference not defined
        "B_AEL": float(np.interp(T_target, soc_scalar["T_K"],
                                 soc_scalar["Bi_SOC_B_elastic_derived_debye_A2"])),
        "B_AGL": float(np.interp(T_target, soc_scalar["T_K"],
                                 soc_scalar["Bi_SOC_B_quasi_harmonic_debye_A2"])),
        "B_harm": float(np.interp(T_target, soc_curves["T_K"],
                                  soc_curves["B_iso_SOC_A2"])),
    }
    return out


def load_cF8_room_temperature_BT() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    specs = {
        "C":  ("C",  295.0),
        "Si": ("Si", 295.0),
        "Ge": ("Ge", 295.0),
    }
    for label, (name, T_target) in specs.items():
        B_exp_anchor = load_cf8_literature_median(name)
        c = load_curves(name)
        Bh = float(np.interp(T_target, c["T"], c["B_harm"]))
        Be = float(np.interp(T_target, c["T"], c["B_emp"]))
        Ba = float(np.interp(T_target, c["T"], c["B_AEL"]))
        Bq = float(np.interp(T_target, c["T"], c["B_AGL"]))
        out[label] = {
            "T_target": T_target, "T_exp": T_target, "B_exp": B_exp_anchor,
            "B_emp": Be, "B_AEL": Ba, "B_AGL": Bq, "B_harm": Bh,
        }
    return out


def regime_label(sub2: float, mean_w: float) -> str:
    """A7 regime label from Discussion / Table 'regime_recipe'."""
    if sub2 <= 0.15 and mean_w >= 4.0:
        return "stiff (As-like)"
    if 0.25 < sub2 < 0.45 and 2.5 <= mean_w <= 3.5:
        return "soft (Sb-like)"
    if sub2 >= 0.45 and mean_w <= 2.5:
        return "boundary (Bi-like)"
    return "intermediate / off-A7"


def main() -> None:
    a7 = load_a7_descriptors()
    si = load_kv_descriptor("Si")
    c = load_kv_descriptor("C")
    ge = load_kv_descriptor("Ge")
    bi_soc_phonon = load_kv_descriptor("Bi_SOC_phonon")  # SOC harmonic descriptors

    # ---------- descriptor table ----------
    cols = [
        "material", "structure", "soc_status", "V_per_atom_A3",
        "G_VRH_GPa", "Theta_D_AEL_K", "Theta_D_AGL_K",
        "mean_freq_THz", "log_avg_freq_THz",
        "sub_1THz_DOS_frac", "sub_2THz_DOS_frac",
        "leverage_total", "leverage_below_2THz",
        "leverage_fraction_below_2THz", "regime_A7_label",
    ]
    rows: list[dict[str, str]] = []

    def emit_a7(mat_label: str, source_key: str) -> dict[str, str]:
        r = a7[source_key]
        try:
            sub2 = float(r["dos_frac_0_2THz"])
            mean_w = float(r["mean_freq_THz"])
            label = regime_label(sub2, mean_w)
        except ValueError:
            label = "n/a (overridden below)"
        return {
            "material": mat_label,
            "structure": "A7 (hR2, R-3m)",
            "soc_status": r["soc_status"],
            "V_per_atom_A3": r["V_per_atom_A3"],
            "G_VRH_GPa": r["G_VRH_GPa"],
            "Theta_D_AEL_K": r["debye_T_elastic_K"],
            "Theta_D_AGL_K": r["debye_T_quasi_harmonic_K"],
            "mean_freq_THz": r["mean_freq_THz"],
            "log_avg_freq_THz": r["log_avg_freq_THz"],
            "sub_1THz_DOS_frac": r["dos_frac_0_1THz"],
            "sub_2THz_DOS_frac": r["dos_frac_0_2THz"],
            "leverage_total": "",          # not in legacy CSV
            "leverage_below_2THz": "",
            "leverage_fraction_below_2THz": "",
            "regime_A7_label": label,
        }

    rows.append(emit_a7("As",     "As"))
    rows.append(emit_a7("Sb",     "Sb"))
    rows.append(emit_a7("Bi",     "Bi_noSOC"))
    bi_soc_a7_row = emit_a7("Bi (SOC)", "Bi_SOC")
    # Override with the SOC phonon descriptors (the legacy row stores NA)
    bi_soc_a7_row.update({
        "mean_freq_THz": f"{bi_soc_phonon['mean_freq_THz']:.5f}",
        "log_avg_freq_THz": f"{bi_soc_phonon['log_avg_freq_THz']:.5f}",
        "sub_1THz_DOS_frac": f"{bi_soc_phonon['dos_frac_below_1THz']:.5f}",
        "sub_2THz_DOS_frac": f"{bi_soc_phonon['dos_frac_below_2THz']:.5f}",
        "leverage_total": f"{bi_soc_phonon['leverage_total']:.5f}",
        "leverage_below_2THz": f"{bi_soc_phonon['leverage_below_2THz']:.5f}",
        "leverage_fraction_below_2THz": f"{bi_soc_phonon['leverage_fraction_below_2THz']:.5f}",
        "regime_A7_label": regime_label(bi_soc_phonon["dos_frac_below_2THz"],
                                        bi_soc_phonon["mean_freq_THz"]),
    })
    rows.append(bi_soc_a7_row)

    def emit_cf8(label: str, dd: dict[str, float]) -> dict[str, str]:
        sub2 = dd["dos_frac_below_2THz"]
        mean_w = dd["mean_freq_THz"]
        return {
            "material": label,
            "structure": "cF8 (diamond)",
            "soc_status": "noSOC",
            "V_per_atom_A3": "",
            "G_VRH_GPa": f"{dd.get('ael_shear_modulus_vrh', float('nan')):.4g}",
            "Theta_D_AEL_K": f"{dd.get('ael_debye_temperature', float('nan')):.4g}",
            "Theta_D_AGL_K": f"{dd.get('agl_debye', float('nan')):.4g}",
            "mean_freq_THz": f"{mean_w:.5f}",
            "log_avg_freq_THz": f"{dd['log_avg_freq_THz']:.5f}",
            "sub_1THz_DOS_frac": f"{dd['dos_frac_below_1THz']:.5f}",
            "sub_2THz_DOS_frac": f"{sub2:.5f}",
            "leverage_total": f"{dd['leverage_total']:.5f}",
            "leverage_below_2THz": f"{dd['leverage_below_2THz']:.5f}",
            "leverage_fraction_below_2THz": f"{dd['leverage_fraction_below_2THz']:.5f}",
            "regime_A7_label": regime_label(sub2, mean_w),
        }

    rows.append(emit_cf8("C",  c))
    rows.append(emit_cf8("Si", si))
    rows.append(emit_cf8("Ge", ge))

    with OUT_TABLE.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {OUT_TABLE}")

    # ---------- room-temperature residual table ----------
    a7_BT = load_a7_room_temperature_BT()
    cf8_BT = load_cF8_room_temperature_BT()
    BT = {**a7_BT, **cf8_BT}

    res_cols = [
        "material", "structure", "T_K", "T_exp_K", "B_exp_A2",
        "B_emp_A2", "B_AEL_A2", "B_AGL_A2", "B_harm_A2",
        "frac_dev_emp_pct", "frac_dev_AEL_pct",
        "frac_dev_AGL_pct", "frac_dev_harm_pct",
        "best_model",
    ]
    res_rows: list[dict[str, str]] = []
    for mat, d in BT.items():
        models = {"emp": d["B_emp"], "AEL": d["B_AEL"],
                  "AGL": d["B_AGL"], "harm": d["B_harm"]}
        # ignore NaNs when picking the best
        finite = {k: v for k, v in models.items()
                  if isinstance(v, float) and v == v}
        best = min(finite, key=lambda k: abs(finite[k] / d["B_exp"] - 1))
        struct = ("A7 (hR2)" if mat in ("As", "Sb", "Bi", "Bi_SOC")
                  else "cF8 (diamond)")
        res_rows.append({
            "material": mat,
            "structure": struct,
            "T_K": f"{d['T_target']:.1f}",
            "T_exp_K": f"{d['T_exp']:.1f}",
            "B_exp_A2": f"{d['B_exp']:.4f}",
            "B_emp_A2": "" if d["B_emp"] != d["B_emp"] else f"{d['B_emp']:.4f}",
            "B_AEL_A2": f"{d['B_AEL']:.4f}",
            "B_AGL_A2": f"{d['B_AGL']:.4f}",
            "B_harm_A2": f"{d['B_harm']:.4f}",
            "frac_dev_emp_pct": "" if d["B_emp"] != d["B_emp"]
                else f"{(d['B_emp']/d['B_exp']-1)*100:+.1f}",
            "frac_dev_AEL_pct": f"{(d['B_AEL']/d['B_exp']-1)*100:+.1f}",
            "frac_dev_AGL_pct": f"{(d['B_AGL']/d['B_exp']-1)*100:+.1f}",
            "frac_dev_harm_pct": f"{(d['B_harm']/d['B_exp']-1)*100:+.1f}",
            "best_model": best,
        })

    with OUT_RESIDUALS.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=res_cols)
        w.writeheader()
        for r in res_rows:
            w.writerow(r)
    print(f"Wrote {OUT_RESIDUALS}")
    res_by_material = {row["material"]: row for row in res_rows}

    # ---------- prose memo ----------
    print(f"\n{'material':10s} {'sub-2 THz':>10s} {'mean omega':>10s} "
          f"{'regime':>22s} {'best model @ Troom':>22s} "
          f"{'harm dev %':>10s} {'AGL dev %':>10s}")
    for r in rows:
        mat = r["material"]
        rr = next((x for x in res_rows if x["material"] == mat
                   or (mat == "Bi (SOC)" and x["material"] == "Bi_SOC")), None)
        if not rr:
            continue
        sub2 = float(r["sub_2THz_DOS_frac"]) if r["sub_2THz_DOS_frac"] else float("nan")
        mw = float(r["mean_freq_THz"]) if r["mean_freq_THz"] else float("nan")
        print(f"{mat:10s} {sub2:10.4f} {mw:10.3f} "
              f"{r['regime_A7_label']:>22s} {rr['best_model']:>22s} "
              f"{rr['frac_dev_harm_pct']:>10s} {rr['frac_dev_AGL_pct']:>10s}")

    memo = []
    memo.append("# Cross-material synthesis (As, Sb, Bi, Bi-SOC, C, Si, Ge)\n")
    memo.append("Generated automatically by `scripts/generators/cross_material_synthesis.py`.\n")
    memo.append("## Table 1 — descriptor + regime label\n")
    memo.append("| material | structure | sub-2 THz | mean omega (THz) | "
                "regime (A7 rule) |\n|---|---|---:|---:|---|\n")
    for r in rows:
        memo.append(
            f"| {r['material']} | {r['structure']} | "
            f"{r['sub_2THz_DOS_frac']} | {r['mean_freq_THz']} | "
            f"{r['regime_A7_label']} |\n")

    memo.append("\n## Table 2 — room-temperature B(T_room) residuals\n")
    memo.append("| material | T (K) | B_exp (A^2) | B_harm dev | "
                "B_AGL dev | B_AEL dev | best model |\n|---|---:|---:|---:|---:|---:|---|\n")
    for rr in res_rows:
        memo.append(
            f"| {rr['material']} | {rr['T_K']} | {rr['B_exp_A2']} | "
            f"{rr['frac_dev_harm_pct']}% | {rr['frac_dev_AGL_pct']}% | "
            f"{rr['frac_dev_AEL_pct']}% | {rr['best_model']} |\n")

    memo.append("\n## Interpretation\n")
    memo.append(
        "1. **Bi SOC harmonic phonon now exists.** Compared to scalar-relativistic Bi, "
        "the SOC phonon DOS shifts mass to lower frequencies "
        f"(sub-1 THz fraction {bi_soc_phonon['dos_frac_below_1THz']:.3f} vs. "
        f"{a7['Bi_noSOC']['dos_frac_0_1THz']}, mean_w "
        f"{bi_soc_phonon['mean_freq_THz']:.2f} vs. {a7['Bi_noSOC']['mean_freq_THz']} THz, "
        "leverage proxy +26%). The harmonic descriptor trend anticipates a larger "
        "harmonic-model overshoot, as observed in the new "
        "B_iso(T) residuals: harmonic SOC RMSE = 1.12 A^2 (R^2 = -4.71) vs "
        "harmonic noSOC RMSE = 0.62 A^2 (R^2 = -0.75). The SOC point therefore does "
        "not contradict the harmonic descriptor trend and shifts Bi toward the lower-frequency "
        "end of the tested set. This supports using the SOC result as a "
        "sensitivity test, but it does not uniquely identify the origin or sign of the "
        "missing finite-temperature physics. Caveat: the SOC APL run "
        "carries non-trivial EDDRMM and LRF_COMMUTATOR warnings; the absolute "
        "SOC harmonic numbers should "
        "be quoted as a *qualitative sensitivity probe*, not a converged production "
        "benchmark.\n\n"
    )
    memo.append(
        "2. **C is the extreme-stiff limit.** Sub-2 THz DOS fraction "
        f"{c['dos_frac_below_2THz']:.2e}, mean_w {c['mean_freq_THz']:.1f} THz. Against the "
        f"source-balanced room-temperature literature median ({res_by_material['C']['B_exp_A2']} A^2), "
        f"the harmonic value differs by {res_by_material['C']['frac_dev_harm_pct']}%, while the "
        f"empirical, elastic-derived, and quasi-harmonic scalar values differ by "
        f"{res_by_material['C']['frac_dev_emp_pct']}%, {res_by_material['C']['frac_dev_AEL_pct']}%, "
        f"and {res_by_material['C']['frac_dev_AGL_pct']}%. C therefore tests the stiff, "
        "zero-point-dominated limit but does not by itself establish scalar-Debye accuracy.\n\n"
    )
    memo.append(
        "3. **Si and Ge expose a non-A7 caveat in the 'scalar-tolerant' label.** "
        "Both pass the A7 sub-2 THz threshold (Si 0.006, Ge 0.079 vs. As 0.10). "
        "In the source-balanced harmonic/scalar comparison at 295 K:\n"
        f"   - Si: harm {res_by_material['Si']['frac_dev_harm_pct']}% / "
        f"AGL {res_by_material['Si']['frac_dev_AGL_pct']}% / "
        f"AEL {res_by_material['Si']['frac_dev_AEL_pct']}% / "
        f"emp {res_by_material['Si']['frac_dev_emp_pct']}%\n"
        f"   - Ge: harm {res_by_material['Ge']['frac_dev_harm_pct']}% / "
        f"AGL {res_by_material['Ge']['frac_dev_AGL_pct']}% / "
        f"AEL {res_by_material['Ge']['frac_dev_AEL_pct']}% / "
        f"emp {res_by_material['Ge']['frac_dev_emp_pct']}%\n"
        "Scalar Debye undershoots by ~20-40% in both Si and Ge. **[mechanism corrected "
        "2026-05-29 — see data/20260529_debye_moment_verification/]** This is NOT caused "
        "by the bimodal DOS / optical peak: in the classical limit, the mean-square-displacement "
        "slope is governed by <w^-2> = INT g(w) w^-2 dw. Its 1/w^2 weight makes the "
        "high-frequency optical contribution secondary. A window decomposition of the <f^-2> excess (real "
        "DOS minus an ideal Debye DOS at the same Theta) localizes the discrepancy to the "
        "2-4 THz transverse-acoustic band (Si +0.0060, Ge +0.0315 THz^-2), not the optical "
        "region; for Ge the Debye parabola actually overcounts <f^-2> across the optical "
        "window (-0.0121). The correct mechanism is a frequency-moment mismatch: the scalar "
        "models use Theta_D from elastic constants (AEL), calorimetric/empirical fits (emp), "
        "or the quasi-harmonic volume response (AGL), none of which necessarily matches the "
        "classical displacement moment. The Debye-Waller (DWF) Debye temperature that reproduces <w^-2> is "
        "systematically lower (Theta_DWF/Theta_emp = 0.87 C, 0.78 Si, 0.69 Ge), matching the "
        "room-temperature underestimate. In the tested systems, *scalar Debye is "
        "most reliable when the elastic/calorimetric Theta_D also approximates the "
        "inverse-square (Debye-Waller) moment <w^-2>*, not simply 'when sub-2 THz fraction is "
        "small'. This is the same classical <w^-2> sensitivity examined in the A7 soft-mode analysis "
        "(A7 amplifies it through soft modes; cF8 exposes it through a moment-mismatched "
        "Theta_D), consistent with the manuscript's caveat that A7 thresholds are "
        "family-specific. Do NOT write 'a single Debye temperature cannot reproduce the "
        "bimodal/optical-peak shape' — a lattice-dynamics reviewer will note the optical peak "
        "is secondary for the classical inverse-square moment.\n\n"
    )
    memo.append(
        "4. **Experimental accuracy remains separate from the harmonic moment diagnostic.** "
        "Low-temperature deviations are present in the cF8 controls as well as Bi, and their "
        "magnitudes depend on the available literature reference. The moment ratio diagnoses "
        "scalar-versus-harmonic mismatch; it does not identify the remaining electronic-structure, "
        "finite-temperature, or experimental contributions. The cF8 controls provide a second "
        "tested system for the harmonic trend, not a general cross-family validation.\n\n"
    )
    memo.append(
        "5. **Best-model ranking is consistent with the A7-calibrated separation, with important limits.** "
        "At room T the model that minimizes |B_model/B_exp - 1|:\n"
        "   - As: empirical Debye (-15.6%) ≈ harm (-15.6%); both within the "
        "below-resolution noise floor identified in Methods.\n"
        "   - Sb: harm (-3.3%) << every scalar Debye (-37 to -56%).\n"
        "   - Bi noSOC: AGL (-8.7%) << harm (+51.6%).\n"
        "   - Bi SOC: AGL (+28.5%) better than harm (+102.5%); AGL itself is degraded "
        "by SOC, exactly the picture in Fig. 6 of the manuscript.\n"
        f"   - C, Si, Ge (cF8 controls): the source-balanced room-temperature ranking is "
        f"{res_by_material['C']['best_model']} for C, {res_by_material['Si']['best_model']} for Si, "
        f"and {res_by_material['Ge']['best_model']} for Ge. These controls show that A7 "
        "frequency-window thresholds cannot be transferred without recalibration. The moment "
        "ratio diagnoses scalar-versus-harmonic mismatch; it does not guarantee agreement of "
        "either model with experiment.\n\n"
    )
    memo.append(
        "## Manuscript implication\n"
        "The new data support an A7-calibrated treatment guide, while the cF8 controls "
        "show why other structural families require independent validation and recalibration. "
        "The Bi SOC harmonic point is a real "
        "data point and should replace the hedged language in Methods that says "
        "'No SOC harmonic phonon displacement curve is included in the present "
        "benchmark.' — the curve exists, it is reported with the convergence caveats "
        "documented in `Bi_SOC_phonon_README.md`. It shows that the harmonic mismatch "
        "grows with SOC, but does not by itself establish an anharmonic cause.\n"
    )

    OUT_MEMO.write_text("".join(memo))
    print(f"Wrote {OUT_MEMO}")


if __name__ == "__main__":
    main()
