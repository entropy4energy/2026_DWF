#!/usr/bin/env python3
"""Build wide CSV tables consumed by the per-figure driver scripts.

Reads the authoritative long-form CSV exports from data/processed/ (produced
by rebuild_dwf_revision_package.py) and writes wide-form CSVs back to
the same data/processed/ directory, where the per-figure driver scripts
(fig02_AsSbBi_comparison.py, fig06_Bi_SOC_sensitivity.py) pick them up.

Run from this directory: python3 export_figure_wide_csvs.py

[debug] prints the output directory and row counts.
"""

from __future__ import annotations

import csv
from collections import defaultdict
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# Every path resolves through scripts/paths.py, the single source of truth
# for this package.
sys.path.insert(0, str(BASE_DIR.parent))
from paths import PROCESSED  # noqa: E402

CSV_DIR = PROCESSED
OUT_DIR = CSV_DIR

from rebuild_dwf_revision_package import (
    METHOD_LABELS,
    METHOD_ORDER_MAIN,
)


def _parse_float(cell: str) -> float | None:
    if cell in ("", "NA"):
        return None
    return float(cell)


def load_curves_by_method(material: str) -> dict[str, dict[float, float | None]]:
    path = CSV_DIR / "bt_curves_full.csv"
    by_method: dict[str, dict[float, float | None]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if row["material"] != material:
                continue
            method = row["method"]
            t_k = float(row["T_K"])
            raw_b = row["B_A2"]
            by_method[method][t_k] = _parse_float(raw_b) if raw_b not in ("", "NA") else None
    return dict(by_method)


def main_curve_labels() -> list[str]:
    return [METHOD_LABELS[mid] for mid in METHOD_ORDER_MAIN]


def write_comparison_wide(material: str, stem: str) -> int:
    """As_comparison / Sb_comparison / Bi_comparison: main panel curves (wide by T_K)."""
    by_method = load_curves_by_method(material)
    labels = main_curve_labels()
    temps: set[float] = set()
    for label in labels:
        temps.update(by_method.get(label, {}).keys())
    sorted_temps = sorted(temps)
    col_names = ["T_K"] + [f"B_A2__{label.replace(' ', '_')}" for label in labels]
    rows_out = []
    for t_k in sorted_temps:
        row = [t_k]
        for label in labels:
            val = by_method.get(label, {}).get(t_k)
            row.append("" if val is None else val)
        rows_out.append(row)
    out_path = OUT_DIR / f"{stem}_main_curves_wide.csv"
    with out_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(col_names)
        writer.writerows(rows_out)
    print(f"[debug] wrote {out_path} rows={len(rows_out)} cols={len(col_names)}")
    return len(rows_out)


def write_experiment_scatter(material: str, stem: str) -> int:
    path = CSV_DIR / "bt_at_experiment_temperatures.csv"
    seen: set[float] = set()
    rows_out: list[list[object]] = []
    with path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if row["material"] != material:
                continue
            t_k = float(row["T_K"])
            if t_k in seen:
                continue
            seen.add(t_k)
            b_exp = _parse_float(row["B_exp_A2"])
            rows_out.append([t_k, "" if b_exp is None else b_exp])
    rows_out.sort(key=lambda r: r[0])
    out_path = OUT_DIR / f"{stem}_experiment_points.csv"
    with out_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["T_K", "B_exp_A2"])
        writer.writerows(rows_out)
    print(f"[debug] wrote {out_path} rows={len(rows_out)}")
    return len(rows_out)


def write_residuals_wide(material: str, stem: str) -> int:
    """Residual columns match comparison figure lower-left panel (main four methods)."""
    path = CSV_DIR / "bt_at_experiment_temperatures.csv"
    stub_map = [
        ("empirical_digitized_reference", "residual_empirical_digitized_reference"),
        ("elastic_derived_debye", "residual_elastic_derived_debye"),
        ("quasi_harmonic_debye", "residual_quasi_harmonic_debye"),
        ("harmonic_phonon_spectrum_isotropic", "residual_harmonic_phonon_spectrum_isotropic"),
    ]
    rows_out: list[list[object]] = []
    with path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if row["material"] != material:
                continue
            t_k = float(row["T_K"])
            b_exp = _parse_float(row["B_exp_A2"])
            out_row: list[object] = [t_k, "" if b_exp is None else b_exp]
            for _, res_col in stub_map:
                v = _parse_float(row[res_col])
                out_row.append("" if v is None else v)
            rows_out.append(out_row)
    rows_out.sort(key=lambda r: r[0])
    headers = ["T_K", "B_exp_A2"] + [f"residual__{s}" for s, _ in stub_map]
    out_path = OUT_DIR / f"{stem}_residuals_at_experiment_wide.csv"
    with out_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)
        writer.writerows(rows_out)
    print(f"[debug] wrote {out_path} rows={len(rows_out)}")
    return len(rows_out)


def write_sb_apl_wide() -> int:
    """Sb_apl_components: three harmonic directional curves on the same grid."""
    material = "Sb"
    by_method = load_curves_by_method(material)
    labels = [
        METHOD_LABELS["harmonic_phonon_spectrum_isotropic"],
        METHOD_LABELS["harmonic_phonon_spectrum_basal_plane_average"],
        METHOD_LABELS["harmonic_phonon_spectrum_c_axis"],
    ]
    temps: set[float] = set()
    for label in labels:
        temps.update(by_method.get(label, {}).keys())
    sorted_temps = sorted(temps)
    col_names = ["T_K"] + [f"B_A2__{label.replace(' ', '_')}" for label in labels]
    rows_out = []
    for t_k in sorted_temps:
        row = [t_k]
        for label in labels:
            val = by_method.get(label, {}).get(t_k)
            row.append("" if val is None else val)
        rows_out.append(row)
    out_path = OUT_DIR / "Sb_apl_components_main_curves_wide.csv"
    with out_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(col_names)
        writer.writerows(rows_out)
    print(f"[debug] wrote {out_path} rows={len(rows_out)}")
    write_experiment_scatter("Sb", "Sb_apl_components")
    return len(rows_out)


def write_bi_soc_wide() -> int:
    """Bi_SOC_relativistic_sensitivity: shared grid + empirical ref + noSOC/SOC elastic & quasi."""
    by_no = load_curves_by_method("Bi_noSOC")
    by_soc = load_curves_by_method("Bi_SOC")
    lab_emp = METHOD_LABELS["empirical_digitized_reference"]
    lab_el = METHOD_LABELS["elastic_derived_debye"]
    lab_qh = METHOD_LABELS["quasi_harmonic_debye"]
    temps = set(by_no.get(lab_emp, {})) | set(by_no.get(lab_el, {})) | set(by_no.get(lab_qh, {}))
    sorted_temps = sorted(temps)
    headers = [
        "T_K",
        "B_empirical_digitized_reference_A2",
        "Bi_noSOC_B_elastic_derived_debye_A2",
        "Bi_SOC_B_elastic_derived_debye_A2",
        "Bi_noSOC_B_quasi_harmonic_debye_A2",
        "Bi_SOC_B_quasi_harmonic_debye_A2",
    ]
    rows_out = []
    for t_k in sorted_temps:
        rows_out.append(
            [
                t_k,
                by_no.get(lab_emp, {}).get(t_k, ""),
                by_no.get(lab_el, {}).get(t_k, ""),
                by_soc.get(lab_el, {}).get(t_k, ""),
                by_no.get(lab_qh, {}).get(t_k, ""),
                by_soc.get(lab_qh, {}).get(t_k, ""),
            ]
        )
    out_path = OUT_DIR / "Bi_SOC_relativistic_sensitivity_main_curves_wide.csv"
    with out_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)
        writer.writerows(rows_out)
    print(f"[debug] wrote {out_path} rows={len(rows_out)}")

    # Residuals: merge Bi_noSOC and Bi_SOC rows on T_K (same experiment B)
    path = CSV_DIR / "bt_at_experiment_temperatures.csv"
    by_t: dict[float, dict[str, float | None]] = {}
    with path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if row["material"] not in ("Bi_noSOC", "Bi_SOC"):
                continue
            t_k = float(row["T_K"])
            entry = by_t.setdefault(t_k, {})
            entry["B_exp_A2"] = _parse_float(row["B_exp_A2"])
            mat = row["material"]
            for stub, res_col in (
                ("elastic", "residual_elastic_derived_debye"),
                ("quasi", "residual_quasi_harmonic_debye"),
            ):
                key = f"{mat}__{stub}"
                entry[key] = _parse_float(row[res_col])
    res_headers = [
        "T_K",
        "B_exp_A2",
        "residual_Bi_noSOC_elastic_derived_debye",
        "residual_Bi_SOC_elastic_derived_debye",
        "residual_Bi_noSOC_quasi_harmonic_debye",
        "residual_Bi_SOC_quasi_harmonic_debye",
    ]
    res_rows = []
    for t_k in sorted(by_t):
        e = by_t[t_k]
        res_rows.append(
            [
                t_k,
                e.get("B_exp_A2", ""),
                e.get("Bi_noSOC__elastic", ""),
                e.get("Bi_SOC__elastic", ""),
                e.get("Bi_noSOC__quasi", ""),
                e.get("Bi_SOC__quasi", ""),
            ]
        )
    res_path = OUT_DIR / "Bi_SOC_relativistic_sensitivity_residuals_at_experiment_wide.csv"
    with res_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(res_headers)
        writer.writerows(res_rows)
    print(f"[debug] wrote {res_path} rows={len(res_rows)}")
    return len(rows_out)


def write_summary_heatmaps() -> int:
    """Same numeric content as summary_heatmaps figure (As, Sb, Bi_noSOC × main methods)."""
    path = CSV_DIR / "benchmark_metrics.csv"
    # key: (material, method_label) -> row
    metrics: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            metrics[(row["material"], row["method"])] = row
    materials = ["As", "Sb", "Bi_noSOC"]
    labels = main_curve_labels()
    out_path = OUT_DIR / "summary_heatmaps_benchmark_matrix.csv"
    with out_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(
            [
                "method",
                "NRMSE_As",
                "NRMSE_Sb",
                "NRMSE_Bi_noSOC",
                "R2_As",
                "R2_Sb",
                "R2_Bi_noSOC",
            ]
        )
        count = 0
        for label in labels:
            nrmse_vals = []
            r2_vals = []
            for mat in materials:
                row = metrics.get((mat, label))
                if row is None:
                    nrmse_vals.append("")
                    r2_vals.append("")
                else:
                    nrmse_vals.append(row["NRMSE"])
                    r2_vals.append(row["R2"])
            writer.writerow([label, *nrmse_vals, *r2_vals])
            count += 1
    print(f"[debug] wrote {out_path} rows={count}")
    return count


def main() -> None:
    if not (CSV_DIR / "bt_curves_full.csv").exists():
        raise FileNotFoundError(
            f"Missing {CSV_DIR / 'bt_curves_full.csv'}; run rebuild_dwf_revision_package.py first."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[debug] output directory: {OUT_DIR}")
    for material, stem in (("As", "As_comparison"), ("Sb", "Sb_comparison"), ("Bi_noSOC", "Bi_comparison")):
        write_comparison_wide(material, stem)
        write_experiment_scatter(material, stem)
        write_residuals_wide(material, stem)
    write_sb_apl_wide()
    write_bi_soc_wide()
    write_summary_heatmaps()
    print("[debug] export_figure_wide_csvs.py finished OK")


if __name__ == "__main__":
    main()
