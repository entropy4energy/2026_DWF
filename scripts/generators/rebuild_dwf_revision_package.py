#!/usr/bin/env python3
"""Rebuild the DWF manuscript revision package from raw calculations."""

from __future__ import annotations

import csv
import lzma
import math
import runpy
import sys
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TextIO

BASE_DIR = Path(__file__).resolve().parent
# Every path resolves through scripts/paths.py, the single source of truth for
# this package. Nothing outside the repository root is read or written.
#
# In the manuscript project this orchestrator reached outside the tree for two
# inputs -- the raw AFLOW run trees and the plotting config that carries the
# hand-digitized Fischer 1978 arrays. Both now live inside the package:
# data/dft_raw/ and scripts/vendor/dwf_plot_config.py. That is what makes the
# seven master tables below regenerable from a fresh clone.
sys.path.insert(0, str(BASE_DIR.parent))
from paths import (  # noqa: E402
    PROCESSED,
    REPO_ROOT,
    VENDOR,
    calc_dir,
)

PIPELINE_ROOT = VENDOR

CSV_DIR = PROCESSED

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks


H_SI = 6.62607015e-34
KB_SI = 1.380649e-23
AMU_TO_KG = 1.66053906660e-27
AMU_TO_G = 1.66053906660e-24
EIGHT_PI_SQUARED = 8.0 * np.pi**2
THZ_PER_EV = 241.7989242084918

CSV_FLOAT_PRECISION = 8
FULL_CURVE_STEP_K = 10.0
CROSSCHECK_TOL = 5.0e-6


METHOD_ORDER_ALL = (
    "empirical_digitized_reference",
    "elastic_derived_debye",
    "quasi_harmonic_debye",
    "harmonic_phonon_spectrum_isotropic",
    "harmonic_phonon_spectrum_basal_plane_average",
    "harmonic_phonon_spectrum_c_axis",
)

METHOD_ORDER_MAIN = (
    "empirical_digitized_reference",
    "elastic_derived_debye",
    "quasi_harmonic_debye",
    "harmonic_phonon_spectrum_isotropic",
)

# Methods scored against the Fischer (1978) points. Fischer refined an isotropic
# B only, so the basal-plane and c-axis components have no like-for-like
# reference: their curves are exported, but they get no residuals or metrics.
SCORED_METHODS = METHOD_ORDER_MAIN

METHOD_LABELS = {
    "empirical_digitized_reference": "empirical digitized reference",
    "elastic_derived_debye": "elastic-derived Debye",
    "quasi_harmonic_debye": "quasi-harmonic Debye",
    "harmonic_phonon_spectrum_isotropic": "harmonic phonon spectrum (isotropic)",
    "harmonic_phonon_spectrum_basal_plane_average": "harmonic phonon spectrum (basal-plane average)",
    "harmonic_phonon_spectrum_c_axis": "harmonic phonon spectrum (c-axis)",
}

EXPORT_METHOD_COLUMNS = OrderedDict(
    [
        ("empirical_digitized_reference", "empirical_digitized_reference"),
        ("elastic_derived_debye", "elastic_derived_debye"),
        ("quasi_harmonic_debye", "quasi_harmonic_debye"),
        ("harmonic_phonon_spectrum_isotropic", "harmonic_phonon_spectrum_isotropic"),
        ("harmonic_phonon_spectrum_basal_plane_average", "harmonic_phonon_spectrum_basal_plane_average"),
        ("harmonic_phonon_spectrum_c_axis", "harmonic_phonon_spectrum_c_axis"),
    ]
)

DESCRIPTOR_COLUMNS = [
    "material",
    "soc_status",
    "a_hex_A",
    "c_hex_A",
    "c_over_a",
    "x_Wyckoff",
    "V_per_atom_A3",
    "density_g_cm3",
    "B_VRH_GPa",
    "G_VRH_GPa",
    "Y_VRH_GPa",
    "poisson_ratio",
    "pugh_ratio_G_over_B",
    "elastic_anisotropy",
    "C11_GPa",
    "C33_GPa",
    "C44_GPa",
    "C66_GPa",
    "C33_over_C11",
    "C44_over_C66",
    "B_Voigt_GPa",
    "B_Reuss_GPa",
    "G_Voigt_GPa",
    "G_Reuss_GPa",
    "v_trans_m_s",
    "v_long_m_s",
    "v_avg_m_s",
    "debye_T_elastic_K",
    "debye_T_quasi_harmonic_K",
    "acoustic_debye_T_K",
    "gruneisen_param",
    "thermal_expansion_300K_1_per_K",
    "B_static_300K_GPa",
    "B_isothermal_300K_GPa",
    "kappa_300K_W_per_mK",
    "Cv_300K_kB_per_cell",
    "Cp_300K_kB_per_cell",
    "ZPE_meV_per_atom",
    "mean_freq_THz",
    "log_avg_freq_THz",
    "min_freq_THz",
    "imaginary_freqs",
    "first_peak_THz",
    "dos_int_0_1THz",
    "dos_int_0_2THz",
    "dos_frac_0_1THz",
    "dos_frac_0_2THz",
]


@dataclass(frozen=True)
class MaterialSpec:
    output_name: str
    config_key: str
    soc_status: str
    source_dir: Path
    has_phonons: bool
    figure_symbol: str


MATERIALS = OrderedDict(
    [
        (
            "As",
            MaterialSpec(
                output_name="As",
                config_key="As",
                soc_status="noSOC",
                source_dir=calc_dir("As"),
                has_phonons=True,
                figure_symbol="As",
            ),
        ),
        (
            "Sb",
            MaterialSpec(
                output_name="Sb",
                config_key="Sb",
                soc_status="noSOC",
                source_dir=calc_dir("Sb"),
                has_phonons=True,
                figure_symbol="Sb",
            ),
        ),
        (
            "Bi_noSOC",
            MaterialSpec(
                output_name="Bi_noSOC",
                config_key="Bi",
                soc_status="noSOC",
                source_dir=calc_dir("Bi"),
                has_phonons=True,
                figure_symbol="Bi",
            ),
        ),
        (
            "Bi_SOC",
            MaterialSpec(
                output_name="Bi_SOC",
                config_key="Bi",
                soc_status="SOC",
                source_dir=calc_dir("Bi_SOC"),
                has_phonons=False,
                figure_symbol="Bi",
            ),
        ),
    ]
)


def open_text_auto(path: Path) -> TextIO:
    if path.suffix == ".xz":
        return lzma.open(path, "rt", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def load_pipeline_elements() -> dict[str, dict[str, object]]:
    config_path = PIPELINE_ROOT / "dwf_plot_config.py"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing plotting config: {config_path}")
    namespace = runpy.run_path(str(config_path))
    elements = namespace.get("ELEMENTS")
    if not isinstance(elements, dict):
        raise ValueError(f"Could not load ELEMENTS from {config_path}")
    return elements


def parse_scalar_table(path: Path) -> dict[str, float]:
    data: dict[str, float] = {}
    with open_text_auto(path) as infile:
        for raw_line in infile:
            line = raw_line.strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            try:
                data[key] = float(value.split()[0])
            except ValueError:
                continue
    return data


def read_elastic_tensor(path: Path) -> np.ndarray:
    rows = []
    with open_text_auto(path) as infile:
        for raw_line in infile:
            if raw_line.lstrip().startswith("#"):
                continue
            parts = raw_line.split()
            if len(parts) == 6:
                rows.append([float(part) for part in parts])
    if len(rows) < 6:
        raise ValueError(f"Could not read a 6x6 elastic tensor from {path}")
    return np.asarray(rows[:6], dtype=float)


def read_apl_displacements(path: Path) -> dict[str, np.ndarray]:
    rows_by_temp: dict[float, list[tuple[float, float, float]]] = {}
    current_temp: float | None = None
    with open_text_auto(path) as infile:
        for raw_line in infile:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("["):
                continue
            parts = line.split()
            if len(parts) == 5:
                try:
                    current_temp = float(parts[0])
                    x_val = float(parts[2])
                    y_val = float(parts[3])
                    z_val = float(parts[4])
                except ValueError:
                    continue
            elif len(parts) == 4 and current_temp is not None:
                try:
                    x_val = float(parts[1])
                    y_val = float(parts[2])
                    z_val = float(parts[3])
                except ValueError:
                    continue
            else:
                continue
            rows_by_temp.setdefault(current_temp, []).append((x_val, y_val, z_val))
    if not rows_by_temp:
        raise ValueError(f"No APL displacement rows found in {path}")

    temperatures = np.asarray(sorted(rows_by_temp), dtype=float)
    u_x = np.asarray([np.mean([row[0] for row in rows_by_temp[temp]]) for temp in temperatures], dtype=float)
    u_y = np.asarray([np.mean([row[1] for row in rows_by_temp[temp]]) for temp in temperatures], dtype=float)
    u_z = np.asarray([np.mean([row[2] for row in rows_by_temp[temp]]) for temp in temperatures], dtype=float)
    u_basal = 0.5 * (u_x + u_y)
    u_iso = (u_x + u_y + u_z) / 3.0
    return {
        "temperature": temperatures,
        "U_x": u_x,
        "U_y": u_y,
        "U_z": u_z,
        "U_basal": u_basal,
        "U_iso": u_iso,
        "B_basal": EIGHT_PI_SQUARED * u_basal,
        "B_c_axis": EIGHT_PI_SQUARED * u_z,
        "B_iso": EIGHT_PI_SQUARED * u_iso,
    }


def read_structure_from_contcar(path: Path, mass_amu: float) -> dict[str, float]:
    with open_text_auto(path) as infile:
        lines = [line.rstrip() for line in infile if line.strip()]
    scale = float(lines[1].split()[0])
    lattice = np.asarray([[float(value) for value in lines[idx].split()] for idx in range(2, 5)], dtype=float)
    lattice *= scale
    species_counts = [int(value) for value in lines[6].split()]
    natoms = int(sum(species_counts))
    coord_mode_idx = 7
    coords = np.asarray(
        [[float(value) for value in lines[coord_mode_idx + 1 + idx].split()[:3]] for idx in range(natoms)],
        dtype=float,
    )
    a_hex = float(np.linalg.norm(lattice[0]))
    c_hex = float(np.linalg.norm(lattice[2]))
    volume = float(abs(np.linalg.det(lattice)))
    v_per_atom = volume / natoms
    density = natoms * mass_amu * AMU_TO_G / (volume * 1.0e-24)

    wyckoff_candidates = []
    for coord in coords:
        x_frac = coord[0] % 1.0
        y_frac = coord[1] % 1.0
        if min(abs(x_frac), abs(1.0 - x_frac)) < 1.0e-6 and min(abs(y_frac), abs(1.0 - y_frac)) < 1.0e-6:
            z_frac = coord[2] % 1.0
            wyckoff_candidates.append(z_frac if z_frac <= 0.5 else 1.0 - z_frac)
    if not wyckoff_candidates:
        raise ValueError(f"Could not identify the Wyckoff coordinate in {path}")

    return {
        "a_hex_A": a_hex,
        "c_hex_A": c_hex,
        "c_over_a": c_hex / a_hex,
        "x_Wyckoff": float(np.mean(wyckoff_candidates)),
        "V_per_atom_A3": v_per_atom,
        "density_g_cm3": density,
    }


def read_phdos_descriptors(path: Path) -> dict[str, float | str]:
    rows = []
    with open_text_auto(path) as infile:
        for idx, raw_line in enumerate(infile):
            if idx < 6:
                continue
            parts = raw_line.split()
            if len(parts) < 2:
                continue
            rows.append((float(parts[0]) * THZ_PER_EV, float(parts[1])))
    if not rows:
        raise ValueError(f"No phonon DOS rows found in {path}")

    data = np.asarray(rows, dtype=float)
    freq = data[:, 0]
    dos = data[:, 1]

    imag_mask = (freq < -1.0e-3) & (dos > 1.0e-6 * np.max(dos))
    positive_mask = freq > 0.0
    freq_pos = freq[positive_mask]
    dos_pos = dos[positive_mask]

    total_int = float(np.trapezoid(dos_pos, freq_pos))

    def partial_integral(cutoff: float) -> float:
        mask = freq_pos <= cutoff
        if not np.any(mask):
            return 0.0
        return float(np.trapezoid(dos_pos[mask], freq_pos[mask]))

    dos_int_0_1 = partial_integral(1.0)
    dos_int_0_2 = partial_integral(2.0)
    mean_freq = float(np.trapezoid(freq_pos * dos_pos, freq_pos) / total_int)
    log_mask = (freq_pos > 0.0) & (dos_pos > 0.0)
    log_avg = float(
        np.exp(
            np.trapezoid(np.log(freq_pos[log_mask]) * dos_pos[log_mask], freq_pos[log_mask])
            / np.trapezoid(dos_pos[log_mask], freq_pos[log_mask])
        )
    )

    peaks, _ = find_peaks(dos_pos)
    prominent_candidates = [
        freq_pos[index]
        for index in peaks
        if freq_pos[index] >= 0.5 and dos_pos[index] >= 0.15 * np.max(dos_pos)
    ]
    if prominent_candidates:
        first_peak = float(prominent_candidates[0])
    elif len(peaks):
        first_peak = float(freq_pos[peaks[0]])
    else:
        first_peak = float(freq_pos[np.argmax(dos_pos)])

    return {
        "min_freq_THz": float(np.min(freq_pos)),
        "dos_int_0_1THz": dos_int_0_1,
        "dos_int_0_2THz": dos_int_0_2,
        "dos_frac_0_1THz": dos_int_0_1 / total_int,
        "dos_frac_0_2THz": dos_int_0_2 / total_int,
        "mean_freq_THz": mean_freq,
        "log_avg_freq_THz": log_avg,
        "imaginary_freqs": "Yes" if np.any(imag_mask) else "No",
        "first_peak_THz": first_peak,
    }


def sanitize_curve_points(temperatures: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    temps = np.asarray(temperatures, dtype=float).copy()
    vals = np.asarray(values, dtype=float).copy()
    near_zero_mask = (temps < 0.0) & (temps > -10.0)
    temps[near_zero_mask] = 0.0
    valid_mask = np.isfinite(temps) & np.isfinite(vals)
    temps = temps[valid_mask]
    vals = vals[valid_mask]
    order = np.argsort(temps)
    temps = temps[order]
    vals = vals[order]
    unique_temps = []
    unique_vals = []
    start = 0
    for idx in range(1, len(temps) + 1):
        if idx == len(temps) or temps[idx] != temps[start]:
            unique_temps.append(temps[start])
            unique_vals.append(vals[start:idx].mean())
            start = idx
    return np.asarray(unique_temps, dtype=float), np.asarray(unique_vals, dtype=float)


def make_pchip_interpolator(temperatures: np.ndarray, values: np.ndarray) -> PchipInterpolator:
    temps, vals = sanitize_curve_points(temperatures, values)
    if len(temps) < 2:
        raise ValueError("Need at least two points for interpolation")
    return PchipInterpolator(temps, vals, extrapolate=False)


def evaluate_interpolator(interpolator: PchipInterpolator, x_dst: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    x_dst = np.asarray(x_dst, dtype=float)
    y_dst = np.asarray(interpolator(x_dst), dtype=float)
    outside = (x_dst < x_min) | (x_dst > x_max)
    y_dst[outside] = np.nan
    return y_dst


@lru_cache(maxsize=None)
def _debye_integral_cached(theta_over_t: float) -> float:
    if theta_over_t <= 0.0:
        return float(np.pi**2 / 6.0)

    def integrand(x_value: float) -> float:
        return x_value / np.expm1(x_value)

    integral_value, _ = quad(integrand, 0.0, theta_over_t, limit=200, epsabs=1.0e-10, epsrel=1.0e-9)
    return float(integral_value)


@lru_cache(maxsize=None)
def _debye_b_factor_scalar(temp: float, theta_d: float, mass_amu: float) -> float:
    if theta_d <= 0.0:
        return float("nan")
    mass_kg = mass_amu * AMU_TO_KG
    prefactor = 6.0 * H_SI**2 / (mass_kg * KB_SI * theta_d)
    if temp <= 0.0:
        return float(prefactor * 0.25 * 1.0e20)
    theta_over_t = theta_d / temp
    integral_value = _debye_integral_cached(theta_over_t)
    return float(prefactor * (0.25 + (temp / theta_d) ** 2 * integral_value) * 1.0e20)


def debye_b_factor(temperatures: np.ndarray, theta_d: float | np.ndarray, mass_amu: float) -> np.ndarray:
    temps = np.asarray(temperatures, dtype=float)
    theta_values = np.asarray(theta_d, dtype=float)
    theta_values = np.broadcast_to(theta_values, temps.shape)
    result = np.full_like(temps, np.nan, dtype=float)
    for index in np.ndindex(temps.shape):
        theta_local = float(theta_values[index])
        if np.isfinite(theta_local) and theta_local > 0.0:
            result[index] = _debye_b_factor_scalar(float(temps[index]), theta_local, mass_amu)
    return result


def evaluate_model(exp_values: np.ndarray, predicted_values: np.ndarray) -> dict[str, float]:
    exp_values = np.asarray(exp_values, dtype=float)
    predicted_values = np.asarray(predicted_values, dtype=float)
    mask = np.isfinite(exp_values) & np.isfinite(predicted_values)
    n_points = int(np.sum(mask))
    if n_points < 2:
        return {"N_points": n_points, "RMSE": np.nan, "MAE": np.nan, "NRMSE": np.nan, "R2": np.nan}
    exp_values = exp_values[mask]
    predicted_values = predicted_values[mask]
    residual = predicted_values - exp_values
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    exp_range = float(np.max(exp_values) - np.min(exp_values))
    nrmse = rmse / exp_range if exp_range > 0.0 else 0.0
    ss_tot = float(np.sum((exp_values - np.mean(exp_values)) ** 2))
    ss_res = float(np.sum((exp_values - predicted_values) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return {"N_points": n_points, "RMSE": rmse, "MAE": mae, "NRMSE": nrmse, "R2": r2}


def plot_temperature_max(exp_temps: np.ndarray) -> float:
    exp_temps = np.asarray(exp_temps, dtype=float)
    exp_min = float(np.min(exp_temps))
    exp_max = float(np.max(exp_temps))
    exp_span = max(exp_max - exp_min, 1.0)
    return exp_max + max(25.0, 0.05 * exp_span)


def full_curve_grid(exp_temps: np.ndarray) -> np.ndarray:
    max_temp = plot_temperature_max(exp_temps)
    max_grid = math.ceil(max_temp / FULL_CURVE_STEP_K) * FULL_CURVE_STEP_K
    return np.arange(0.0, max_grid + 0.5 * FULL_CURVE_STEP_K, FULL_CURVE_STEP_K, dtype=float)


def format_csv_number(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "NA"
        abs_value = abs(float(value))
        if abs_value != 0.0 and (abs_value < 1.0e-4 or abs_value >= 1.0e4):
            return f"{float(value):.{CSV_FLOAT_PRECISION}e}"
        return f"{float(value):.{CSV_FLOAT_PRECISION}f}".rstrip("0").rstrip(".")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def read_zpe_mev_per_atom(path: Path) -> float:
    return parse_scalar_table(path)["energy_zero_point_atom_apl"]


def low_frequency_note(dos_frac_0_2thz: float) -> str:
    if dos_frac_0_2thz < 0.15:
        return "low-frequency weight is limited; spectrum is relatively stiff"
    if dos_frac_0_2thz < 0.45:
        return "substantial low-frequency spectral weight remains below 2 THz"
    return "very soft spectrum; roughly half of the phonon weight lies below 2 THz"


def build_source_paths(spec: MaterialSpec) -> dict[str, Path]:
    return {
        "structure": spec.source_dir / "CONTCAR.static.xz",
        "ael": spec.source_dir / "aflow.ael.out.xz",
        "agl": spec.source_dir / "aflow.agl.out.xz",
        "elastic_tensor": spec.source_dir / "AEL_Elastic_constants.out.xz",
        "compliance_tensor": spec.source_dir / "AEL_Compliance_tensor.out.xz",
        "apl": spec.source_dir / "aflow.apl.out.xz",
        "apl_displacements": spec.source_dir / "aflow.apl.displacements.out.xz",
        "phdos": spec.source_dir / "PHDOSCAR.xz",
    }


def build_material_bundle(spec: MaterialSpec, config: dict[str, object]) -> dict[str, object]:
    source_paths = build_source_paths(spec)
    mass_amu = float(config["mass_amu"])
    exp_temps = np.asarray(config["exp_temps"], dtype=float)
    exp_values = np.asarray(config["exp_b"], dtype=float)
    empirical_temps, empirical_values = sanitize_curve_points(
        np.asarray(config["empirical_temps"], dtype=float),
        np.asarray(config["empirical_b"], dtype=float),
    )

    ael_summary = parse_scalar_table(source_paths["ael"])
    agl_summary = parse_scalar_table(source_paths["agl"])
    elastic_tensor = read_elastic_tensor(source_paths["elastic_tensor"])
    structure = read_structure_from_contcar(source_paths["structure"], mass_amu)

    descriptor_row: dict[str, object] = {
        "material": spec.output_name,
        "soc_status": spec.soc_status,
        **structure,
        "B_VRH_GPa": ael_summary["ael_bulk_modulus_vrh"],
        "G_VRH_GPa": ael_summary["ael_shear_modulus_vrh"],
        "Y_VRH_GPa": ael_summary["ael_youngs_modulus_vrh"],
        "poisson_ratio": ael_summary["ael_poisson_ratio"],
        "pugh_ratio_G_over_B": ael_summary["ael_pughs_modulus_ratio"],
        "elastic_anisotropy": ael_summary["ael_elastic_anisotropy"],
        "C11_GPa": 0.5 * (elastic_tensor[0, 0] + elastic_tensor[1, 1]),
        "C33_GPa": elastic_tensor[2, 2],
        "C44_GPa": elastic_tensor[3, 3],
        "C66_GPa": elastic_tensor[5, 5],
        "B_Voigt_GPa": ael_summary["ael_bulk_modulus_voigt"],
        "B_Reuss_GPa": ael_summary["ael_bulk_modulus_reuss"],
        "G_Voigt_GPa": ael_summary["ael_shear_modulus_voigt"],
        "G_Reuss_GPa": ael_summary["ael_shear_modulus_reuss"],
        "v_trans_m_s": ael_summary["ael_speed_sound_transverse"],
        "v_long_m_s": ael_summary["ael_speed_sound_longitudinal"],
        "v_avg_m_s": ael_summary["ael_speed_sound_average"],
        "debye_T_elastic_K": ael_summary["ael_debye_temperature"],
        "debye_T_quasi_harmonic_K": agl_summary["agl_debye"],
        "acoustic_debye_T_K": agl_summary["agl_acoustic_debye"],
        "gruneisen_param": agl_summary["agl_gruneisen"],
        "thermal_expansion_300K_1_per_K": agl_summary["agl_thermal_expansion_300K"],
        "B_static_300K_GPa": agl_summary["agl_bulk_modulus_static_300K"],
        "B_isothermal_300K_GPa": agl_summary["agl_bulk_modulus_isothermal_300K"],
        "kappa_300K_W_per_mK": agl_summary["agl_thermal_conductivity_300K"],
        "Cv_300K_kB_per_cell": agl_summary["agl_heat_capacity_Cv_300K"],
        "Cp_300K_kB_per_cell": agl_summary["agl_heat_capacity_Cp_300K"],
    }
    descriptor_row["C33_over_C11"] = descriptor_row["C33_GPa"] / descriptor_row["C11_GPa"]
    descriptor_row["C44_over_C66"] = descriptor_row["C44_GPa"] / descriptor_row["C66_GPa"]

    grid_temps = full_curve_grid(exp_temps)
    empirical_interp = make_pchip_interpolator(empirical_temps, empirical_values)
    empirical_t_min = float(np.min(empirical_temps))
    empirical_t_max = float(np.max(empirical_temps))

    curves_grid: dict[str, np.ndarray] = {}
    curves_exp: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}

    curves_grid["empirical_digitized_reference"] = evaluate_interpolator(
        empirical_interp, grid_temps, empirical_t_min, empirical_t_max
    )
    curves_exp["empirical_digitized_reference"] = evaluate_interpolator(
        empirical_interp, exp_temps, empirical_t_min, empirical_t_max
    )
    curves_grid["elastic_derived_debye"] = debye_b_factor(grid_temps, ael_summary["ael_debye_temperature"], mass_amu)
    curves_exp["elastic_derived_debye"] = debye_b_factor(exp_temps, ael_summary["ael_debye_temperature"], mass_amu)

    # The quasi-harmonic model uses the single AGL summary Debye temperature
    # (agl_debye, the Theta_D^QH of the descriptor table), as the cF8 analyses
    # and the SI unification figure do, not the temperature-dependent Theta(T)
    # column of AGL_thermal_properties_temperature.out.
    curves_grid["quasi_harmonic_debye"] = debye_b_factor(grid_temps, agl_summary["agl_debye"], mass_amu)
    curves_exp["quasi_harmonic_debye"] = debye_b_factor(exp_temps, agl_summary["agl_debye"], mass_amu)

    low_freq_summary_row: dict[str, object] | None = None
    if spec.has_phonons:
        apl_data = read_apl_displacements(source_paths["apl_displacements"])
        apl_t_min = float(np.min(apl_data["temperature"]))
        apl_t_max = float(np.max(apl_data["temperature"]))

        apl_iso_interp = make_pchip_interpolator(apl_data["temperature"], apl_data["B_iso"])
        apl_basal_interp = make_pchip_interpolator(apl_data["temperature"], apl_data["B_basal"])
        apl_c_axis_interp = make_pchip_interpolator(apl_data["temperature"], apl_data["B_c_axis"])

        curves_grid["harmonic_phonon_spectrum_isotropic"] = evaluate_interpolator(
            apl_iso_interp, grid_temps, apl_t_min, apl_t_max
        )
        curves_exp["harmonic_phonon_spectrum_isotropic"] = evaluate_interpolator(
            apl_iso_interp, exp_temps, apl_t_min, apl_t_max
        )
        curves_grid["harmonic_phonon_spectrum_basal_plane_average"] = evaluate_interpolator(
            apl_basal_interp, grid_temps, apl_t_min, apl_t_max
        )
        curves_exp["harmonic_phonon_spectrum_basal_plane_average"] = evaluate_interpolator(
            apl_basal_interp, exp_temps, apl_t_min, apl_t_max
        )
        curves_grid["harmonic_phonon_spectrum_c_axis"] = evaluate_interpolator(
            apl_c_axis_interp, grid_temps, apl_t_min, apl_t_max
        )
        curves_exp["harmonic_phonon_spectrum_c_axis"] = evaluate_interpolator(
            apl_c_axis_interp, exp_temps, apl_t_min, apl_t_max
        )

        descriptor_row["ZPE_meV_per_atom"] = read_zpe_mev_per_atom(source_paths["apl"])
        phdos_desc = read_phdos_descriptors(source_paths["phdos"])
        descriptor_row.update(phdos_desc)
        low_freq_summary_row = {
            "material": spec.output_name,
            "soc_status": spec.soc_status,
            "min_freq_THz": phdos_desc["min_freq_THz"],
            "first_peak_THz": phdos_desc["first_peak_THz"],
            "dos_frac_0_1THz": phdos_desc["dos_frac_0_1THz"],
            "dos_frac_0_2THz": phdos_desc["dos_frac_0_2THz"],
            "mean_freq_THz": phdos_desc["mean_freq_THz"],
            "log_avg_freq_THz": phdos_desc["log_avg_freq_THz"],
            "short_note": low_frequency_note(float(phdos_desc["dos_frac_0_2THz"])),
        }
    else:
        descriptor_row.update(
            {
                "ZPE_meV_per_atom": None,
                "mean_freq_THz": None,
                "log_avg_freq_THz": None,
                "min_freq_THz": None,
                "imaginary_freqs": None,
                "first_peak_THz": None,
                "dos_int_0_1THz": None,
                "dos_int_0_2THz": None,
                "dos_frac_0_1THz": None,
                "dos_frac_0_2THz": None,
            }
        )
        low_freq_summary_row = {
            "material": spec.output_name,
            "soc_status": spec.soc_status,
            "min_freq_THz": None,
            "first_peak_THz": None,
            "dos_frac_0_1THz": None,
            "dos_frac_0_2THz": None,
            "mean_freq_THz": None,
            "log_avg_freq_THz": None,
            "short_note": "phonon DOS not available from the SOC calculation",
        }

    available_methods = ["elastic_derived_debye", "quasi_harmonic_debye"]
    if spec.has_phonons:
        available_methods = list(METHOD_ORDER_ALL)
    elif spec.output_name == "Bi_SOC":
        available_methods = ["elastic_derived_debye", "quasi_harmonic_debye"]

    for method_id in available_methods:
        if method_id in SCORED_METHODS:
            metrics[method_id] = evaluate_model(exp_values, curves_exp[method_id])

    return {
        "spec": spec,
        "config": config,
        "source_paths": source_paths,
        "exp_temps": exp_temps,
        "exp_values": exp_values,
        "empirical_temps": empirical_temps,
        "empirical_values": empirical_values,
        "grid_temps": grid_temps,
        "curves_grid": curves_grid,
        "curves_exp": curves_exp,
        "metrics": metrics,
        "descriptor_row": descriptor_row,
        "low_freq_summary_row": low_freq_summary_row,
        "available_methods": available_methods,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_csv_number(row.get(key)) for key in fieldnames})


def build_descriptor_rows(bundles: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    return [bundles[name]["descriptor_row"] for name in MATERIALS]


def build_low_frequency_rows(bundles: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for name in MATERIALS:
        row = bundles[name]["low_freq_summary_row"]
        if row is not None:
            rows.append(row)
    return rows


def build_bi_delta_rows(bundles: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    bi_no = bundles["Bi_noSOC"]["descriptor_row"]
    bi_soc = bundles["Bi_SOC"]["descriptor_row"]
    descriptor_names = [
        "a_hex_A",
        "c_hex_A",
        "V_per_atom_A3",
        "density_g_cm3",
        "B_VRH_GPa",
        "G_VRH_GPa",
        "Y_VRH_GPa",
        "poisson_ratio",
        "elastic_anisotropy",
        "debye_T_elastic_K",
        "debye_T_quasi_harmonic_K",
        "acoustic_debye_T_K",
        "gruneisen_param",
        "thermal_expansion_300K_1_per_K",
        "B_static_300K_GPa",
        "B_isothermal_300K_GPa",
        "kappa_300K_W_per_mK",
        "C11_GPa",
        "C33_GPa",
        "C44_GPa",
        "C66_GPa",
        "B_Voigt_GPa",
        "B_Reuss_GPa",
        "G_Voigt_GPa",
        "G_Reuss_GPa",
        "v_trans_m_s",
        "v_long_m_s",
        "v_avg_m_s",
    ]

    rows = []
    for name in descriptor_names:
        old_val = bi_no.get(name)
        new_val = bi_soc.get(name)
        if old_val is None or new_val is None:
            continue
        abs_change = float(new_val) - float(old_val)
        percent_change = 100.0 * abs_change / float(old_val) if float(old_val) != 0.0 else np.nan
        if "debye" in name.lower():
            direction = "drops" if abs_change < 0 else "rises"
            short = f"SOC {direction} the characteristic Debye scale"
        elif "kappa" in name.lower():
            short = "SOC lowers the lattice thermal conductivity" if abs_change < 0 else "SOC raises the lattice thermal conductivity"
        elif "thermal_expansion" in name.lower():
            short = "SOC increases the thermal expansion coefficient" if abs_change > 0 else "SOC decreases the thermal expansion coefficient"
        elif "poisson" in name.lower():
            short = "SOC shifts the Poisson ratio upward" if abs_change > 0 else "SOC shifts the Poisson ratio downward"
        elif abs_change < 0:
            short = "SOC softens this quantity"
        else:
            short = "SOC increases this quantity"
        rows.append(
            {
                "descriptor": name,
                "Bi_noSOC": old_val,
                "Bi_SOC": new_val,
                "abs_change": abs_change,
                "percent_change": percent_change,
                "interpretation_short": short,
            }
        )
    return rows


def build_metric_rows(bundles: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for material_name, bundle in bundles.items():
        for method_id, metric in bundle["metrics"].items():
            rows.append(
                {
                    "material": material_name,
                    "soc_status": bundle["spec"].soc_status,
                    "method": METHOD_LABELS[method_id],
                    "N_points": metric["N_points"],
                    "RMSE": metric["RMSE"],
                    "MAE": metric["MAE"],
                    "NRMSE": metric["NRMSE"],
                    "R2": metric["R2"],
                }
            )
    return rows


def build_bt_at_experiment_rows(bundles: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for material_name, bundle in bundles.items():
        exp_temps = bundle["exp_temps"]
        exp_values = bundle["exp_values"]
        available = set(bundle["available_methods"])
        for idx, temp in enumerate(exp_temps):
            row: dict[str, object] = {
                "material": material_name,
                "soc_status": bundle["spec"].soc_status,
                "T_K": temp,
                "B_exp_A2": exp_values[idx],
            }
            for method_id, column_stub in EXPORT_METHOD_COLUMNS.items():
                pred_value = residual = None
                if method_id in bundle["curves_exp"] and (material_name != "Bi_SOC" or method_id in available):
                    pred_value = float(bundle["curves_exp"][method_id][idx])
                    residual = pred_value - exp_values[idx] if np.isfinite(pred_value) else None
                row[f"B_{column_stub}_A2"] = pred_value
                if method_id in SCORED_METHODS:
                    row[f"residual_{column_stub}"] = residual
            rows.append(row)
    return rows


def build_full_curve_rows(bundles: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for material_name, bundle in bundles.items():
        for method_id, values in bundle["curves_grid"].items():
            if material_name == "Bi_SOC" and method_id not in bundle["available_methods"]:
                continue
            for temp, b_value in zip(bundle["grid_temps"], values):
                rows.append(
                    {
                        "material": material_name,
                        "soc_status": bundle["spec"].soc_status,
                        "T_K": temp,
                        "method": METHOD_LABELS[method_id],
                        "B_A2": b_value,
                    }
                )
    return rows


def build_ranking_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    by_material: dict[str, list[dict[str, object]]] = {}
    for row in metric_rows:
        by_material.setdefault(str(row["material"]), []).append(row)
    for material_name, group in by_material.items():
        soc_status = str(group[0]["soc_status"])
        for metric_name, reverse in (("NRMSE", False), ("RMSE", False), ("R2", True)):
            sortable = [row for row in group if np.isfinite(float(row[metric_name]))]
            sortable.sort(key=lambda item: float(item[metric_name]), reverse=reverse)
            for rank, row in enumerate(sortable, start=1):
                rows.append(
                    {
                        "material": material_name,
                        "soc_status": soc_status,
                        "rank_metric": metric_name,
                        "rank": rank,
                        "method": row["method"],
                        "metric_value": row[metric_name],
                    }
                )
    return rows


def export_package_tables(bundles: dict[str, dict[str, object]]) -> dict[str, Path]:
    descriptor_rows = build_descriptor_rows(bundles)
    low_frequency_rows = build_low_frequency_rows(bundles)
    delta_rows = build_bi_delta_rows(bundles)
    metric_rows = build_metric_rows(bundles)
    bt_exp_rows = build_bt_at_experiment_rows(bundles)
    full_curve_rows = build_full_curve_rows(bundles)
    ranking_rows = build_ranking_rows(metric_rows)

    descriptor_path = CSV_DIR / "descriptor_table.csv"
    delta_path = CSV_DIR / "Bi_SOC_vs_noSOC_delta.csv"
    metrics_path = CSV_DIR / "benchmark_metrics.csv"
    bt_exp_path = CSV_DIR / "bt_at_experiment_temperatures.csv"
    full_curve_path = CSV_DIR / "bt_curves_full.csv"
    ranking_path = CSV_DIR / "material_method_ranking.csv"
    low_freq_path = CSV_DIR / "low_frequency_phonon_summary.csv"

    write_csv(descriptor_path, descriptor_rows, DESCRIPTOR_COLUMNS)
    write_csv(
        delta_path,
        delta_rows,
        ["descriptor", "Bi_noSOC", "Bi_SOC", "abs_change", "percent_change", "interpretation_short"],
    )
    write_csv(
        metrics_path,
        metric_rows,
        ["material", "soc_status", "method", "N_points", "RMSE", "MAE", "NRMSE", "R2"],
    )
    bt_fields = ["material", "soc_status", "T_K", "B_exp_A2"]
    for method_id, column_stub in EXPORT_METHOD_COLUMNS.items():
        bt_fields.append(f"B_{column_stub}_A2")
        if method_id in SCORED_METHODS:
            bt_fields.append(f"residual_{column_stub}")
    write_csv(bt_exp_path, bt_exp_rows, bt_fields)
    write_csv(full_curve_path, full_curve_rows, ["material", "soc_status", "T_K", "method", "B_A2"])
    write_csv(ranking_path, ranking_rows, ["material", "soc_status", "rank_metric", "rank", "method", "metric_value"])
    write_csv(
        low_freq_path,
        low_frequency_rows,
        [
            "material",
            "soc_status",
            "min_freq_THz",
            "first_peak_THz",
            "dos_frac_0_1THz",
            "dos_frac_0_2THz",
            "mean_freq_THz",
            "log_avg_freq_THz",
            "short_note",
        ],
    )

    return {
        "descriptor_table": descriptor_path,
        "bi_delta": delta_path,
        "benchmark_metrics": metrics_path,
        "bt_at_experiment_temperatures": bt_exp_path,
        "bt_curves_full": full_curve_path,
        "material_method_ranking": ranking_path,
        "low_frequency_phonon_summary": low_freq_path,
    }


def build_crosscheck_text(
    bundles: dict[str, dict[str, object]],
    package_paths: dict[str, Path],
) -> str:
    exp_table: dict[tuple[str, str], list[tuple[float, float]]] = {}
    with package_paths["bt_at_experiment_temperatures"].open("r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            material = row["material"]
            for method_id, column_stub in EXPORT_METHOD_COLUMNS.items():
                pred_col = f"B_{column_stub}_A2"
                value = row[pred_col]
                if value == "NA" or value == "":
                    continue
                exp_table.setdefault((material, method_id), []).append((float(row["T_K"]), float(value)))

    full_curve_table: dict[tuple[str, str], list[tuple[float, float]]] = {}
    label_to_method_id = {label: method_id for method_id, label in METHOD_LABELS.items()}
    with package_paths["bt_curves_full"].open("r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if row["B_A2"] in {"NA", ""}:
                continue
            material = row["material"]
            method_id = label_to_method_id[row["method"]]
            full_curve_table.setdefault((material, method_id), []).append((float(row["T_K"]), float(row["B_A2"])))

    metric_table: dict[tuple[str, str], dict[str, float]] = {}
    with package_paths["benchmark_metrics"].open("r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            metric_table[(row["material"], row["method"])] = {
                "RMSE": float(row["RMSE"]),
                "MAE": float(row["MAE"]),
                "NRMSE": float(row["NRMSE"]),
                "R2": float(row["R2"]),
            }

    lines = [
        "# Rebuild Cross-Check",
        "",
        "Direct source-derived predictions were compared against the written CSV package after export.",
        "",
        "| Material | Method | max |delta| at experiment temps | max |delta| on full curve grid | max metric delta |",
        "|---|---|---:|---:|---:|",
    ]
    worst_diff = 0.0
    for material_name, bundle in bundles.items():
        for method_id in bundle["available_methods"]:
            exp_pairs = exp_table[(material_name, method_id)]
            exp_pairs.sort()
            exp_saved = np.asarray([value for _, value in exp_pairs], dtype=float)
            exp_direct = np.asarray(bundle["curves_exp"][method_id], dtype=float)
            exp_direct_pairs = sorted(
                (float(temp), float(value))
                for temp, value in zip(bundle["exp_temps"], exp_direct)
                if np.isfinite(value)
            )
            exp_direct_aligned = np.asarray([value for _, value in exp_direct_pairs], dtype=float)
            exp_diff = float(np.nanmax(np.abs(exp_saved - exp_direct_aligned)))

            full_pairs = full_curve_table[(material_name, method_id)]
            full_pairs.sort()
            full_direct = np.asarray(bundle["curves_grid"][method_id], dtype=float)
            full_direct_map = {
                float(temp): float(value)
                for temp, value in zip(bundle["grid_temps"], full_direct)
                if np.isfinite(value)
            }
            full_saved = np.asarray([value for _, value in full_pairs], dtype=float)
            full_direct_aligned = np.asarray([full_direct_map[float(temp)] for temp, _ in full_pairs], dtype=float)
            full_diff = float(np.nanmax(np.abs(full_saved - full_direct_aligned)))

            if method_id in bundle["metrics"]:
                metric_saved = metric_table[(material_name, METHOD_LABELS[method_id])]
                metric_direct = bundle["metrics"][method_id]
                metric_diff = max(
                    abs(metric_saved["RMSE"] - metric_direct["RMSE"]),
                    abs(metric_saved["MAE"] - metric_direct["MAE"]),
                    abs(metric_saved["NRMSE"] - metric_direct["NRMSE"]),
                    abs(metric_saved["R2"] - metric_direct["R2"]),
                )
                metric_cell = f"{metric_diff:.3e}"
            else:
                metric_diff, metric_cell = 0.0, "not scored"
            worst_diff = max(worst_diff, exp_diff, full_diff, metric_diff)
            lines.append(
                f"| {material_name} | {METHOD_LABELS[method_id]} | {exp_diff:.3e} | {full_diff:.3e} | {metric_cell} |"
            )

    if worst_diff > CROSSCHECK_TOL:
        raise RuntimeError(
            f"Cross-check failed: maximum difference {worst_diff:.6e} exceeds tolerance {CROSSCHECK_TOL:.6e}"
        )

    lines.extend(
        [
            "",
            f"Maximum observed difference: {worst_diff:.3e}.",
            "All rebuilt material-method pairs match the saved CSV package within the requested tiny formatting-level tolerance.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    elements = load_pipeline_elements()
    bundles = {name: build_material_bundle(spec, elements[spec.config_key]) for name, spec in MATERIALS.items()}

    package_paths = export_package_tables(bundles)
    print(build_crosscheck_text(bundles, package_paths))
    print(f"Rebuilt the master tables in {CSV_DIR.relative_to(REPO_ROOT)}/ from the raw calculations.")


if __name__ == "__main__":
    main()
