#!/usr/bin/env python3
"""Analyze the new C (diamond) APL+AEL+AGL calculation.

C is the second non-A7 control material added after Si. It tests whether the
"scalar Debye is OK in stiff regime" recipe holds at the extreme stiff end of
the periodic table (Theta_D ~ 2200 K, omega_LO ~ 40 THz, light atom).

Mirrors the Si protocol used in `Si_phonon_analysis.py`:
  - DOS descriptors (mean omega, sub-1/2 THz fractions, leverage proxy)
  - Site-mean <u^2>(T) from aflow.apl.displacements.out
  - B_iso(T) = 8 pi^2 <u^2>(T)
  - Scalar Debye B(T) from
      * empirical Debye temperature (lit value, ~2230 K)
      * elastic-derived Theta_D (AEL)
      * quasi-harmonic Theta_D (AGL)

Experimental reference: a single point at 295 K, the source-balanced
median of the 290-300 K values in data/literature/cF8_literature_B_sources.csv
(cross_material_synthesis.load_cf8_literature_median). It is the same
reference that SI Fig. S1 and main-text Table III use.
"""
from __future__ import annotations

import csv
import lzma
import math
from pathlib import Path

import numpy as np

# numpy 2.0 renamed trapz -> trapezoid; alias keeps old and new numpy working.
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# Every path resolves through scripts/paths.py, the single source of truth for
# this package. Nothing outside the repository root is read or written.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import PROCESSED, calc_dir  # noqa: E402
from cross_material_synthesis import load_cf8_literature_median  # noqa: E402


C_DIR = calc_dir("C")
OUT_DIR = PROCESSED
OUT_DIR.mkdir(parents=True, exist_ok=True)
DESCRIPTORS_CSV = OUT_DIR / "C_phonon_descriptors.csv"
CURVES_CSV = OUT_DIR / "C_phonon_B_curves.csv"
BENCHMARK_CSV = OUT_DIR / "C_phonon_experimental_benchmark.csv"

EIGHT_PI_SQ = 8.0 * math.pi**2
H_SI = 6.62607015e-34
KB_SI = 1.380649e-23
AMU_TO_KG = 1.66053906660e-27
M_C = 12.011  # carbon-12 atomic mass


def open_text(path: Path):
    if path.suffix == ".xz":
        return lzma.open(path, "rt", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def read_phonon_dos(path: Path) -> tuple[np.ndarray, np.ndarray]:
    f, d = [], []
    with open_text(path) as fh:
        for line in fh:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                f.append(float(parts[0])); d.append(float(parts[3]))
            except ValueError:
                continue
    return np.asarray(f), np.asarray(d)


def read_apl_displacements(path: Path) -> dict[str, np.ndarray]:
    rows: dict[float, list[tuple[float, float, float]]] = {}
    cur = None
    with open_text(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("["):
                continue
            p = s.split()
            if len(p) == 5:
                try:
                    cur = float(p[0]); xv = float(p[2]); yv = float(p[3]); zv = float(p[4])
                except ValueError:
                    continue
            elif len(p) == 4 and cur is not None:
                try:
                    xv = float(p[1]); yv = float(p[2]); zv = float(p[3])
                except ValueError:
                    continue
            else:
                continue
            rows.setdefault(cur, []).append((xv, yv, zv))
    Ts = np.asarray(sorted(rows))
    ux = np.asarray([np.mean([r[0] for r in rows[T]]) for T in Ts])
    uy = np.asarray([np.mean([r[1] for r in rows[T]]) for T in Ts])
    uz = np.asarray([np.mean([r[2] for r in rows[T]]) for T in Ts])
    u_iso = (ux + uy + uz) / 3.0
    return {
        "T": Ts,
        "U_x": ux, "U_y": uy, "U_z": uz, "U_iso": u_iso,
        "B_iso": EIGHT_PI_SQ * u_iso,
    }


def parse_kv(path: Path) -> dict[str, float]:
    data: dict[str, float] = {}
    with open_text(path) as fh:
        for line in fh:
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            try:
                data[k.strip()] = float(v.split()[0])
            except ValueError:
                continue
    return data


def dos_descriptors(freq: np.ndarray, dos: np.ndarray) -> dict[str, float]:
    pos = freq > 0
    f = freq[pos]; d = dos[pos]
    norm = _trapz(d, f)
    dn = d / norm
    mean_w = float(_trapz(f * dn, f))
    log_w = float(math.exp(_trapz(np.log(f) * dn, f)))

    def frac_below(w):
        m = f < w
        return float(_trapz(dn[m], f[m]))

    cutoff = 0.10
    mask = f >= cutoff
    leverage_total = float(_trapz(dn[mask] / f[mask] ** 2, f[mask]))
    mask2 = (f >= cutoff) & (f < 2.0)
    leverage_below2 = float(_trapz(dn[mask2] / f[mask2] ** 2, f[mask2]))
    return {
        "mean_freq_THz": mean_w,
        "log_avg_freq_THz": log_w,
        "dos_frac_below_1THz": frac_below(1.0),
        "dos_frac_below_2THz": frac_below(2.0),
        "leverage_total": leverage_total,
        "leverage_below_2THz": leverage_below2,
        "leverage_fraction_below_2THz": leverage_below2 / leverage_total if leverage_total else float("nan"),
        "min_freq_THz": float(freq.min()),
        "first_peak_THz": float(f[np.argmax(dn)]),
    }


def debye_phi(x: float) -> float:
    if x < 1e-8:
        return 1.0
    n = 4000
    grid = np.linspace(1e-12, x, n)
    integrand = grid / (np.exp(grid) - 1.0)
    return float(_trapz(integrand, grid)) / x


def B_debye(T: np.ndarray, theta: float, mass_amu: float) -> np.ndarray:
    """Standard isotropic Debye expression in A^2."""
    M_kg = mass_amu * AMU_TO_KG
    out = np.empty_like(T, dtype=float)
    for i, Ti in enumerate(T):
        x = theta / Ti if Ti > 0 else 1.0e9
        phi = debye_phi(x) if Ti > 0 else 0.0
        Bm2 = (6 * H_SI ** 2) / (M_kg * KB_SI * theta) * (phi / x + 0.25)
        out[i] = Bm2 * 1.0e20  # A^2
    return out


def main() -> None:
    end = C_DIR / "aflow.end.out"
    print("=== C (diamond) calc completion ===")
    print(f"  {end.read_text().splitlines()[0][:120] if end.exists() else 'MISSING'}")

    ael = parse_kv(C_DIR / "aflow.ael.out.xz")
    agl = parse_kv(C_DIR / "aflow.agl.out.xz")
    print("\n=== AEL / AGL summary ===")
    keys_show = [
        ("ael_bulk_modulus_vrh", "B_VRH (GPa)"),
        ("ael_shear_modulus_vrh", "G_VRH (GPa)"),
        ("ael_youngs_modulus_vrh", "Y (GPa)"),
        ("ael_poisson_ratio", "Poisson ratio"),
        ("ael_speed_sound_average", "v_avg (m/s)"),
        ("ael_debye_temperature", "Theta_D AEL (K)"),
        ("agl_debye", "Theta_D AGL (K)"),
        ("agl_acoustic_debye", "acoustic Theta_D (K)"),
        ("agl_gruneisen", "Gruneisen"),
        ("agl_thermal_expansion_300K", "alpha 300K (1/K)"),
        ("agl_bulk_modulus_static_300K", "B_static 300K (GPa)"),
        ("agl_thermal_conductivity_300K", "kappa 300K (W/m/K)"),
    ]
    for k, lab in keys_show:
        v = ael.get(k) or agl.get(k)
        if v is None:
            continue
        print(f"  {lab:24s} {v:12.4f}")

    f, d = read_phonon_dos(C_DIR / "aflow.apl.phonon_dos.out.xz")
    desc = dos_descriptors(f, d)
    print("\n=== C phonon DOS descriptors ===")
    for k in ["min_freq_THz", "first_peak_THz", "mean_freq_THz", "log_avg_freq_THz",
              "dos_frac_below_1THz", "dos_frac_below_2THz",
              "leverage_total", "leverage_below_2THz", "leverage_fraction_below_2THz"]:
        print(f"  {k:36s} {desc[k]:12.5f}")

    csv_path = PROCESSED / "descriptor_table.csv"
    a7_rows = {}
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            a7_rows[r["material"]] = r

    si_desc_path = PROCESSED / "Si_phonon_descriptors.csv"
    si_desc = {}
    if si_desc_path.exists():
        with si_desc_path.open() as fh:
            for row in csv.reader(fh):
                if len(row) >= 2:
                    si_desc[row[0]] = row[1]

    print("\n=== descriptor cross-comparison (A7 vs Si vs C) ===")
    print(f"{'material':10s} {'mean_w (THz)':>14s} {'sub-2 THz':>12s} {'sub-1 THz':>12s} {'log_avg':>10s} {'leverage':>10s}")
    for mat, row in a7_rows.items():
        if row["mean_freq_THz"] == "NA":
            continue
        print(f"{mat:10s} {float(row['mean_freq_THz']):14.3f} "
              f"{float(row['dos_frac_0_2THz']):12.4f} "
              f"{float(row['dos_frac_0_1THz']):12.4f} "
              f"{float(row['log_avg_freq_THz']):10.3f} "
              f"{'-':>10s}")
    if si_desc:
        try:
            print(f"{'Si':10s} {float(si_desc['mean_freq_THz']):14.3f} "
                  f"{float(si_desc['dos_frac_below_2THz']):12.4f} "
                  f"{float(si_desc['dos_frac_below_1THz']):12.4f} "
                  f"{float(si_desc['log_avg_freq_THz']):10.3f} "
                  f"{float(si_desc['leverage_total']):10.4f}")
        except (KeyError, ValueError):
            pass
    print(f"{'C':10s} {desc['mean_freq_THz']:14.3f} "
          f"{desc['dos_frac_below_2THz']:12.4f} "
          f"{desc['dos_frac_below_1THz']:12.4f} "
          f"{desc['log_avg_freq_THz']:10.3f} "
          f"{desc['leverage_total']:10.4f}")

    disp = read_apl_displacements(C_DIR / "aflow.apl.displacements.out.xz")
    print("\n=== C harmonic phonon B_iso(T)  (sample T) ===")
    sample = [0, 80, 100, 295, 300, 500, 700, 1000, 1500]
    for T in sample:
        Bi = float(np.interp(T, disp["T"], disp["B_iso"]))
        ui = float(np.interp(T, disp["T"], disp["U_iso"]))
        print(f"  T = {T:6.1f} K   <u^2> = {ui:.5f} A^2   B = {Bi:.4f} A^2")

    theta_emp = 2230.0  # K -- standard cited diamond Debye temperature
    theta_ael = ael.get("ael_debye_temperature", math.nan)
    theta_agl = agl.get("agl_debye", math.nan)
    Tg = disp["T"]
    B_emp = B_debye(Tg, theta_emp, M_C)
    B_ael = B_debye(Tg, theta_ael, M_C)
    B_agl = B_debye(Tg, theta_agl, M_C)

    print(f"\n=== scalar Debye B(T) (sample T) ===")
    print(f"  Theta_D emp (lit)   = {theta_emp:.0f} K")
    print(f"  Theta_D AEL         = {theta_ael:.2f} K")
    print(f"  Theta_D AGL         = {theta_agl:.2f} K")
    print(f"\n  T (K)   B_emp     B_AEL     B_AGL     B_harm")
    for T in [0, 80, 290, 295, 300, 500, 700, 1000, 1500]:
        Be = float(np.interp(T, Tg, B_emp))
        Ba = float(np.interp(T, Tg, B_ael))
        Bq = float(np.interp(T, Tg, B_agl))
        Bh = float(np.interp(T, Tg, disp["B_iso"]))
        print(f"  {T:5.0f}   {Be:.4f}    {Ba:.4f}    {Bq:.4f}    {Bh:.4f}")

    # literature reference point (see module docstring)
    exp_T = np.array([295.0])
    exp_B = np.array([load_cf8_literature_median("C")])
    exp_src = ["290-300 K literature median"]
    print("\n=== benchmark vs experimental B(T) (literature) ===")
    print(f"  {'T':>5s} {'B_exp':>8s} {'src':>32s}  {'B_emp':>8s} {'B_AEL':>8s} {'B_AGL':>8s} {'B_harm':>8s}")
    rows_bench = []
    for T, Bx, src in zip(exp_T, exp_B, exp_src):
        Be = float(np.interp(T, Tg, B_emp))
        Ba = float(np.interp(T, Tg, B_ael))
        Bq = float(np.interp(T, Tg, B_agl))
        Bh = float(np.interp(T, Tg, disp["B_iso"]))
        print(f"  {T:5.0f} {Bx:8.4f} {src:>32s}  {Be:8.4f} {Ba:8.4f} {Bq:8.4f} {Bh:8.4f}")
        rows_bench.append((T, Bx, src, Be, Ba, Bq, Bh))

    # fractional deviation at the reference point
    print("\n=== Fractional deviation (B_model / B_exp - 1) at each T ===")
    print(f"  {'T':>5s} {'B_exp':>8s}  {'emp':>8s} {'AEL':>8s} {'AGL':>8s} {'harm':>8s}")
    for r in rows_bench:
        T, Bx, _src, Be, Ba, Bq, Bh = r
        print(f"  {T:5.0f} {Bx:8.4f}  {Be/Bx-1:+8.2%} {Ba/Bx-1:+8.2%} "
              f"{Bq/Bx-1:+8.2%} {Bh/Bx-1:+8.2%}")

    # write CSVs
    with DESCRIPTORS_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["descriptor", "value"])
        for k, v in desc.items():
            w.writerow([k, f"{v:.6g}"])
        w.writerow(["theta_D_empirical_lit_K", f"{theta_emp:.0f}"])
        for k in ["ael_debye_temperature", "agl_debye", "ael_bulk_modulus_vrh",
                  "ael_shear_modulus_vrh", "ael_poisson_ratio",
                  "agl_gruneisen", "agl_thermal_conductivity_300K",
                  "agl_thermal_expansion_300K"]:
            v = ael.get(k) or agl.get(k)
            if v is not None:
                w.writerow([k, f"{v:.6g}"])

    with CURVES_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["T_K", "B_iso_harmonic_A2", f"B_emp_{theta_emp:.0f}K_A2",
                    "B_AEL_A2", "B_AGL_A2"])
        for i, T in enumerate(Tg):
            w.writerow([f"{T:.1f}", f"{disp['B_iso'][i]:.6f}",
                        f"{B_emp[i]:.6f}", f"{B_ael[i]:.6f}", f"{B_agl[i]:.6f}"])

    with BENCHMARK_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["T_K", "B_exp_A2", "source", f"B_emp_{theta_emp:.0f}K_A2",
                    "B_AEL_A2", "B_AGL_A2", "B_harm_A2"])
        for r in rows_bench:
            w.writerow([f"{r[0]:.1f}", f"{r[1]:.4f}", r[2],
                        f"{r[3]:.4f}", f"{r[4]:.4f}", f"{r[5]:.4f}", f"{r[6]:.4f}"])

    print(f"\nWrote {DESCRIPTORS_CSV}, {CURVES_CSV}, {BENCHMARK_CSV}")


if __name__ == "__main__":
    main()
