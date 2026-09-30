#!/usr/bin/env python3
"""Analyze the new Bi SOC APL phonon calculation.

Reuses the same harmonic-displacement convention used in the manuscript
(`B = 8 pi^2 <u^2>`, with site-mean <u^2> taken from
`aflow.apl.displacements.out`). Compares against:
  - Fischer 1978 isotropic-equivalent B(T) for Bi
  - Bi noSOC harmonic phonon B(T) (csv/Bi_comparison_main_curves_wide.csv)
  - Bi SOC scalar Debye curves (csv/Bi_SOC_relativistic_sensitivity_main_curves_wide.csv)
  - Bi noSOC empirical Debye reference

Also recomputes the descriptor pair used in the regime guide:
  - mean phonon frequency
  - sub-2 THz DOS fraction
  - omega^-2 leverage proxy below 2 THz (omega >= 0.10 THz cutoff)
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


SOC_DIR = calc_dir("Bi_SOC")
NOSOC_DIR = calc_dir("Bi")
CSV_DIR = PROCESSED
OUT_DIR = CSV_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
DESCRIPTORS_CSV = OUT_DIR / "Bi_SOC_phonon_descriptors.csv"
BENCHMARK_CSV = OUT_DIR / "Bi_SOC_phonon_benchmark.csv"
CURVES_CSV = OUT_DIR / "Bi_SOC_phonon_harmonic_curves.csv"

EIGHT_PI_SQ = 8.0 * math.pi**2


def open_text(path: Path):
    if path.suffix == ".xz":
        return lzma.open(path, "rt", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def read_phonon_dos(path: Path) -> tuple[np.ndarray, np.ndarray]:
    freqs, dos = [], []
    with open_text(path) as fh:
        for line in fh:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                freqs.append(float(parts[0]))
                dos.append(float(parts[3]))
            except ValueError:
                continue
    return np.asarray(freqs), np.asarray(dos)


def read_apl_displacements(path: Path) -> dict[str, np.ndarray]:
    rows: dict[float, list[tuple[float, float, float]]] = {}
    current_T = None
    with open_text(path) as fh:
        for line in fh:
            line = line.rstrip()
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("["):
                continue
            parts = stripped.split()
            if len(parts) == 5:
                try:
                    current_T = float(parts[0])
                    x = float(parts[2]); y = float(parts[3]); z = float(parts[4])
                except ValueError:
                    continue
            elif len(parts) == 4 and current_T is not None:
                try:
                    x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                except ValueError:
                    continue
            else:
                continue
            rows.setdefault(current_T, []).append((x, y, z))
    Ts = np.asarray(sorted(rows))
    ux = np.asarray([np.mean([r[0] for r in rows[T]]) for T in Ts])
    uy = np.asarray([np.mean([r[1] for r in rows[T]]) for T in Ts])
    uz = np.asarray([np.mean([r[2] for r in rows[T]]) for T in Ts])
    u_basal = 0.5 * (ux + uy)
    u_iso = (ux + uy + uz) / 3.0
    return {
        "T": Ts,
        "B_basal": EIGHT_PI_SQ * u_basal,
        "B_c": EIGHT_PI_SQ * uz,
        "B_iso": EIGHT_PI_SQ * u_iso,
    }


def dos_descriptors(freqs: np.ndarray, dos: np.ndarray) -> dict[str, float]:
    pos = freqs > 0
    f = freqs[pos]; d = dos[pos]
    norm = _trapz(d, f)
    d_n = d / norm
    mean_w = float(_trapz(f * d_n, f))
    log_w = float(math.exp(_trapz(np.log(f) * d_n, f)))

    def frac_below(w_max):
        mask = f < w_max
        return float(_trapz(d_n[mask], f[mask]))

    frac_1 = frac_below(1.0)
    frac_2 = frac_below(2.0)

    cutoff = 0.10
    mask_lev = (f >= cutoff)
    leverage_total = float(_trapz(d_n[mask_lev] / f[mask_lev] ** 2, f[mask_lev]))
    mask_lev_2 = (f >= cutoff) & (f < 2.0)
    leverage_below2 = float(_trapz(d_n[mask_lev_2] / f[mask_lev_2] ** 2, f[mask_lev_2]))
    return {
        "mean_freq_THz": mean_w,
        "log_avg_freq_THz": log_w,
        "dos_frac_below_1THz": frac_1,
        "dos_frac_below_2THz": frac_2,
        "leverage_total": leverage_total,
        "leverage_below_2THz": leverage_below2,
        "leverage_fraction_below_2THz": leverage_below2 / leverage_total if leverage_total else float("nan"),
    }


def imaginary_summary(freqs: np.ndarray, dos: np.ndarray) -> dict[str, float]:
    neg = freqs < -1e-6
    if not neg.any():
        return {
            "min_freq_THz": float(freqs.min()),
            "neg_dos_weight": 0.0,
            "neg_min_freq_THz": 0.0,
        }
    f_neg = freqs[neg]; d_neg = dos[neg]
    return {
        "min_freq_THz": float(freqs.min()),
        "neg_dos_weight": float(_trapz(d_neg, f_neg)),
        "neg_min_freq_THz": float(f_neg.min()),
    }


def read_experimental_points() -> tuple[np.ndarray, np.ndarray]:
    Ts = []; Bs = []
    with (CSV_DIR / "Bi_comparison_experiment_points.csv").open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            Ts.append(float(row["T_K"]))
            Bs.append(float(row["B_exp_A2"]))
    return np.asarray(Ts), np.asarray(Bs)


def benchmark(model_T: np.ndarray, model_B: np.ndarray,
              exp_T: np.ndarray, exp_B: np.ndarray) -> dict[str, float]:
    interp = np.interp(exp_T, model_T, model_B)
    res = interp - exp_B
    rmse = float(np.sqrt(np.mean(res ** 2)))
    mae = float(np.mean(np.abs(res)))
    exp_range = float(np.max(exp_B) - np.min(exp_B))
    nrmse = rmse / exp_range if exp_range > 0.0 else float("nan")
    ss_res = float(np.sum(res ** 2))
    ss_tot = float(np.sum((exp_B - np.mean(exp_B)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return {"N": int(exp_T.size), "RMSE": rmse, "MAE": mae, "NRMSE": nrmse, "R2": r2}


def main() -> None:
    print("=== Bi SOC APL: directory check ===")
    end = SOC_DIR / "aflow.end.out"
    print(f"  aflow.end.out present: {end.exists()}")
    if end.exists():
        print(f"  status line: {end.read_text().splitlines()[0][:120]}")

    # ---- DOS / descriptors ----
    soc_freq, soc_dos = read_phonon_dos(SOC_DIR / "aflow.apl.phonon_dos.out.xz")
    nosoc_freq, nosoc_dos = read_phonon_dos(NOSOC_DIR / "aflow.apl.phonon_dos.out.xz")
    soc_desc = dos_descriptors(soc_freq, soc_dos)
    nosoc_desc = dos_descriptors(nosoc_freq, nosoc_dos)
    soc_imag = imaginary_summary(soc_freq, soc_dos)
    nosoc_imag = imaginary_summary(nosoc_freq, nosoc_dos)

    print("\n=== Bi phonon DOS descriptors ===")
    keys = [
        "mean_freq_THz", "log_avg_freq_THz",
        "dos_frac_below_1THz", "dos_frac_below_2THz",
        "leverage_total", "leverage_below_2THz", "leverage_fraction_below_2THz",
    ]
    print(f"{'descriptor':40s} {'noSOC':>14s} {'SOC':>14s} {'delta %':>10s}")
    for k in keys:
        v0 = nosoc_desc[k]; v1 = soc_desc[k]
        d = (v1 - v0) / v0 * 100.0 if v0 else float("nan")
        print(f"{k:40s} {v0:14.5f} {v1:14.5f} {d:9.2f}")
    print("\n=== imaginary-mode summary ===")
    print(f"  noSOC min_freq = {nosoc_imag['min_freq_THz']:+.4f} THz, neg_weight = {nosoc_imag['neg_dos_weight']:.4e}")
    print(f"  SOC   min_freq = {soc_imag['min_freq_THz']:+.4f} THz, neg_weight = {soc_imag['neg_dos_weight']:.4e}")

    # ---- B_iso(T) ----
    soc_disp = read_apl_displacements(SOC_DIR / "aflow.apl.displacements.out.xz")
    nosoc_disp = read_apl_displacements(NOSOC_DIR / "aflow.apl.displacements.out.xz")

    exp_T, exp_B = read_experimental_points()
    bench_soc = benchmark(soc_disp["T"], soc_disp["B_iso"], exp_T, exp_B)
    bench_nosoc = benchmark(nosoc_disp["T"], nosoc_disp["B_iso"], exp_T, exp_B)
    # The basal and c-axis components are not scored: Fischer's B is isotropic.

    # SOC scalar Debye for context (read from csv)
    soc_curves = np.genfromtxt(
        CSV_DIR / "Bi_SOC_relativistic_sensitivity_main_curves_wide.csv",
        delimiter=",", names=True, dtype=float, encoding="utf-8",
    )
    soc_T = soc_curves["T_K"]
    soc_eD = soc_curves["Bi_SOC_B_elastic_derived_debye_A2"]
    soc_qhD = soc_curves["Bi_SOC_B_quasi_harmonic_debye_A2"]
    bench_soc_eD = benchmark(soc_T, soc_eD, exp_T, exp_B)
    bench_soc_qhD = benchmark(soc_T, soc_qhD, exp_T, exp_B)

    print("\n=== Bi B_iso(T) benchmark vs Fischer 1978 (N = {} points) ===".format(bench_soc["N"]))
    rows = [
        ("harmonic phonon (isotropic) noSOC", bench_nosoc),
        ("harmonic phonon (isotropic) SOC ", bench_soc),
        ("scalar elastic Debye SOC         ", bench_soc_eD),
        ("scalar quasi-harmonic Debye SOC  ", bench_soc_qhD),
    ]
    print(f"{'method':36s} {'RMSE':>8s} {'NRMSE':>8s} {'R2':>8s}")
    for name, b in rows:
        print(f"{name:36s} {b['RMSE']:8.3f} {b['NRMSE']:8.3f} {b['R2']:8.3f}")

    # ---- T-resolved residuals at experimental points ----
    print("\n=== T-resolved residuals (model - exp) at Fischer T points ===")
    h_T = soc_disp["T"]; h_B = soc_disp["B_iso"]
    h_B_int = np.interp(exp_T, h_T, h_B)
    h_B_int_nosoc = np.interp(exp_T, nosoc_disp["T"], nosoc_disp["B_iso"])
    print(f"{'T (K)':>8s} {'B_exp':>8s} {'B_noSOC':>8s} {'B_SOC':>8s} {'r_noSOC':>9s} {'r_SOC':>9s}")
    for T, Bx, b0, b1 in zip(exp_T, exp_B, h_B_int_nosoc, h_B_int):
        print(f"{T:8.1f} {Bx:8.3f} {b0:8.3f} {b1:8.3f} {b0 - Bx:+9.3f} {b1 - Bx:+9.3f}")

    # ---- write tidy csv outputs ----
    desc_csv = DESCRIPTORS_CSV
    with desc_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["descriptor", "noSOC", "SOC", "delta_pct"])
        for k in keys:
            v0 = nosoc_desc[k]; v1 = soc_desc[k]
            w.writerow([k, f"{v0:.6g}", f"{v1:.6g}", f"{(v1 - v0) / v0 * 100:.4f}" if v0 else ""])
        w.writerow(["min_freq_THz", f"{nosoc_imag['min_freq_THz']:+.6f}",
                    f"{soc_imag['min_freq_THz']:+.6f}", ""])
        w.writerow(["neg_dos_weight", f"{nosoc_imag['neg_dos_weight']:.6e}",
                    f"{soc_imag['neg_dos_weight']:.6e}", ""])

    bench_csv = BENCHMARK_CSV
    with bench_csv.open("w", newline="") as fh:
        # Same NRMSE as evaluate_model() in rebuild_dwf_revision_package.py.
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["# NRMSE_definition: range  (RMSE / (max(B_exp) - min(B_exp)))"])
        w.writerow(["method", "soc_status", "N", "RMSE_A2", "MAE_A2", "NRMSE", "R2"])
        for name, status, b in [
            ("harmonic phonon (isotropic)", "noSOC", bench_nosoc),
            ("harmonic phonon (isotropic)", "SOC", bench_soc),
            ("elastic-derived Debye", "SOC", bench_soc_eD),
            ("quasi-harmonic Debye", "SOC", bench_soc_qhD),
        ]:
            w.writerow([name, status, b["N"], f"{b['RMSE']:.5f}", f"{b['MAE']:.5f}",
                        f"{b['NRMSE']:.5f}", f"{b['R2']:.5f}"])

    curve_csv = CURVES_CSV
    with curve_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["T_K", "B_basal_SOC_A2", "B_c_axis_SOC_A2", "B_iso_SOC_A2",
                    "B_iso_noSOC_A2"])
        T_grid = soc_disp["T"]
        b0_iso = np.interp(T_grid, nosoc_disp["T"], nosoc_disp["B_iso"])
        for i, T in enumerate(T_grid):
            w.writerow([f"{T:.1f}", f"{soc_disp['B_basal'][i]:.6f}",
                        f"{soc_disp['B_c'][i]:.6f}", f"{soc_disp['B_iso'][i]:.6f}",
                        f"{b0_iso[i]:.6f}"])

    print(f"\nWrote {desc_csv}, {bench_csv}, {curve_csv}")


if __name__ == "__main__":
    main()
