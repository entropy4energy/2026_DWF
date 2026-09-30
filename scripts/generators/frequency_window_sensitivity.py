#!/usr/bin/env python3
"""Frequency-window sensitivity of the low-frequency phonon descriptors.

Answers the question "why 2 THz?" by recomputing, for every material and for
window edges W = 1, 2, 3, 4 THz, the two descriptors that the manuscript uses
to separate soft from stiff spectra:

  1. DOS fraction below W
         F_DOS(W) = int_0^W g(f) df / int_0^fmax g(f) df
     with g normalized on the tabulated positive-frequency range.

  2. Inverse-square weighted ("displacement leverage") fraction below W
         F_inv2(W) = int_fmin^W g/f^2 df / int_fmin^fmax g/f^2 df
     which is the fraction of the classical (high-T) harmonic displacement
     integral B_iso ~ <f^-2> that comes from below W. The acoustic endpoint
     f_min is needed because g/f^2 diverges as f -> 0; f_min = 0.10 THz is the
     value already used elsewhere in this project, and its effect is bounded by
     the f_min sweep reported in Table tab:theta_dwf_cutoff.

  3. As a check that (2) is not an artifact of the classical limit, the same
     fraction with the full quantum displacement weight at 295 K
         w(f) = (1/f) * coth(h f / 2 k_B T),
     which is what actually enters B_iso(295 K). w -> 2 k_B T / (h f^2) as
     f -> 0, so it carries the same acoustic endpoint and the same f_min.

Reads only the archived APL phonon DOS files (aflow.apl.phonon_dos.out.xz).
No new DFT is performed. Conventions (DOS column, normalization, f_min = 0.10
THz) follow scripts/generators/Si_phonon_analysis.py so that the W = 2 THz
column reproduces the sub-2 THz fractions already quoted in the manuscript;
the script asserts this against hardcoded reference values and prints the
residuals.

Outputs
  data/processed/frequency_window_sensitivity.csv        tidy, one row per (material, W)
  data/processed/frequency_window_sensitivity_table.tex  SI-ready LaTeX table

Usage
  python3 frequency_window_sensitivity.py [--fmin 0.10] [--windows 1 2 3 4]
                                          [--temperature 295]
"""
from __future__ import annotations

import argparse
import csv
import lzma
import math
import sys
from pathlib import Path

import numpy as np

# numpy 2.0 renamed trapz -> trapezoid; alias keeps old and new numpy working.
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# Every path resolves through scripts/paths.py, the single source of truth for
# this package. Nothing outside the repository root is read or written.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DFT_RAW, PROCESSED, calc_dir  # noqa: E402

H_SI = 6.62607015e-34
KB_SI = 1.380649e-23
THZ = 1.0e12

# material -> (family label, LaTeX label, soc flag, data/dft_raw/ subdirectory)
MATERIALS = [
    ("As",     "A7",   "As",            "noSOC", "As"),
    ("Sb",     "A7",   "Sb",            "noSOC", "Sb"),
    ("Bi",     "A7",   "Bi",            "noSOC", "Bi"),
    ("Bi_SOC", "A7",   r"Bi~(\SOC)",    "SOC",   "Bi_SOC"),
    ("C",      "cF8",  "C",             "noSOC", "C"),
    ("Si",     "cF8",  "Si",            "noSOC", "Si"),
    ("Ge",     "cF8",  "Ge",            "noSOC", "Ge"),
]

# Published sub-2 THz DOS fractions (descriptor_table.csv / Table
# tab:cF8_descriptors) used as a regression check on the W = 2 THz column.
REFERENCE_DOS_FRAC_2THZ = {
    "As": 0.10139152,
    "Sb": 0.36186793,
    "Bi": 0.50107951,
    "Bi_SOC": 0.50136,
    "C": 0.00005,
    "Si": 0.00582,
    "Ge": 0.07880,
}
# Published inverse-square weighted fraction below 2 THz
# (cross_material_descriptors_table.csv, leverage_fraction_below_2THz).
# As/Sb/Bi were never computed there and are filled in by this script.
REFERENCE_INV2_FRAC_2THZ = {
    "Bi_SOC": 0.91147,
    "C": 0.02032,
    "Si": 0.13582,
    "Ge": 0.41351,
}


def _resolve_calc_dir(material: str) -> Path:
    """Return the raw-calculation directory for ``material``.

    The original AFLOW trees used four different nesting patterns; they are
    flattened to one directory per material under data/dft_raw/, so no layout
    probing is needed. See scripts/paths.py.
    """
    return calc_dir(material)


def open_text(path: Path):
    if path.suffix == ".xz":
        return lzma.open(path, "rt", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def read_phonon_dos(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (f in THz, pDOS) from an aflow.apl.phonon_dos.out file."""
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


def integrate(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    """Trapezoidal integral of y(x) from lo to hi, interpolating at both ends.

    Bounds are clamped to the tabulated range, so the (untabulated) sliver
    below the first grid point is excluded from numerator and denominator
    alike. Interpolating the endpoints instead of snapping to the nearest grid
    point makes the window fractions independent of the DOS grid spacing.
    """
    lo = max(lo, float(x[0]))
    hi = min(hi, float(x[-1]))
    if hi <= lo:
        return 0.0
    inner = (x > lo) & (x < hi)
    xs = np.concatenate(([lo], x[inner], [hi]))
    ys = np.concatenate(([float(np.interp(lo, x, y))], y[inner],
                         [float(np.interp(hi, x, y))]))
    return float(_trapz(ys, xs))


def legacy_frac_below(f: np.ndarray, g: np.ndarray, W: float) -> float:
    """Grid-snapped fraction, reproducing the convention in the older scripts."""
    m = f < W
    if m.sum() < 2:
        return 0.0
    return float(_trapz(g[m], f[m]) / _trapz(g, f))


def bose_weight(f: np.ndarray, T: float) -> np.ndarray:
    """Quantum harmonic displacement weight (1/f) coth(h f / 2 k_B T)."""
    x = H_SI * f * THZ / (2.0 * KB_SI * T)
    return (1.0 / f) / np.tanh(x)


def cumulative_percentile(f: np.ndarray, g: np.ndarray, q: float) -> float:
    """Frequency below which a fraction q of the normalized DOS weight lies."""
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(f))))
    cum /= cum[-1]
    return float(np.interp(q, cum, f))


def spectral_gap(f: np.ndarray, g: np.ndarray, rel_thresh: float = 0.005,
                 f_lo: float = 0.3) -> tuple[float, float] | None:
    """Widest interior interval where the DOS falls below rel_thresh * peak.

    For the A7 materials this is the acoustic-optic gap. The diamond-structure
    materials have overlapping acoustic and optic manifolds and return None.
    Only gaps above f_lo and below the highest nonzero frequency count, so the
    acoustic Gamma endpoint and the empty tail above the spectrum are excluded.
    """
    f_top = float(f[g > 0].max()) if np.any(g > 0) else float(f[-1])
    below = g < rel_thresh * g.max()
    best: tuple[float, float] | None = None
    best_w = 0.0
    i = 0
    while i < len(f):
        if not below[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(f) and below[j + 1]:
            j += 1
        lo, hi = float(f[i]), float(f[j])
        if lo > f_lo and hi < f_top and (hi - lo) > best_w:
            best_w, best = hi - lo, (lo, hi)
        i = j + 1
    return best


def analyze(f_raw: np.ndarray, d_raw: np.ndarray, windows: list[float],
            f_min: float, T: float) -> dict:
    pos = f_raw > 0
    f, d = f_raw[pos], d_raw[pos]
    g = d / _trapz(d, f)                      # normalized on the tabulated range

    w_inv2 = g / f**2                         # classical high-T displacement weight
    w_bose = g * bose_weight(f, T)            # quantum displacement weight at T

    dos_tot = integrate(f, g, f[0], f[-1])
    inv2_tot = integrate(f, w_inv2, f_min, f[-1])
    bose_tot = integrate(f, w_bose, f_min, f[-1])

    gap = spectral_gap(f, g)
    out = {
        "f_min_tab_THz": float(f[0]),
        "f_max_THz": float(f[-1]),
        "f_top_THz": float(f[d > 0].max()) if np.any(d > 0) else float(f[-1]),
        "mean_f_THz": float(_trapz(f * g, f)),
        "f_median_THz": cumulative_percentile(f, g, 0.5),
        "gap_lo_THz": gap[0] if gap else math.nan,
        "gap_hi_THz": gap[1] if gap else math.nan,
        "inv2_total": inv2_tot,
        "windows": {},
    }
    for W in windows:
        out["windows"][W] = {
            "dos_frac": integrate(f, g, f[0], W) / dos_tot,
            "dos_frac_legacy": legacy_frac_below(f, g, W),
            "inv2_frac": integrate(f, w_inv2, f_min, W) / inv2_tot if inv2_tot else math.nan,
            "bose_frac": integrate(f, w_bose, f_min, W) / bose_tot if bose_tot else math.nan,
        }
    return out


def fmt(v: float, sig: int = 3) -> str:
    """Compact fixed/scientific formatting for table cells."""
    if not np.isfinite(v):
        return "NA"
    if v == 0:
        return "0"
    if v < 1.0e-3:
        m, e = f"{v:.1e}".split("e")
        return f"{m}e{int(e)}"
    return f"{v:.{sig}f}"


def tex_cell(v: float) -> str:
    if not np.isfinite(v):
        return "NA"
    if v < 1.0e-3:
        m, e = f"{v:.1e}".split("e")
        return rf"${m}\times10^{{{int(e)}}}$"
    return f"{v:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fmin", type=float, default=0.10,
                    help="acoustic endpoint for the inverse-square integrals (THz)")
    ap.add_argument("--windows", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0],
                    help="window upper edges (THz)")
    ap.add_argument("--temperature", type=float, default=295.0,
                    help="temperature for the quantum-weighted fraction (K)")
    ap.add_argument("--outdir", type=Path, default=PROCESSED)
    args = ap.parse_args()

    windows = list(args.windows)
    args.outdir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for key, family, tex_label, soc, rel in MATERIALS:
        path = _resolve_calc_dir(rel) / "aflow.apl.phonon_dos.out.xz"
        if not path.exists():
            print(f"  !! missing DOS for {key}: {path}")
            continue
        f, d = read_phonon_dos(path)
        results[key] = analyze(f, d, windows, args.fmin, args.temperature)
        results[key].update(family=family, tex_label=tex_label, soc=soc,
                            n_points=len(f))

    # ---- regression check against the numbers already in the manuscript ----
    print("=== consistency check at W = 2 THz (new vs published) ===")
    print(f"  {'material':10s} {'DOS frac new':>13s} {'published':>11s} {'diff':>10s}"
          f"   {'inv2 frac new':>14s} {'published':>11s} {'diff':>10s}")
    worst = 0.0
    for key in results:
        w2 = results[key]["windows"].get(2.0)
        if w2 is None:
            continue
        ref_d = REFERENCE_DOS_FRAC_2THZ.get(key)
        ref_i = REFERENCE_INV2_FRAC_2THZ.get(key)
        dd = abs(w2["dos_frac"] - ref_d) if ref_d is not None else float("nan")
        di = abs(w2["inv2_frac"] - ref_i) if ref_i is not None else float("nan")
        for v in (dd, di):
            if np.isfinite(v):
                worst = max(worst, v)
        print(f"  {key:10s} {w2['dos_frac']:13.6f} "
              f"{(f'{ref_d:.6f}' if ref_d is not None else '--'):>11s} "
              f"{(f'{dd:.2e}' if np.isfinite(dd) else '--'):>10s}   "
              f"{w2['inv2_frac']:14.6f} "
              f"{(f'{ref_i:.6f}' if ref_i is not None else 'new'):>11s} "
              f"{(f'{di:.2e}' if np.isfinite(di) else '--'):>10s}")
    print(f"  worst absolute deviation from published values: {worst:.2e}"
          "  (published C/Si/Ge DOS fractions are quoted to 2 significant"
          " figures, so ~1e-5 is expected)")

    # ---- main table ----
    print(f"\n=== frequency-window sensitivity "
          f"(f_min = {args.fmin:.2f} THz, T = {args.temperature:.0f} K) ===")
    head = f"  {'material':10s} {'f_max':>7s} "
    head += " ".join(f"{'F_DOS<' + str(int(W)):>10s}" for W in windows) + "  "
    head += " ".join(f"{'F_inv2<' + str(int(W)):>10s}" for W in windows)
    print(head)
    for key, res in results.items():
        row = f"  {key:10s} {res['f_max_THz']:7.2f} "
        row += " ".join(f"{fmt(res['windows'][W]['dos_frac']):>10s}" for W in windows) + "  "
        row += " ".join(f"{fmt(res['windows'][W]['inv2_frac']):>10s}" for W in windows)
        print(row)

    print(f"\n=== quantum-weighted check: displacement fraction below W at "
          f"{args.temperature:.0f} K ===")
    print(f"  {'material':10s} " + " ".join(f"{'F_B<' + str(int(W)):>10s}" for W in windows)
          + "     (classical F_inv2 in parentheses below)")
    for key, res in results.items():
        row = f"  {key:10s} " + " ".join(
            f"{fmt(res['windows'][W]['bose_frac']):>10s}" for W in windows)
        row += "     (" + ", ".join(
            fmt(res['windows'][W]['inv2_frac']) for W in windows) + ")"
        print(row)

    # ---- is the soft-to-stiff ranking window-independent? ----
    print("\n=== ranking by each descriptor at each window (stiff -> soft) ===")
    for col, name in [("dos_frac", "F_DOS "), ("inv2_frac", "F_inv2")]:
        orders = {}
        for W in windows:
            order = sorted(results, key=lambda k: results[k]["windows"][W][col])
            orders[W] = order
            print(f"  {name} W={W:g} THz: " + " < ".join(order))
        ref = orders[windows[0]]
        changed = [W for W in windows if orders[W] != ref]
        if not changed:
            print(f"  {name} ranking is identical at all {len(windows)} windows.")
        else:
            for W in changed:
                swaps = [(a, b) for a, b in zip(ref, orders[W]) if a != b]
                pairs = sorted({tuple(sorted(p)) for p in swaps})
                detail = "; ".join(
                    f"{a}/{b} differ by "
                    f"{abs(results[a]['windows'][W][col] - results[b]['windows'][W][col]):.4f}"
                    for a, b in pairs)
                print(f"  {name} ranking changes at W={W:g} THz: {detail}")

    # ---- where the window edges sit in the spectrum ----
    print("\n=== spectrum landmarks (why a given edge is or is not informative) ===")
    print(f"  {'material':10s} {'f_top':>7s} {'f_median':>9s} {'DOS gap (THz)':>16s}"
          "   window edges inside the gap")
    for key, res in results.items():
        lo, hi = res["gap_lo_THz"], res["gap_hi_THz"]
        if np.isfinite(lo):
            inside = [f"{W:g}" for W in windows if lo <= W <= hi]
            gap_s = f"{lo:.2f}-{hi:.2f}"
        else:
            inside, gap_s = [], "none (manifolds overlap)"
        print(f"  {key:10s} {res['f_top_THz']:7.2f} {res['f_median_THz']:9.2f} "
              f"{gap_s:>16s}   {', '.join(inside) if inside else '--'}")
    print("  f_top is the highest frequency with nonzero DOS; f_median is the 50%"
          " cumulative-weight frequency.\n  A gap column entry means the DOS stays"
          " below 0.5% of its peak across that interval.")

    # ---- grid-convention check ----
    print("\n=== endpoint convention: interpolated vs grid-snapped DOS fraction ===")
    max_gap = 0.0
    for key, res in results.items():
        gaps = [abs(res["windows"][W]["dos_frac"] - res["windows"][W]["dos_frac_legacy"])
                for W in windows]
        max_gap = max(max_gap, max(gaps))
        print(f"  {key:10s} max |interp - snapped| = {max(gaps):.2e}")
    print(f"  worst case over all materials and windows: {max_gap:.2e}")

    # ---- CSV ----
    csv_path = args.outdir / "frequency_window_sensitivity.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["material", "family", "soc_status", "window_THz",
                    "dos_frac_below_W", "invsq_frac_below_W",
                    "quantum_disp_frac_below_W", "temperature_K",
                    "f_min_invsq_THz", "f_max_THz", "f_top_THz", "mean_f_THz",
                    "f_median_THz", "dos_gap_lo_THz", "dos_gap_hi_THz",
                    "invsq_total_THz_minus2", "dos_frac_below_W_grid_snapped",
                    "n_dos_points"])
        for key, res in results.items():
            for W in windows:
                d = res["windows"][W]
                w.writerow([key, res["family"], res["soc"], f"{W:g}",
                            f"{d['dos_frac']:.6g}", f"{d['inv2_frac']:.6g}",
                            f"{d['bose_frac']:.6g}", f"{args.temperature:g}",
                            f"{args.fmin:g}", f"{res['f_max_THz']:.4f}",
                            f"{res['f_top_THz']:.4f}",
                            f"{res['mean_f_THz']:.5f}",
                            f"{res['f_median_THz']:.4f}",
                            "NA" if not np.isfinite(res["gap_lo_THz"]) else f"{res['gap_lo_THz']:.4f}",
                            "NA" if not np.isfinite(res["gap_hi_THz"]) else f"{res['gap_hi_THz']:.4f}",
                            f"{res['inv2_total']:.6g}",
                            f"{d['dos_frac_legacy']:.6g}", res["n_points"]])

    # ---- LaTeX fragment ----
    tex_path = args.outdir / "frequency_window_sensitivity_table.tex"
    wcols = "c" * len(windows)
    nw = len(windows)
    span = r"\multicolumn{%d}{c}{$W$ (THz)}" % nw
    lines = [
        r"\begin{table}[h!]",
        r"\vspace{0.25cm}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{\textbf{Sensitivity of the low-frequency descriptors to the choice",
        r"of frequency window.}",
        r"$F_{\mathrm{DOS}}$ is the phonon",
        r"\underline{d}ensity-\underline{o}f-\underline{s}tates fraction below the",
        r"window edge $W$, and $F_{\mathrm{inv}}$ is the fraction of the",
        r"inverse-square moment that sets the classical harmonic displacement,",
        rf"evaluated with an acoustic endpoint $f_{{\min}}={args.fmin:.2f}$~THz",
        r"(\Eq~\ref{eq:window_fractions};",
        r"\Table~\ref{tab:theta_dwf_cutoff} bounds the effect of that endpoint).",
        r"The $W=2$~THz columns are the sub-2~THz fractions quoted in the main text.",
        r"Entries of $1.000$ for Bi are exact: its spectrum ends below 4~THz.}",
        r"%.",
        r"\label{tab:freq_window_sensitivity}",
        r"\vspace{-0.25cm}",
        rf"\begin{{tabular}}{{ll{wcols}{wcols}}}",
        r"\hline",
        rf"Family & Material & \multicolumn{{{nw}}}{{c}}{{$F_{{\mathrm{{DOS}}}}(W)$}} &",
        rf"\multicolumn{{{nw}}}{{c}}{{$F_{{\mathrm{{inv}}}}(W)$}} \\",
        rf"\cline{{3-{2 + nw}}} \cline{{{3 + nw}-{2 + 2 * nw}}}",
        r" & & " + span + " & " + span + r" \\",
        r" & & " + " & ".join(f"{W:g}" for W in windows) + " & "
        + " & ".join(f"{W:g}" for W in windows) + r" \\",
        r"%.",
        r"\hline",
    ]
    prev_family = None
    for key, res in results.items():
        if prev_family is not None and res["family"] != prev_family:
            lines.append(r"\hline")
        prev_family = res["family"]
        fam = res["family"] if res["family"] == "A7" else f"${res['family']}$"
        cells = [tex_cell(res["windows"][W]["dos_frac"]) for W in windows]
        cells += [tex_cell(res["windows"][W]["inv2_frac"]) for W in windows]
        lines.append(f"{fam} & {res['tex_label']} & " + " & ".join(cells) + r" \\")
        lines.append(r"%.")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    tex_path.write_text("\n".join(lines) + "\n")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
