"""Convergence of the moment integration with respect to the phonon-DOS frequency mesh.

Backs SI Table S5 and the numerical bound quoted in the Methods. Post-processing
only -- no new first-principles calculations. Reads the archived APL phonon DOS
(written with DOSPOINTS=2000 uniform bins, DOSMESH=21x21x21, DOSMETHOD=LT) and
re-integrates it on coarser frequency meshes and in the exact fine-mesh limit.

Scope. Two ingredients are held fixed and are *not* tested here: the 21x21x21
q-mesh and the linear-tetrahedron DOS construction, both internal to AFLOW.
What is bounded is the frequency-quadrature error of the moment integrals that
this work layers on top of the archived spectra.

Coarser meshes use weight-conserving rebinning: the tabulated DOS is treated as
a piecewise-linear function, integrated exactly over each coarse bin, and
divided by the coarse bin width. This preserves int g df and reproduces the
bin-centre convention of an AFLOW run with a smaller DOSPOINTS, so a coarse
level is what the pipeline would have seen had the DOS been written at that
resolution. Because a rebinned level is a bin *average* sampled at bin centres,
its quadrature error is midpoint-like and carries the opposite sign to the
trapezoidal error of the archived point values; convergence across levels is
therefore not required to be monotonic. The bound is the spread over all
levels, which the table reports directly.

The fine-mesh limit is not an interpolation -- refining a piecewise-linear g
adds no information. It is the exact integral of that piecewise-linear g:
  <f^-2>: closed form per interval, a (1/f0 - 1/f1) + b ln(f1/f0)
          for g(f) = a + b f on [f0, f1];
  B_iso weight f^-1 coth(h f / 2 kB T): 32-point Gauss-Legendre per interval.
Both are verified at run time against trapezoidal subdivision refinement of the
same piecewise-linear function (--selftest, on by default).

What each column means. Theta_DWF is computed by this work from the tabulated
DOS, so the mesh study bounds its numerical error directly. The published
harmonic B_iso(T) curves instead come from AFLOW's own mode-resolved
mean-square displacements (aflow.apl.displacements.out), which never touch the
tabulated frequency grid; the B_iso reported here is the DOS-route
reconstruction of that quantity. Its offset from the published value is the
DOS-versus-mode-resolved difference, not a quadrature error, and the script
prints both so the two are never conflated.
"""
import argparse
import csv
import lzma
import math
import sys
from pathlib import Path

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz

# Every path resolves through scripts/paths.py, the single source of truth for
# this package. Nothing outside the repository root is read or written.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DFT_RAW, PROCESSED  # noqa: E402

H_SI = 6.62607015e-34        # J s
HBAR_SI = 1.054571817e-34    # J s
KB_SI = 1.380649e-23         # J / K
AMU = 1.66053906660e-27      # kg
THZ = 1e12                   # s^-1 per THz

# label, family, tex label, SOC status, atomic mass (amu), dft_raw/ subdirectory
MATERIALS = [
    ("As", "A7", "As", "noSOC", 74.921595, "As"),
    ("Sb", "A7", "Sb", "noSOC", 121.760, "Sb"),
    ("Bi", "A7", "Bi", "noSOC", 208.98040, "Bi"),
    ("Bi_SOC", "A7", r"Bi~(\SOC)", "SOC", 208.98040, "Bi_SOC"),
    ("C", "cF8", "C", "noSOC", 12.011, "C"),
    ("Si", "cF8", "Si", "noSOC", 28.0855, "Si"),
    ("Ge", "cF8", "Ge", "noSOC", 72.630, "Ge"),
]

# Materials featured in the SI table: the softest member of the set (worst case
# for a low-frequency-weighted moment) and a stiff control.
FEATURED = ["Bi", "Si"]

# Theta_DWF at f_min = 0, as published in SI Table S3 (K).
REFERENCE_THETA_DWF = {"As": 232.1, "Sb": 163.4, "Bi": 105.1, "Bi_SOC": 91.1,
                       "C": 1931.4, "Si": 500.8, "Ge": 259.8}

# Published harmonic isotropic B_iso at 295 K (A^2), linearly interpolated from
# the 290/300 K entries of bt_curves_full.csv and {C,Si,Ge}_phonon_B_curves.csv.
# Bi_SOC is absent from those long-format tables; the script falls back to the
# archived AFLOW displacements for it, which is what those tables are built from.
REFERENCE_B_295 = {"As": 0.873898, "Sb": 1.064190, "Bi": 1.506755,
                   "C": 0.143018, "Si": 0.511583, "Ge": 0.694770}


def resolve(rel, name):
    p = DFT_RAW / rel / name
    return p if p.exists() else None


def read_dos(path):
    """Columns of aflow.apl.phonon_dos.out: 0 = f (THz), 3 = pDOS."""
    f, d = [], []
    with lzma.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                f.append(float(parts[0]))
                d.append(float(parts[3]))
            except ValueError:
                continue
    return np.array(f), np.array(d)


def read_aflow_msd(path, temperature):
    """Isotropic mean-square displacement (A^2) from aflow.apl.displacements.out.

    The file lists T, species, and <u_x^2>, <u_y^2>, <u_z^2> for every atom;
    rows after the first of a temperature block omit the temperature field.
    Returns the value at `temperature` (linear interpolation on the tabulated T
    grid), averaged over the three Cartesian components and over all atoms.
    """
    by_T, current = {}, None
    with lzma.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            parts = line.split()
            try:
                if len(parts) == 5:
                    current = float(parts[0])
                    vals = [float(x) for x in parts[2:5]]
                elif len(parts) == 4 and current is not None:
                    vals = [float(x) for x in parts[1:4]]
                else:
                    continue
            except ValueError:
                continue
            by_T.setdefault(current, []).append(sum(vals) / 3.0)
    if not by_T:
        return float("nan")
    temps = np.array(sorted(by_T))
    means = np.array([float(np.mean(by_T[t])) for t in temps])
    return float(np.interp(temperature, temps, means))


def positive_bins(f, d):
    """Restrict to positive frequencies and recover the histogram bin edges.

    The archived grid is uniform bin centres (the first sample sits at
    delta/2), so the edges are centre +/- delta/2.
    """
    pos = f > 0
    fc, dc = f[pos], d[pos]
    delta = float(np.median(np.diff(fc)))
    return fc, dc, delta, float(fc[0] - 0.5 * delta), float(fc[-1] + 0.5 * delta)


def segments(f, g):
    """Piecewise-linear coefficients g(f) = a + b f on each [f_i, f_{i+1}]."""
    f0, f1 = f[:-1], f[1:]
    g0, g1 = g[:-1], g[1:]
    b = (g1 - g0) / (f1 - f0)
    return f0, f1, g0 - b * f0, b


def pw_norm(f, g):
    """Exact integral of the piecewise-linear interpolant -- the trapezoid rule
    is exact here, so normalizations are shared with the trapezoidal moments."""
    return 0.5 * float(((g[:-1] + g[1:]) * (f[1:] - f[:-1])).sum())


def pw_integral(f, g, lo, hi):
    """Exact integral of the piecewise-linear interpolant of g over [lo, hi]."""
    lo, hi = max(lo, float(f[0])), min(hi, float(f[-1]))
    if hi <= lo:
        return 0.0
    inner = (f > lo) & (f < hi)
    xs = np.concatenate(([lo], f[inner], [hi]))
    ys = np.concatenate(([float(np.interp(lo, f, g))], g[inner],
                         [float(np.interp(hi, f, g))]))
    return float(_trapz(ys, xs))


def rebin(fc, dc, lo, hi, n_bins):
    """Weight-conserving coarsening onto `n_bins` uniform bins over [lo, hi].

    Each coarse value is the exact piecewise-linear integral of the fine DOS
    over the coarse bin divided by the bin width, sampled at the bin centre --
    the same convention as the archived file. Bins beyond the tabulated support
    integrate to zero, so the total weight is preserved.
    """
    edges = np.linspace(lo, hi, n_bins + 1)
    width = edges[1] - edges[0]
    vals = np.array([pw_integral(fc, dc, edges[j], edges[j + 1]) / width
                     for j in range(n_bins)])
    return 0.5 * (edges[:-1] + edges[1:]), vals


def trapz_moment(f, g, weight):
    """The pipeline's rule: normalize by the trapezoidal integral of the DOS,
    then apply the trapezoidal rule to the weighted DOS."""
    return float(_trapz(g * weight(f), f) / _trapz(g, f))


def exact_inv2(f, g):
    """Exact <f^-2> for a piecewise-linear g.

    Per interval, int (a + b f) f^-2 df = a (1/f0 - 1/f1) + b ln(f1/f0).
    """
    f0, f1, a, b = segments(f, g)
    return float((a * (1.0 / f0 - 1.0 / f1) + b * np.log(f1 / f0)).sum()
                 / pw_norm(f, g))


# 32-point Gauss-Legendre nodes/weights on [-1, 1].
_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)


def exact_weighted(f, g, weight):
    """Exact <weight> for a piecewise-linear g, by Gauss-Legendre per interval.

    The weight is smooth on every interval -- the f -> 0 divergence of
    f^-1 coth(h f / 2 kB T) lies outside the support, since the first bin centre
    is positive -- so 32 nodes are converged to machine precision.
    """
    f0, f1, a, b = segments(f, g)
    half, mid = 0.5 * (f1 - f0), 0.5 * (f1 + f0)
    x = mid[:, None] + half[:, None] * _GL_X[None, :]
    y = (a[:, None] + b[:, None] * x) * weight(x)
    return float((half * (y * _GL_W[None, :]).sum(axis=1)).sum() / pw_norm(f, g))


def subdivided_moment(f, g, weight, k):
    """Trapezoidal moment of the SAME piecewise-linear function sampled on k
    sub-intervals per tabulated interval. Converges as O(k^-2) to the exact
    value, which makes it an independent check on the two routines above."""
    xs = [np.linspace(f[i], f[i + 1], k + 1)[:-1] for i in range(len(f) - 1)]
    xs.append(np.array([f[-1]]))
    x = np.concatenate(xs)
    return trapz_moment(x, np.interp(x, f, g), weight)


def theta_dwf(inv2):
    """Theta_DWF = h sqrt(3/<f^-2>) / kB, with <f^-2> in THz^-2."""
    return H_SI * math.sqrt(3.0 / inv2) * THZ / KB_SI


def b_iso(weighted_mean, mass_amu):
    """B_iso = 8 pi^2 <u^2> from <f^-1 coth(h f / 2 kB T)> in THz^-1."""
    u2_m2 = HBAR_SI / (4.0 * math.pi * mass_amu * AMU) * weighted_mean / THZ
    return 8.0 * math.pi ** 2 * u2_m2 * 1e20


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--temperature", type=float, default=295.0,
                    help="temperature for B_iso in K (default 295)")
    ap.add_argument("--factors", type=int, nargs="+", default=[16, 8, 4, 2, 1],
                    help="coarsening factors applied to the archived mesh "
                         "(default 16 8 4 2 1; 1 = the archived mesh itself)")
    ap.add_argument("--outdir", type=Path, default=PROCESSED,
                    help="directory for the CSV and LaTeX outputs")
    ap.add_argument("--no-selftest", action="store_true",
                    help="skip the subdivision-refinement check of the exact "
                         "integrals")
    args = ap.parse_args()

    T = args.temperature

    def quantum_weight(f):
        return (1.0 / f) / np.tanh(H_SI * f * THZ / (2.0 * KB_SI * T))

    def inv_square(f):
        return 1.0 / f ** 2

    rows, selftest = [], []
    for name, family, tex, soc, mass, rel in MATERIALS:
        dos_path = resolve(rel, "aflow.apl.phonon_dos.out.xz")
        if dos_path is None:
            print(f"  MISSING phonon DOS for {name}", file=sys.stderr)
            continue
        fc, dc, delta, lo, hi = positive_bins(*read_dos(dos_path))
        n_archived = len(fc)

        msd_path = resolve(rel, "aflow.apl.displacements.out.xz")
        b_aflow = (8.0 * math.pi ** 2 * read_aflow_msd(msd_path, T)
                   if msd_path is not None else float("nan"))

        def record(mesh, n_bins, factor, width, rule, inv2, wmean):
            rows.append(dict(
                material=name, family=family, soc_status=soc,
                mesh=mesh, n_bins=n_bins, coarsening_factor=factor,
                delta_f_THz=width, rule=rule,
                inv_sq_moment_THz_minus2=inv2,
                theta_dwf_K=theta_dwf(inv2),
                B_iso_dos_route_A2=b_iso(wmean, mass),
                temperature_K=T,
                B_iso_aflow_msd_A2=b_aflow,
                theta_dwf_published_K=REFERENCE_THETA_DWF.get(name, ""),
                B_iso_published_A2=REFERENCE_B_295.get(name, ""),
            ))

        for factor in sorted(set(args.factors), reverse=True):
            n_bins = max(2, n_archived // factor)
            gf, gd = ((fc, dc) if factor == 1        # archived mesh, untouched
                      else rebin(fc, dc, lo, hi, n_bins))
            record(f"{len(gf)} bins", len(gf), factor, (hi - lo) / len(gf),
                   "trapezoid", trapz_moment(gf, gd, inv_square),
                   trapz_moment(gf, gd, quantum_weight))

        inv2_x = exact_inv2(fc, dc)
        wmean_x = exact_weighted(fc, dc, quantum_weight)
        record("exact limit", n_archived, 0, 0.0, "exact piecewise-linear",
               inv2_x, wmean_x)

        if not args.no_selftest:
            for k in (16, 256):
                selftest.append((
                    name, k,
                    subdivided_moment(fc, dc, inv_square, k) / inv2_x - 1.0,
                    subdivided_moment(fc, dc, quantum_weight, k) / wmean_x - 1.0))

    if not rows:
        sys.exit("no phonon DOS files found under "
                 f"{DFT_RAW} -- is data/dft_raw/ present in this checkout?")

    # Rows are keyed by coarsening factor, not by bin count: the factor is
    # material-independent even if a spectrum were tabulated on a different
    # number of positive bins. Factor 0 denotes the exact fine-mesh limit.
    def pick(name, factor):
        for r in rows:
            if r["material"] == name and r["coarsening_factor"] == factor:
                return r
        return None

    def archived(name):
        return pick(name, 1)

    materials = [m[0] for m in MATERIALS if archived(m[0]) is not None]
    factors = sorted(set(args.factors), reverse=True) + [0]

    def dev(name, factor, quantity, ref_factor):
        return 100.0 * (pick(name, factor)[quantity]
                        / pick(name, ref_factor)[quantity] - 1.0)

    def worst(quantity, factor, ref_factor):
        return max(abs(dev(n, factor, quantity, ref_factor)) for n in materials)

    def worst_abs(quantity, factor, ref_factor):
        return max(abs(pick(n, factor)[quantity] - pick(n, ref_factor)[quantity])
                   for n in materials)

    # ---- self-test of the exact integrals ----------------------------------
    if selftest:
        by_k = {}
        for _, k, e2, eq in selftest:
            by_k.setdefault(k, []).append(max(abs(e2), abs(eq)))
        print("=== self-test: exact integrals vs trapezoidal subdivision "
              "refinement ===")
        for k in sorted(by_k):
            print(f"  {k:4d} sub-intervals per tabulated interval: "
                  f"worst relative gap {max(by_k[k]):.2e}")
        ks = sorted(by_k)
        ratio = max(by_k[ks[0]]) / max(by_k[ks[-1]])
        print(f"  gap shrinks {ratio:.0f}x for a {ks[-1] // ks[0]}x refinement "
              f"(O(k^-2) expects {(ks[-1] // ks[0]) ** 2}x), so the closed-form "
              f"and\n  Gauss-Legendre values are the converged limit, not an "
              f"independent approximation.")

    # ---- regression check against the published values ---------------------
    print(f"\n=== archived mesh vs published values (T = {T:g} K) ===")
    print(f"  {'material':8s} {'Th_DWF':>8s} {'Th_pub':>8s} {'dev':>6s}   "
          f"{'B_DOS':>7s} {'B_pub':>7s} {'dev%':>7s}   {'B_AFLOW':>7s} {'dev%':>7s}")
    for name in materials:
        r = archived(name)
        tp, bp = REFERENCE_THETA_DWF.get(name), REFERENCE_B_295.get(name)
        pub = f"{bp:7.4f} {100 * (r['B_iso_dos_route_A2'] / bp - 1):+7.2f}" \
            if bp else f"{'--':>7s} {'--':>7s}"
        print(f"  {name:8s} {r['theta_dwf_K']:8.1f} {tp:8.1f} "
              f"{r['theta_dwf_K'] - tp:+6.2f}   "
              f"{r['B_iso_dos_route_A2']:7.4f} {pub}   "
              f"{r['B_iso_aflow_msd_A2']:7.4f} "
              f"{100 * (r['B_iso_dos_route_A2'] / r['B_iso_aflow_msd_A2'] - 1):+7.2f}")
    print(f"  Theta_DWF reproduces SI Table S3 to "
          f"{max(abs(archived(n)['theta_dwf_K'] - REFERENCE_THETA_DWF[n]) for n in materials):.2f} K "
          f"(that table quotes one decimal).")
    print("  B_pub equals 8 pi^2 x the archived AFLOW mean-square displacement,")
    print("  so the B_DOS-B_pub gap is the DOS-versus-mode-resolved difference,")
    print("  NOT a quadrature error: the published B_iso(T) curves never touch")
    print("  the tabulated frequency grid.")

    # ---- the convergence tables --------------------------------------------
    def label(factor):
        return pick(materials[0], factor)["mesh"]

    for quantity, fmt, title in (
            ("theta_dwf_K", "9.2f", "Theta_DWF (K)"),
            ("B_iso_dos_route_A2", "9.4f", f"B_iso({T:g} K), DOS route (A^2)")):
        print(f"\n=== frequency-mesh convergence of {title} ===")
        print(f"  {'mesh':>16s} " + " ".join(f"{m:>9s}" for m in materials))
        for factor in factors:
            print(f"  {label(factor):>16s} "
                  + " ".join(format(pick(n, factor)[quantity], fmt)
                             for n in materials))
        print(f"  {'% from exact':>16s} " + " ".join(
            f"{dev(n, 1, quantity, 0):+9.3f}" for n in materials))

    # ---- the numbers quoted in the Methods --------------------------------
    coarsest = factors[0]
    bound = dict(
        theta_exact=worst("theta_dwf_K", 1, 0),
        b_exact=worst("B_iso_dos_route_A2", 1, 0),
        theta_coarse=worst("theta_dwf_K", coarsest, 1),
        b_coarse=worst("B_iso_dos_route_A2", coarsest, 1),
        theta_exact_abs=worst_abs("theta_dwf_K", 1, 0),
        b_exact_abs=worst_abs("B_iso_dos_route_A2", 1, 0),
        theta_coarse_abs=worst_abs("theta_dwf_K", coarsest, 1),
        b_coarse_abs=worst_abs("B_iso_dos_route_A2", coarsest, 1),
        selftest_gap=max(max(abs(e2), abs(eq))
                         for _, k, e2, eq in selftest if k == 256) if selftest else None,
        n_coarse=pick(materials[0], coarsest)["n_bins"],
        n_archived=pick(materials[0], 1)["n_bins"],
        n_materials=len(materials),
        coarsest=coarsest,
    )
    print("\n=== numerical bound for the Methods ===")
    print(f"  worst case over {bound['n_materials']} calculations:")
    print(f"    archived {bound['n_archived']}-bin mesh vs the exact "
          f"Delta_f -> 0 limit")
    print(f"      Theta_DWF     {bound['theta_exact']:.3f} %  "
          f"({bound['theta_exact_abs']:.3f} K)")
    print(f"      B_iso({T:g} K)  {bound['b_exact']:.3f} %  "
          f"({bound['b_exact_abs']:.5f} A^2)")
    print(f"    coarsening to {bound['n_coarse']} bins (factor {coarsest}) "
          f"vs the archived mesh")
    print(f"      Theta_DWF     {bound['theta_coarse']:.3f} %  "
          f"({bound['theta_coarse_abs']:.3f} K)")
    print(f"      B_iso({T:g} K)  {bound['b_coarse']:.3f} %  "
          f"({bound['b_coarse_abs']:.5f} A^2)")
    # Round upward so a "less than" statement quotes a conservative bound.
    def up(x):
        return math.ceil(100 * x) / 100
    print(f"  quotable bounds (rounded up): stored vs exact < "
          f"{up(bound['theta_exact']):.2f} % (Theta_DWF), < {up(bound['b_exact']):.2f} % "
          f"(B_iso); {bound['n_coarse']} bins vs stored < "
          f"{up(bound['theta_coarse']):.2f} % and < {up(bound['b_coarse']):.2f} %")
    print("  most mesh-sensitive material: " + max(
        materials, key=lambda n: abs(dev(n, coarsest, "theta_dwf_K", 1))))
    print("\n  per-material change from the archived mesh to "
          f"{bound['n_coarse']} bins:")
    for n in materials:
        print(f"    {n:8s} Theta_DWF {dev(n, coarsest, 'theta_dwf_K', 1):+7.3f} %"
              f"  ({pick(n, coarsest)['theta_dwf_K'] - archived(n)['theta_dwf_K']:+7.3f} K)"
              f"   B_iso {dev(n, coarsest, 'B_iso_dos_route_A2', 1):+7.3f} %"
              f"  ({pick(n, coarsest)['B_iso_dos_route_A2'] - archived(n)['B_iso_dos_route_A2']:+9.5f} A^2)")

    # ---- outputs -----------------------------------------------------------
    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "frequency_mesh_convergence.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"\nWrote {csv_path}")

    tex_path = args.outdir / "frequency_mesh_convergence_table.tex"
    featured = [n for n in FEATURED if n in materials]
    tex_path.write_text("\n".join(
        build_table(featured, factors, pick, dev, bound)) + "\n")
    print(f"Wrote {tex_path}")


def build_table(featured, factors, pick, dev, bound):
    """Emit tab:freq_mesh_convergence for TEX/7_SI.tex, without caption.

    The caption is written by hand in the SI; the fragment marks where it goes.
    It references labels defined in the SI (eq:pw_inverse_square,
    tab:theta_dwf_cutoff), so it is a fragment rather than a standalone document.
    """
    tex_of = {m[0]: m[2] for m in MATERIALS}
    n = len(featured)
    coarsest = factors[0]
    lines = [
        r"\begin{table}[h!]",
        r"\vspace{0.25cm}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"% caption: written by hand in TEX/7_SI.tex",
        r"\label{tab:freq_mesh_convergence}",
        r"\vspace{-0.25cm}",
        rf"\begin{{tabular}}{{l{'cc' * n}}}",
        r"\hline",
        "Frequency mesh & " + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{{tex_of[m]}}}" for m in featured) + r" \\",
        " ".join(rf"\cline{{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(n)),
        " & " + " &\n".join(
            r"$\Theta_{\mathrm{DWF}}$~(K) & $B_{\mathrm{iso}}$~(\AA$^2$)"
            for _ in featured) + r" \\",
        r"%.",
        r"\hline",
    ]

    def cells(fn):
        return " & ".join(fn(m) for m in featured)

    for factor in factors:
        if factor == 0:
            lines.append(r"\hline")
            name = r"exact, $\Delta f\rightarrow0$"
        else:
            name = "%d~bins%s" % (pick(featured[0], factor)["n_bins"],
                                  " (stored)" if factor == 1 else "")
        lines.append(f"{name} & " + cells(
            lambda m: f"{pick(m, factor)['theta_dwf_K']:.2f} & "
                      f"{pick(m, factor)['B_iso_dos_route_A2']:.4f}") + r" \\")
        lines.append(r"%.")

    def pct(m, factor, ref_factor):
        return " & ".join(
            f"${dev(m, factor, q, ref_factor):+.3f}\\%$"
            for q in ("theta_dwf_K", "B_iso_dos_route_A2"))

    lines += [
        r"\hline",
        r"stored $-$ exact & " + cells(lambda m: pct(m, 1, 0)) + r" \\",
        r"%.",
        "%d~bins $-$ stored & " % bound["n_coarse"]
        + cells(lambda m: pct(m, coarsest, 1)) + r" \\",
        r"%.",
        r"\hline",
        r"published & " + cells(
            lambda m: f"{REFERENCE_THETA_DWF[m]:.1f} & "
                      f"{REFERENCE_B_295[m]:.4f}") + r" \\",
        r"%.",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return lines


if __name__ == "__main__":
    main()
