"""Per-reflection intensity error of each displacement model, for every material.

Backs main-text Table III and the cross-material sentence in the Results, which
points at it. The table is placed beside the Methods paragraph that defines the
ratio, so its caption carries the method and nothing is added to the SI.
Post-processing
only -- no new first-principles calculations. The main text illustrates the
propagation of a displacement-model error into Bragg intensities with Sb; this
script runs the identical calculation for all seven archived calculations at
their own room-temperature experimental references, so the choice of Sb can be
seen against the whole set instead of standing alone.

What is computed. For an isotropic displacement parameter the intensity of a
reflection at scattering variable s = sin(theta)/lambda scales as exp(-2 B s^2),
so a model error Delta_B = B_model - B_exp produces

    Delta_I / I = exp(-2 Delta_B s^2) - 1.

Structure factors, multiplicities, Lorentz-polarization, form factors, profile
shape and background all cancel from this ratio: it depends only on Delta_B and
on s. What the crystal structure supplies is therefore just the list of
reflections that exist -- which s values are actually sampled -- and that is what
the archived CONTCARs are read for.

Because |Delta_I/I| is monotone in s for either sign of Delta_B, its maximum over
an angular window is attained at the window's highest-angle allowed reflection.
The reported maximum is therefore a well-defined high-angle bound rather than a
selected peak; the script asserts this monotonicity rather than assuming it.

Reflections. Miller indices are enumerated over the complete range allowed by the
Ewald limit: h_i = G . a_i / 2 pi, so |h_i| <= q_max |a_i| with q_max = 2/lambda.
(The published Fig. 4 script used a fixed +/-10 cube, which is short of complete
for the A7 c axis; it happens not to change any published number, and the
regression check below confirms that.) Reflections sharing a d-spacing are
grouped, as they would overlap in a pattern, and a group counts as observable
when the summed geometric structure factor over its members is non-zero. Since
each cell here holds a single species, the atomic form factor is a common
prefactor and does not affect which reflections are allowed.

Sources of B. Model and experimental values are taken at a common temperature
per material, from the same tables the manuscript benchmarks against:
bt_at_experiment_temperatures.csv for the A7 set (Fischer 1978 neutron points)
and {C,Si,Ge}_phonon_experimental_benchmark.csv for the diamond-structure
controls. Both tabulate the four model curves evaluated at the experimental
temperatures, so no interpolation onto a nominal temperature is needed. The
cF8 reference is the 290-300 K literature median of SI Fig. S1, placed at
295 K (see load_cf8_literature_median in cross_material_synthesis.py).
"""
import argparse
import csv
import lzma
import math
import sys
from pathlib import Path

import numpy as np

# Every path resolves through scripts/paths.py, the single source of truth for
# this package. Nothing outside the repository root is read or written.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DFT_RAW, PROCESSED  # noqa: E402
from cross_material_synthesis import load_cf8_literature_source_medians  # noqa: E402

CSV_IN = PROCESSED

LAMBDA_CU_KA1 = 1.540598     # Angstrom, Cu K-alpha_1; the Fig. 4 geometry
TWO_THETA_MIN = 2.0          # lower bound of the enumerated range
TWO_THETA_MAX = 180.0        # backscattering limit
D_TOL = 1e-4                 # Angstrom, d-spacing tolerance for grouping
F2_TOL = 1e-8                # geometric |F|^2 below this counts as extinct

# label, family, tex label, SOC status, key in bt_at_experiment_temperatures.csv,
# dft_raw/ subdirectory (same keys as frequency_mesh_convergence.py)
MATERIALS = [
    ("As", "A7", "As", "noSOC", "As", "As"),
    ("Sb", "A7", "Sb", "noSOC", "Sb", "Sb"),
    ("Bi", "A7", "Bi", "noSOC", "Bi_noSOC", "Bi"),
    ("Bi_SOC", "A7", r"Bi~(\SOC)", "SOC", "Bi_SOC", "Bi_SOC"),
    ("C", "cF8", "C", "noSOC", None, "C"),
    ("Si", "cF8", "Si", "noSOC", None, "Si"),
    ("Ge", "cF8", "Ge", "noSOC", None, "Ge"),
]

# key, CSV/print label, tex column head, column in bt_at_experiment_temperatures,
# column in the cF8 benchmark files
TREATMENTS = [
    ("static", "no DWF (static)", "static", None, None),
    ("empirical", "empirical Debye-function reference", "empirical",
     "B_empirical_digitized_reference_A2", "B_emp_"),
    ("elastic", "elastic-property-derived Debye", "elastic",
     "B_elastic_derived_debye_A2", "B_AEL_A2"),
    ("quasi_harmonic", "quasi-harmonic Debye", "quasi-harm.",
     "B_quasi_harmonic_debye_A2", "B_AGL_A2"),
    ("harmonic", "harmonic phonon spectrum (isotropic)", "harmonic",
     "B_harmonic_phonon_spectrum_isotropic_A2", "B_harm_A2"),
]

# The A7 experimental anchors all come from the same neutron study.
A7_EXP_SOURCE = "Fischer 1978"

# Family cell as the other SI tables set it.
FAMILY_TEX = {"A7": "A7", "cF8": "$cF8$"}

# Max |dI/I| (%) over the (70, 180) deg window published for Sb in Fig. 4(c) and
# quoted in the Results; the generalized code must reproduce them exactly.
SB_PUBLISHED = {"static": 149.00, "empirical": 40.72, "elastic": 44.01,
                "quasi_harmonic": 67.34, "harmonic": 3.05}


def resolve(rel, name):
    p = DFT_RAW / rel / name
    return p if p.exists() else None


def rows_of(path):
    """DictReader over a CSV whose leading '#' lines are provenance comments."""
    with open(path) as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def as_float(value):
    if value is None:
        return float("nan")
    text = value.strip()
    if text == "" or text.upper() in {"NA", "NAN"}:
        return float("nan")
    return float(text)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def read_contcar(path):
    """Return (lattice 3x3 in Angstrom, fractional coordinates Nx3, species).

    VASP CONTCAR. A negative scale factor means "target volume"; a positive one
    multiplies the lattice. An optional 'Selective dynamics' line is skipped.
    """
    with lzma.open(path, "rt") as fh:
        lines = fh.readlines()
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[2 + i].split()[:3]]
                        for i in range(3)])
    if scale < 0:                      # -|V|: rescale to that volume
        lattice *= (abs(scale) / abs(np.linalg.det(lattice))) ** (1.0 / 3.0)
    else:
        lattice *= scale
    labels = lines[5].split()
    counts = [int(n) for n in lines[6].split()]
    species = [lab for lab, n in zip(labels, counts) for _ in range(n)]
    cursor = 7
    if lines[cursor].strip().lower().startswith("s"):   # selective dynamics
        cursor += 1
    mode = lines[cursor].strip().lower()
    if not mode.startswith("d"):
        raise ValueError(f"{path}: expected Direct coordinates, got {mode!r}")
    cursor += 1
    frac = np.array([[float(x) for x in lines[cursor + i].split()[:3]]
                     for i in range(sum(counts))])
    return lattice, frac, species


def index_bounds(lattice, lambda_xr):
    """Complete Miller-index bounds implied by the Ewald limit.

    h_i = G . a_i / 2 pi, so |G| <= 2 pi q_max gives |h_i| <= q_max |a_i| with
    q_max = 1/d_min = 2/lambda. Enumerating this box misses no reflection.
    """
    q_max = 2.0 / lambda_xr
    return [int(math.floor(q_max * float(np.linalg.norm(lattice[i]))) ) + 1
            for i in range(3)]


def reflections(lattice, frac, lambda_xr, bounds=None):
    """Group reflections by d-spacing.

    Returns a list of dicts sorted by 2theta, each holding a representative hkl,
    the multiplicity of the d-group, d, 2theta, s and the summed geometric
    structure factor |sum_n exp(2 pi i h.r_n)|^2 over the group's members.
    """
    if bounds is None:
        bounds = index_bounds(lattice, lambda_xr)
    recip = 2 * np.pi * np.linalg.inv(lattice).T
    q_max = 2.0 / lambda_xr
    found = []
    for h in range(-bounds[0], bounds[0] + 1):
        for k in range(-bounds[1], bounds[1] + 1):
            for l in range(-bounds[2], bounds[2] + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                q = np.linalg.norm(h * recip[0] + k * recip[1] + l * recip[2])
                q /= 2 * np.pi
                if q <= 0 or q > q_max:
                    continue
                d = 1.0 / q
                sin_theta = lambda_xr / (2.0 * d)
                if sin_theta >= 1.0:
                    continue
                two_theta = 2.0 * math.degrees(math.asin(sin_theta))
                if two_theta < TWO_THETA_MIN or two_theta > TWO_THETA_MAX:
                    continue
                phase = np.exp(2j * np.pi * (h * frac[:, 0] + k * frac[:, 1]
                                             + l * frac[:, 2]))
                found.append((d, two_theta, (h, k, l), float(abs(phase.sum()) ** 2)))
    # Sorted by descending d, entries within D_TOL of a group's first member are
    # contiguous, so one linear pass groups them.
    found.sort(key=lambda x: -x[0])
    groups, i = [], 0
    while i < len(found):
        d, two_theta, hkl, total = found[i]
        j = i + 1
        while j < len(found) and abs(found[j][0] - d) < D_TOL:
            total += found[j][3]
            j += 1
        groups.append(dict(hkl=hkl, multiplicity=j - i, d_A=d,
                           two_theta_deg=two_theta, s_invA=0.5 / d,
                           geom_F2=total, allowed=total > F2_TOL))
        i = j
    groups.sort(key=lambda g: g["two_theta_deg"])
    return groups


# ---------------------------------------------------------------------------
# Displacement parameters
# ---------------------------------------------------------------------------

def a7_b_values(bt_key, target_T):
    """B_exp and the four model B values for an A7 calculation.

    bt_at_experiment_temperatures.csv pairs every experimental point with the
    model curves evaluated at that same temperature, so the row nearest
    `target_T` gives a consistent set with no interpolation.
    """
    rows = [r for r in rows_of(CSV_IN / "bt_at_experiment_temperatures.csv")
            if r["material"] == bt_key]
    if not rows:
        raise RuntimeError(f"no bt_at_experiment_temperatures rows for {bt_key}")
    row = min(rows, key=lambda r: abs(float(r["T_K"]) - target_T))
    B = {"static": 0.0}
    for key, _, _, col, _ in TREATMENTS:
        if col is not None:
            B[key] = as_float(row.get(col))
    return float(row["T_K"]), as_float(row["B_exp_A2"]), B, A7_EXP_SOURCE


def cf8_b_values(name, target_T):
    """B_exp and the four model B values for a diamond-structure control.

    The benchmark file holds a single row, the 290-300 K literature median at
    295 K. The empirical column is named after that material's empirical Debye
    temperature (B_emp_<Theta>K_A2), so it is found by prefix.
    """
    rows = rows_of(CSV_IN / f"{name}_phonon_experimental_benchmark.csv")
    row = min(rows, key=lambda r: abs(float(r["T_K"]) - target_T))
    B = {"static": 0.0}
    for key, _, _, _, col in TREATMENTS:
        if col is None:
            continue
        if col.endswith("_"):                       # prefix match
            hits = [c for c in row if c.startswith(col)]
            if len(hits) != 1:
                raise RuntimeError(f"{name}: expected one {col}* column, got {hits}")
            col = hits[0]
        B[key] = as_float(row.get(col))
    return float(row["T_K"]), as_float(row["B_exp_A2"]), B, row.get("source", "")


def patch_bi_soc(B, target_T, notes):
    """Fill the two SOC entries that the long-format table leaves empty.

    bt_at_experiment_temperatures.csv carries only the elastic and
    quasi-harmonic curves for Bi with SOC. The harmonic isotropic SOC curve is
    tabulated separately. The empirical Debye-function reference is built from
    the *measured* Debye temperature and therefore has no SOC variant at all --
    it is the same curve as for scalar-relativistic Bi, which is what is used
    here; only the reflection positions differ between the two rows.
    """
    if not math.isnan(B.get("harmonic", float("nan"))):
        return
    curve = rows_of(CSV_IN / "Bi_SOC_phonon_harmonic_curves.csv")
    T = np.array([float(r["T_K"]) for r in curve])
    y = np.array([as_float(r["B_iso_SOC_A2"]) for r in curve])
    order = np.argsort(T)
    B["harmonic"] = float(np.interp(target_T, T[order], y[order]))
    notes.append("harmonic B from Bi_SOC_phonon_harmonic_curves.csv")
    if math.isnan(B.get("empirical", float("nan"))):
        _, _, B_noSOC, _ = a7_b_values("Bi_noSOC", target_T)
        B["empirical"] = B_noSOC["empirical"]
        notes.append("empirical reference has no SOC variant; shared with Bi")


# ---------------------------------------------------------------------------
# Intensity error
# ---------------------------------------------------------------------------

def di_over_i_pct(delta_B, s):
    """100 * (exp(-2 Delta_B s^2) - 1), the fractional intensity change."""
    return 100.0 * (np.exp(-2.0 * delta_B * np.asarray(s) ** 2) - 1.0)


def closest_model(m, b_ref):
    """The displacement treatment with the smallest error against b_ref.

    The window maximum sits on the highest-angle allowed reflection (see the
    monotonicity self-test), so the errors are compared there. An exact tie
    goes to the scalar treatment; the table caption also reports errors that
    are equal at its one-decimal precision as a tie (see build_table).
    """
    err = {k: abs(float(di_over_i_pct(m["B"][k] - b_ref, m["top"]["s_invA"])))
           for k in ["empirical", "elastic", "quasi_harmonic", "harmonic"]}
    return min(err, key=err.get)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--temperature", type=float, default=295.0,
                    help="target temperature in K; each material uses its own "
                         "experimental reference nearest this value (default 295)")
    ap.add_argument("--window", type=float, nargs=2, default=[70.0, 180.0],
                    metavar=("LO", "HI"),
                    help="high-angle window in 2theta degrees (default 70 180, "
                         "the published Fig. 4 window)")
    ap.add_argument("--window-b", type=float, nargs=2, default=[70.0, 160.0],
                    metavar=("LO", "HI"),
                    help="second window, reported alongside the first so the "
                         "numbers can be checked against a detector limit short "
                         "of backscattering (default 70 160, the angle used for "
                         "the acceptance bound in the Discussion)")
    ap.add_argument("--wavelength", type=float, default=LAMBDA_CU_KA1,
                    help=f"X-ray wavelength in A (default {LAMBDA_CU_KA1}, "
                         f"Cu K-alpha_1)")
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="intensity tolerance for the acceptance bound "
                         "|Delta_B| <= ln(1+eps)/(2 s_max^2) (default 0.05)")
    ap.add_argument("--outdir", type=Path, default=PROCESSED,
                    help="directory for the CSV and LaTeX outputs")
    ap.add_argument("--no-selftest", action="store_true",
                    help="skip the completeness and monotonicity checks")
    args = ap.parse_args()

    win_lo, win_hi = args.window
    lam = args.wavelength

    per_reflection, summary, materials = [], [], []
    for name, family, tex, soc, bt_key, rel in MATERIALS:
        path = resolve(rel, "CONTCAR.static.xz")
        if path is None:
            print(f"  MISSING CONTCAR for {name}", file=sys.stderr)
            continue
        lattice, frac, species = read_contcar(path)
        bounds = index_bounds(lattice, lam)
        groups = reflections(lattice, frac, lam, bounds)

        notes = []
        ref_sources = {}
        if bt_key is not None:
            T_exp, B_exp, B, source = a7_b_values(bt_key, args.temperature)
            if name == "Bi_SOC":
                patch_bi_soc(B, T_exp, notes)
        else:
            T_exp, B_exp, B, source = cf8_b_values(name, args.temperature)
            ref_sources = load_cf8_literature_source_medians(name)

        def select(win):
            hits = [g for g in groups if g["allowed"]
                    and win[0] <= g["two_theta_deg"] <= win[1]]
            if not hits:
                raise RuntimeError(f"{name}: no allowed reflection in "
                                   f"[{win[0]}, {win[1]}] deg")
            return hits, max(hits, key=lambda g: g["s_invA"])

        in_win, top = select(args.window)
        in_win_b, top_b = select(args.window_b)
        allowed = [g for g in groups if g["allowed"]]

        materials.append(dict(
            name=name, family=family, tex=tex, soc=soc, T_exp=T_exp,
            B_exp=B_exp, B=B, source=source, ref_sources=ref_sources,
            notes="; ".join(notes),
            n_groups=len(groups), n_allowed=len(allowed),
            n_window=len(in_win), top=top, top_b=top_b, lattice=lattice,
            frac=frac, bounds=bounds, groups=groups))

        for g in groups:
            row = dict(
                material=name, family=family, soc_status=soc,
                h=g["hkl"][0], k=g["hkl"][1], l=g["hkl"][2],
                multiplicity=g["multiplicity"], d_A=g["d_A"],
                two_theta_deg=g["two_theta_deg"], s_invA=g["s_invA"],
                geometric_F2=g["geom_F2"], allowed=int(g["allowed"]),
                in_high_angle_window=int(g["allowed"]
                                         and win_lo <= g["two_theta_deg"] <= win_hi),
                temperature_K=T_exp, B_exp_A2=B_exp)
            for key, _, _, _, _ in TREATMENTS:
                row[f"dI_over_I_pct_{key}"] = float(
                    di_over_i_pct(B[key] - B_exp, g["s_invA"]))
            per_reflection.append(row)

        s_win = np.array([g["s_invA"] for g in in_win])
        s_win_b = np.array([g["s_invA"] for g in in_win_b])
        s_all = np.array([g["s_invA"] for g in allowed])
        limit = math.log(1.0 + args.tolerance) / (2.0 * top["s_invA"] ** 2)
        for key, label, _, _, _ in TREATMENTS:
            dB = B[key] - B_exp
            d_win = np.abs(di_over_i_pct(dB, s_win))
            summary.append(dict(
                material=name, family=family, soc_status=soc, treatment=key,
                treatment_label=label, temperature_K=T_exp, B_exp_A2=B_exp,
                B_model_A2=B[key], delta_B_A2=dB,
                window_lo_deg=args.window[0], window_hi_deg=args.window[1],
                max_abs_dI_over_I_pct_window=float(d_win.max()),
                median_abs_dI_over_I_pct_window=float(np.median(d_win)),
                two_theta_at_max_deg=top["two_theta_deg"],
                s_max_invA=top["s_invA"],
                hkl_at_max="%d %d %d" % top["hkl"],
                n_allowed_in_window=len(in_win),
                window_b_lo_deg=args.window_b[0], window_b_hi_deg=args.window_b[1],
                max_abs_dI_over_I_pct_window_b=float(
                    np.abs(di_over_i_pct(dB, s_win_b)).max()),
                two_theta_at_max_window_b_deg=top_b["two_theta_deg"],
                max_abs_dI_over_I_pct_all=float(
                    np.abs(di_over_i_pct(dB, s_all)).max()),
                delta_B_limit_A2=limit,
                within_tolerance=int(abs(dB) <= limit),
                tolerance=args.tolerance,
                exp_source=source, notes="; ".join(notes)))

    if not materials:
        sys.exit(f"no CONTCARs found under {DFT_RAW} -- is data/dft_raw/ present?")

    def get(name, key):
        for r in summary:
            if r["material"] == name and r["treatment"] == key:
                return r
        return None

    names = [m["name"] for m in materials]

    # ---- self-tests --------------------------------------------------------
    if not args.no_selftest:
        print("=== self-test: enumeration completeness ===")
        for m in materials:
            wider = reflections(m["lattice"], m["frac"], lam,
                                [b + 2 for b in m["bounds"]])
            n_a = sum(1 for g in wider if g["allowed"])
            s_a = max(g["s_invA"] for g in wider
                      if g["allowed"] and win_lo <= g["two_theta_deg"] <= win_hi)
            ok = (n_a == m["n_allowed"]
                  and abs(s_a - m["top"]["s_invA"]) < 1e-12)
            print(f"  {m['name']:7s} bounds {tuple(m['bounds'])}: "
                  f"{m['n_allowed']:4d} allowed groups, s_max={m['top']['s_invA']:.6f}"
                  f"  -> unchanged when widened by 2: {'yes' if ok else 'NO'}")
            if not ok:
                sys.exit(f"{m['name']}: index bounds are not complete "
                         f"({m['n_allowed']} -> {n_a} groups, "
                         f"s_max {m['top']['s_invA']:.6f} -> {s_a:.6f})")

        print("\n=== self-test: |dI/I| is monotone in s, so the window maximum "
              "sits on\n    the highest-angle allowed reflection ===")
        worst = 0.0
        for m in materials:
            in_win = [g for g in m["groups"] if g["allowed"]
                      and win_lo <= g["two_theta_deg"] <= win_hi]
            s_win = np.array([g["s_invA"] for g in in_win])
            for key, _, _, _, _ in TREATMENTS:
                dB = m["B"][key] - m["B_exp"]
                d = np.abs(di_over_i_pct(dB, s_win))
                if int(np.argmax(d)) != int(np.argmax(s_win)) and d.max() > 0:
                    sys.exit(f"{m['name']}/{key}: window maximum is not at "
                             f"s_max -- the monotonicity claim fails")
                worst = max(worst, abs(d.max()
                                       - abs(float(di_over_i_pct(dB, m["top"]["s_invA"])))))
        print(f"  holds for all {len(materials)} x {len(TREATMENTS)} cases "
              f"(largest gap between the brute-force maximum and the value at\n"
              f"  s_max: {worst:.2e} percentage points)")

        if "Sb" in names:
            print("\n=== regression: published Sb values, Fig. 4(c) and Results ===")
            gap = 0.0
            for key, _, _, _, _ in TREATMENTS:
                r = get("Sb", key)
                pub = SB_PUBLISHED[key]
                gap = max(gap, abs(r["max_abs_dI_over_I_pct_window"] - pub))
                print(f"  {key:14s} {r['max_abs_dI_over_I_pct_window']:7.2f}%  "
                      f"published {pub:7.2f}%")
            if gap > 0.005:
                sys.exit(f"Sb regression failed: worst gap {gap:.4f} "
                         f"percentage points")
            print(f"  worst gap {gap:.4f} percentage points, so the generalized "
                  f"enumeration\n  reproduces the published figure exactly.")

    # ---- the table, as printed --------------------------------------------
    head = f"  {'material':8s} {'T(K)':>6s} {'B_exp':>7s} {'2th_max':>8s} "
    for column, top_key, fmt, label in (
            ("max_abs_dI_over_I_pct_window", "top", "10.2f",
             f"max |dI/I| (%) over 2theta in [{win_lo:g}, {win_hi:g}] deg"),
            ("max_abs_dI_over_I_pct_window_b", "top_b", "10.2f",
             f"max |dI/I| (%) over 2theta in "
             f"[{args.window_b[0]:g}, {args.window_b[1]:g}] deg"),
            ("delta_B_A2", "top", "10.4f",
             "Delta_B = B_model - B_exp (A^2)")):
        print(f"\n=== {label} ===")
        print(head + " ".join(f"{t[2]:>10s}" for t in TREATMENTS))
        for m in materials:
            print(f"  {m['name']:8s} {m['T_exp']:6.1f} {m['B_exp']:7.4f} "
                  f"{m[top_key]['two_theta_deg']:8.2f} "
                  + " ".join(format(get(m['name'], t[0])[column], fmt)
                             for t in TREATMENTS))

    # ---- what the manuscript quotes ---------------------------------------
    scalar = ["empirical", "elastic", "quasi_harmonic"]
    print(f"\n=== numbers for the Results and the SI ===")
    harm = {m["name"]: get(m["name"], "harmonic")["max_abs_dI_over_I_pct_window"]
            for m in materials}
    print("  harmonic treatment, max |dI/I| per material:")
    for n in names:
        print(f"    {n:8s} {harm[n]:7.2f}%")
    sc_lo = min(get(n, k)["max_abs_dI_over_I_pct_window"]
                for n in names for k in scalar)
    sc_hi = max(get(n, k)["max_abs_dI_over_I_pct_window"]
                for n in names for k in scalar)
    print(f"  scalar treatments span      {sc_lo:.2f}% to {sc_hi:.2f}%")
    print(f"  harmonic treatment spans    {min(harm.values()):.2f}% to "
          f"{max(harm.values()):.2f}%")
    st = [get(n, "static")["max_abs_dI_over_I_pct_window"] for n in names]
    print(f"  static reference spans      {min(st):.2f}% to {max(st):.2f}%")
    beats = [n for n in names
             if harm[n] > min(get(n, k)["max_abs_dI_over_I_pct_window"]
                              for k in scalar)]
    print(f"  materials where the harmonic error is NOT the smallest: "
          f"{', '.join(beats) if beats else 'none'}")
    print("  cF8 reference sources (290-300 K per-source medians), and the "
          "treatment\n  with the smallest error against each one:")
    for m in materials:
        for key, b in m["ref_sources"].items():
            print(f"    {m['name']:8s} {key:14s} {b:.4f}  {closest_model(m, b)}")
    worst_scalar = {n: max(get(n, k)["max_abs_dI_over_I_pct_window"]
                           for k in scalar) for n in names}
    ranked = sorted(names, key=lambda n: -worst_scalar[n])
    print(f"  worst scalar error, material order: "
          + " > ".join(f"{n} ({worst_scalar[n]:.1f}%)" for n in ranked))
    print(f"  Sb rank among the {len(names)} calculations by worst scalar "
          f"error: {ranked.index('Sb') + 1}")
    if ranked[0] == "Sb":
        print(f"  Sb leads {ranked[1]} by only "
              f"{worst_scalar['Sb'] - worst_scalar[ranked[1]]:.2f} percentage "
              f"points, so it is the top of a narrow range, not an outlier.")
    over = [n for n in names if worst_scalar[n] > 100 * args.tolerance]
    print(f"  worst scalar error exceeds the {args.tolerance:.0%} tolerance for "
          f"{len(over)} of {len(names)}: {', '.join(over)}")
    print(f"\n  acceptance bound at eps = {args.tolerance:.0%}: "
          f"|Delta_B| <= ln(1+eps)/(2 s_max^2)")
    for m in materials:
        r0 = get(m["name"], "static")
        inside = [t[2] for t in TREATMENTS if get(m["name"], t[0])["within_tolerance"]]
        print(f"    {m['name']:8s} s_max={r0['s_max_invA']:.4f} A^-1  "
              f"|Delta_B| <= {r0['delta_B_limit_A2']:.4f} A^2   "
              f"inside: {', '.join(inside) if inside else 'none'}")

    for m in materials:
        if m["notes"]:
            print(f"\n  note ({m['name']}): {m['notes']}")

    # ---- outputs -----------------------------------------------------------
    args.outdir.mkdir(parents=True, exist_ok=True)

    def dump(path, rows):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                            for k, v in r.items()})
        print(f"Wrote {path}")

    print()
    dump(args.outdir / "pattern_error_propagation.csv", per_reflection)
    dump(args.outdir / "pattern_error_propagation_summary.csv", summary)

    tex_path = args.outdir / "pattern_error_propagation_table.tex"
    tex_path.write_text("\n".join(
        build_table(materials, get, win_lo, win_hi)) + "\n")
    print(f"Wrote {tex_path}")


def build_table(materials, get, win_lo, win_hi):
    """Emit the manuscript table.

    The output is verbatim-identical to tab:pattern_error_propagation in
    TEX/6_method.tex apart from the \\revise{} review-markup wrapper lines, so a
    re-run can replace that block. The table lives beside the Methods paragraph
    that defines the ratio, so the caption carries the method and references
    manuscript labels (eq:pattern_error_check, fig:Sb_pattern_pc4); it is a
    fragment, not a standalone document. The ranking and tolerance count are
    printed by main() and quoted in the Results, not in the caption.
    """
    names = [m["name"] for m in materials]
    by_name = {m["name"]: m for m in materials}
    scalar = ["empirical", "elastic", "quasi_harmonic"]
    harm = {n: get(n, "harmonic")["max_abs_dI_over_I_pct_window"] for n in names}

    # The table prints one decimal, so a harmonic error equal to the best
    # scalar one at that precision is reported as a tie rather than a win.
    def err(n, k):
        return get(n, k)["max_abs_dI_over_I_pct_window"]

    best_scalar = {n: min(scalar, key=lambda k: err(n, k)) for n in names}
    harmonic_tied = [n for n in names
                     if f"{harm[n]:.1f}" == f"{err(n, best_scalar[n]):.1f}"]
    harmonic_best = [n for n in names if n not in harmonic_tied
                     and harm[n] < err(n, best_scalar[n])]
    harmonic_beaten = [n for n in names
                       if n not in harmonic_best and n not in harmonic_tied]

    def listed(items):
        tex = [names_tex(materials, n) for n in items]
        return tex[0] if len(tex) == 1 else ", ".join(tex[:-1]) + " and " + tex[-1]

    noun = {"empirical": "empirical reference", "elastic": "elastic model",
            "quasi_harmonic": "quasi-harmonic model"}
    claim = [f"The harmonic error is the smallest for {listed(harmonic_best)}"]
    claim += [f"and ties with the {noun[best_scalar[n]]} for {names_tex(materials, n)}"
              for n in harmonic_tied]
    claim = [c + "," for c in claim[:-1]] + [claim[-1] + ";"]

    # The caption gives harmonic overestimation as the reason a scalar model
    # lands closer; stop if that does not hold for every such row.
    for n in harmonic_beaten:
        if by_name[n]["B"]["harmonic"] <= by_name[n]["B_exp"]:
            raise ValueError(f"{n}: a scalar model lands closer, but the "
                             f"harmonic B does not exceed B_exp")
    plural = len(harmonic_beaten) > 1
    claim += [
        f"for {listed(harmonic_beaten)}, whose harmonic "
        + ("calculations overestimate" if plural else "calculation overestimates"),
        r"the measured $B$, a scalar model lands closer.",
    ]

    # Each cF8 reference is a median over several sources: flag the rows whose
    # closest treatment changes against one of those sources.
    label = {"empirical": "empirical", "elastic": "elastic",
             "quasi_harmonic": "quasi-harmonic", "harmonic": "harmonic"}
    count = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    spread_note = []
    for m in materials:
        at_median = closest_model(m, m["B_exp"])
        per_source = [closest_model(m, b) for b in m["ref_sources"].values()]
        closest = list(dict.fromkeys([at_median] + per_source))
        n_rev = sum(k != at_median for k in per_source)
        if len(closest) == 2:
            spread_note += [
                f"For {m['tex']} the {label[closest[0]]} and "
                f"{label[closest[1]]} errors change order against",
                f"{count[n_rev]} of the {count[len(per_source)]} literature "
                r"sources behind the median.",
            ]
        elif len(closest) > 2:
            raise ValueError(f"{m['name']}: the closest treatment changes "
                             f"between {closest} across the reference sources")

    caption = [
        r"\caption{\textbf{Per-reflection intensity error of each displacement",
        r"model, for every calculation in the set.}",
        r"Entries are $\max|\Delta I/I|$ from \Eq~\ref{eq:pattern_error_check}",
        r"over the allowed Cu~K$\alpha_1$ reflections of the relaxed structures",
        rf"with $2\theta\in[{win_lo:g}^\circ,{win_hi:g}^\circ]$, taking",
        r"$\Delta B=B_{\mathrm{model}}-B^{\mathrm{exp}}$.",
        r"$B^{\mathrm{exp}}$ is the neutron value nearest room temperature for the",
        r"A7 set \cite{Fischer1978_DWF_AsSbBi} and the $290$--$300$~K literature",
        r"median of Figure S1 for the $cF8$ set.",
        r"Each maximum falls on the highest-angle allowed reflection, listed as",
        r"$2\theta_{\max}$.",
        r"The Sb row is the calculation of \Fig~\ref{fig:Sb_pattern_pc4}; the two",
        r"Bi rows share the empirical $B$, which has no spin--orbit variant.",
        *claim,
        *spread_note,
    ]
    caption[-1] += "}"

    lines = [
        r"\begin{table*}[!htp]",
        r"\vspace{0.25cm}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        *caption,
        r"%.",
        r"\label{tab:pattern_error_propagation}",
        r"\vspace{-0.25cm}",
        r"\begin{tabular}{llcccccccc}",
        r"\hline",
        r"Family & Material & $T$ & $B^{\mathrm{exp}}$ & $2\theta_{\max}$ &"
        r" \multicolumn{5}{c}{$\max|\Delta I/I|$ (\%)} \\",
        r"\cline{6-10}",
        r" & & (K) & (\AA$^2$) & (deg) & "
        + " & ".join(t[2] for t in TREATMENTS) + r" \\",
        r"%.",
        r"\hline",
    ]
    previous = None
    for m in materials:
        if previous is not None and m["family"] != previous:
            lines += [r"\hline", r"%."]
        previous = m["family"]
        cells = " & ".join(
            f"{get(m['name'], t[0])['max_abs_dI_over_I_pct_window']:.1f}"
            for t in TREATMENTS)
        lines.append(
            f"{FAMILY_TEX[m['family']]} & {m['tex']} & {m['T_exp']:.0f} & "
            f"{m['B_exp']:.3f} & {m['top']['two_theta_deg']:.1f} & {cells} \\\\")
        lines.append(r"%.")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    return lines


def names_tex(materials, name):
    for m in materials:
        if m["name"] == name:
            return m["tex"]
    return name


if __name__ == "__main__":
    main()
