#!/usr/bin/env python3
"""Independently test the synthesis claim:
'scalar Debye undershoots B in Si/Ge because the diamond DOS is bimodal
 (acoustic + sharp optical peak), which a single Debye temperature cannot reproduce.'

Physics:
  Isotropic MSD (cubic, monatomic):
    <u^2>(T) = (hbar / 2 M) * INT g(w)/w * coth(hbar w / 2 kT) dw
  where g normalized to INT g dw = 1 (per atom, 3 modes -> but we keep per-mode
  normalization and the 3 dof cancel in the standard B factor convention used).
  B_iso = 8 pi^2 <u^2>/3 ... we instead reproduce the SAME convention as the
  manuscript script (B = 8 pi^2 <u^2>, u from APL) so the numbers are comparable.

  We compute B from the *real DOS* (full quantum) and from an *ideal Debye DOS*
  g_D(w) = 3 w^2 / w_D^3 (w<=w_D), with w_D set from each Theta_D used in the paper.
  Then we inspect WHICH moment differs.
"""
import lzma, math
import sys
from pathlib import Path
import numpy as np

# data/20260529_debye_moment_verification/<this file> -> project root
# Every path resolves through scripts/paths.py, the single source of truth
# for this package. Nothing outside the repository root is read or written.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DFT_RAW, FIGURES_SI, LITERATURE, PROCESSED, calc_dir  # noqa: E402
BASE_CANDIDATES = [DFT_RAW]
H = 6.62607015e-34
HBAR = H/(2*math.pi)
KB = 1.380649e-23
AMU = 1.66053906660e-27
THZ = 1e12
A2 = 1e20

MAT = {
 "C":  dict(dir="C",  mass=12.011, theta_emp=2230, theta_ael=2224.75, theta_agl=2098.68),
 "Si": dict(dir="Si", mass=28.085, theta_emp=645,  theta_ael=625.371, theta_agl=611.088),
 "Ge": dict(dir="Ge", mass=72.63,  theta_emp=374,  theta_ael=331.699, theta_agl=324.12),
}

def apl_dos(rel):
    for base in BASE_CANDIDATES:
        path = base / rel / "aflow.apl.phonon_dos.out.xz"
        if path.exists():
            return path
    return BASE_CANDIDATES[0] / rel / "aflow.apl.phonon_dos.out.xz"

def read_dos(path):
    f,d=[],[]
    with lzma.open(path,"rt",errors="replace") as fh:
        for line in fh:
            if line.lstrip().startswith("#") or not line.strip(): continue
            p=line.split()
            if len(p)<4: continue
            try: f.append(float(p[0])); d.append(float(p[3]))
            except ValueError: continue
    f=np.array(f); d=np.array(d)
    m=f>0; f=f[m]; d=d[m]
    d=d/np.trapezoid(d,f)   # normalize INT g df = 1
    return f,d            # f in THz

def msd_from_dos(f_THz, g, T, mass_amu):
    """<u^2> in m^2 using full quantum coth. g normalized to 1 over df(THz)."""
    w = 2*math.pi*f_THz*THZ          # rad/s
    M = mass_amu*AMU
    x = HBAR*w/(2*KB*T)
    coth = 1.0/np.tanh(x)
    integrand = (g) / w * coth       # note g is per dTHz; w in SI
    # convert: INT g(f)/w coth df  -> need consistent; g df is dimensionless weight
    val = np.trapezoid(integrand, f_THz) # units: s/rad * (per THz)*THz cancels? handle below
    # integrand has units (1/THz)/(rad/s); integrate over THz -> 1/(rad/s)=s/rad
    # <u^2> = hbar/(2M) * INT  -> (J s)/kg * s = m^2  (since J=kg m^2/s^2)
    return HBAR/(2*M)*val

def debye_dos(theta_K, f_THz):
    """ideal Debye g(f)=3 f^2/fD^3 for f<=fD, normalized INT=1 automatically."""
    fD = KB*theta_K/H/THZ            # Debye freq in THz
    g = np.where(f_THz<=fD, 3*f_THz**2/fD**3, 0.0)
    return g, fD

def moment_invsq(f_THz, g):
    """<w^-2> in s^2/rad^2 ; report also as <f^-2> THz^-2 for intuition."""
    return np.trapezoid(g/f_THz**2, f_THz)   # THz^-2

print(f"{'mat':>3} {'<f^-2>_real':>12} {'<f^-2>_Demp':>12} {'<f^-2>_Dael':>12}  ratio(real/Demp)")
for m,c in MAT.items():
    f,g = read_dos(apl_dos(c["dir"]))
    inv_real = moment_invsq(f,g)
    gD_emp,_ = debye_dos(c["theta_emp"], f)
    gD_ael,_ = debye_dos(c["theta_ael"], f)
    inv_emp = moment_invsq(f,gD_emp)
    inv_ael = moment_invsq(f,gD_ael)
    print(f"{m:>3} {inv_real:12.4f} {inv_emp:12.4f} {inv_ael:12.4f}  {inv_real/inv_emp:6.3f}")

print()
print("B_iso(295K) [A^2]  via full-quantum DOS integral, convention B=8pi^2<u^2>:")
print(f"{'mat':>3} {'B_realDOS':>10} {'B_Demp':>10} {'B_Dael':>10} {'B_Dagl':>10}")
for m,c in MAT.items():
    f,g = read_dos(apl_dos(c["dir"]))
    T=295.0
    u_real = msd_from_dos(f,g,T,c["mass"])
    gD_emp,_=debye_dos(c["theta_emp"],f); u_emp=msd_from_dos(f,gD_emp,T,c["mass"])
    gD_ael,_=debye_dos(c["theta_ael"],f); u_ael=msd_from_dos(f,gD_ael,T,c["mass"])
    gD_agl,_=debye_dos(c["theta_agl"],f); u_agl=msd_from_dos(f,gD_agl,T,c["mass"])
    k=8*math.pi**2*A2
    print(f"{m:>3} {k*u_real:10.4f} {k*u_emp:10.4f} {k*u_ael:10.4f} {k*u_agl:10.4f}")

print()
print("Where is the DOS weight? cumulative fraction below f:")
for m,c in MAT.items():
    f,g = read_dos(apl_dos(c["dir"]))
    cum=np.concatenate([[0],np.cumsum((g[1:]+g[:-1])/2*np.diff(f))])
    def fb(x): return np.interp(x,f,cum)
    fD_emp=KB*c["theta_emp"]/H/THZ
    print(f"{m:>3} f_D(emp)={fD_emp:6.2f}THz  frac<2THz={fb(2.0):.3f} "
          f"frac<f_D={fb(fD_emp):.3f}  f_max={f.max():.1f}THz")

print()
print("=== DECOMPOSE <f^-2> contribution by reduced-frequency window f/f_D "
      "(real DOS vs Debye-emp), as in SI Fig. S2(a) ===")
print("If optical peak were the cause, high-f window would dominate the EXCESS.")
for m,c in MAT.items():
    f,g = read_dos(apl_dos(c["dir"]))
    gD,fD = debye_dos(c["theta_emp"], f)
    x_edges=[0,0.15,0.3,0.6,1.0]
    edges=[x*fD for x in x_edges]+[max(f.max(),fD)+0.01]
    labels=[f"{a:g}-{b:g}" for a,b in zip(x_edges[:-1],x_edges[1:])]+[">1"]
    print(f"\n{m}: fD_emp={fD:.2f} THz, f_max={f.max():.1f} THz")
    print(f"  {'window f/fD':>14} {'real<f^-2>':>12} {'Debye<f^-2>':>12} {'excess':>11}")
    for lo,hi,lab in zip(edges[:-1],edges[1:],labels):
        mm=(f>=lo)&(f<hi)
        ir=np.trapezoid((g/f**2)[mm],f[mm]) if mm.sum()>=2 else 0.0
        # exact Debye integral: the Debye DOS can extend past the last DOS grid
        # point (C: f_D = 46.5 THz > f_max), which a grid trapezoid would miss
        id_=3.0*max(0.0,min(hi,fD)-lo)/fD**3
        print(f"  {lab:>14} {ir:12.4e} {id_:12.4e} {ir-id_:+11.3e}")

print()
print("=== Moment-matched Debye: what Theta_D would reproduce real <f^-2>? ===")
print("Compares 'Debye-Waller Theta' to the elastic/calorimetric Theta_D used.")
for m,c in MAT.items():
    f,g = read_dos(apl_dos(c["dir"]))
    inv_real = moment_invsq(f,g)
    # ideal Debye: <f^-2> = INT_0^fD 3f^2/fD^3 / f^2 df = 3/fD^2  => fD = sqrt(3/<f^-2>)
    fD_match = math.sqrt(3.0/inv_real)
    theta_match = H*fD_match*THZ/KB
    print(f"{m:>3}  Theta_DW(from <f^-2>)={theta_match:7.1f} K   "
          f"Theta_emp={c['theta_emp']:.0f}  Theta_AEL={c['theta_ael']:.0f}  "
          f"ratio Theta_DW/Theta_emp={theta_match/c['theta_emp']:.3f}")
