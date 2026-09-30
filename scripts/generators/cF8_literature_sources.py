#!/usr/bin/env python3
"""Build data/literature/cF8_literature_B_sources.csv: the complete literature archive of
isotropic Debye-Waller B(T) for the diamond-structure controls (C, Si, Ge).

Every tabulated temperature point from each primary source is included (the SI
Fig. S1 generator curates a subset for plotting). Provenance audited 2026-06-03
against the primary-source PDFs.

Sources and the originally reported quantity:
  Beyer et al. 2023, Acta Cryst A79, 41   -- Table 1, Uiso (1e-4 A^2); B = 8 pi^2 Uiso
                                             (direct synchrotron PXRD ADP refinement)
  Spackman 1986, Acta Cryst A42, 271      -- Table 1 / text, mean B (direct X-ray refinement)
  Sang et al. 2010, Acta Cryst A66, 685   -- Table 1, B (direct CBED measurement)
  Peng et al. 1996, Acta Cryst A52, 456   -- deposited Supplement Table 1 B column
                                             (from experimental phonon DOS)
  Batterman & Chipman 1962, Phys Rev 127, 690 -- Table I,
                                             Theta_M(Si)=543(8) K and
                                             Theta_M(Ge)=290(5) K; B(T) DERIVED
                                             here from Theta_M (NOT a direct B)
  Reid & Pirie 1980, Acta Cryst A36, 957  -- Tables 1/2, lattice-dynamical MODEL calc
                                             (bond-charge column used as representative)

Conversions:
  B = 8 pi^2 Uiso
  Theta_M-derived: B(T) = (6 h^2 / (m kB Theta_M)) * (phi(x)/x + 1/4), x = Theta_M/T,
    phi(x) = (1/x) int_0^x t/(e^t - 1) dt, m = atomic mass.
"""
from __future__ import annotations
import csv, math
from pathlib import Path

# Every path resolves through scripts/paths.py, the single source of truth for
# this package. Nothing outside the repository root is read or written.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import LITERATURE, PROCESSED, VERIFICATION_LOGS  # noqa: E402
OUT = LITERATURE / "cF8_literature_B_sources.csv"

EIGHT_PI2 = 8.0 * math.pi**2
H = 6.62607015e-34
KB = 1.380649e-23
AMU = 1.66053906660e-27

# ---------------------------------------------------------------- Beyer 2023 (C)
# Table 1: (label, T_K, Uiso_1e-4, eUiso_1e-4). "300 K-after" is the post-anneal
# repeat at the 300 K set point.
BEYER_C = [
    ("100 K",      100, 16.54, 0.19), ("200 K",      200, 17.50, 0.12),
    ("300 K",      300, 18.02, 0.19), ("300 K-after", 300, 18.55, 0.19),
    ("400 K",      400, 19.83, 0.20), ("500 K",      500, 22.51, 0.19),
    ("600 K",      600, 24.57, 0.20), ("700 K",      700, 26.91, 0.18),
    ("800 K",      800, 30.74, 0.15), ("900 K",      900, 33.73, 0.22),
    ("1000 K",    1000, 36.63, 0.24),
]

# ------------------------------------------------------------- Peng 1996 (B col)
# Deposited supplement zh0008_82472sup1.pdf, Table 1. This is the full B(T)
# table; the main-paper/supplement absorptive-factor Tables 6/10/24 repeat
# subsets of the same pDOS-derived B values.
PENG_C = [
    (1,0.1286), (2,0.1286), (3,0.1286), (4,0.1286), (5,0.1286),
    (6,0.1286), (7,0.1286), (8,0.1286), (9,0.1286), (10,0.1286),
    (15,0.1286), (20,0.1287), (25,0.1287), (30,0.1288), (40,0.1289),
    (50,0.1291), (60,0.1292), (65,0.1294), (70,0.1295), (75,0.1296),
    (77,0.1296), (80,0.1297), (85,0.1299), (90,0.1300), (100,0.1303),
    (110,0.1307), (120,0.1311), (130,0.1315), (140,0.1320), (150,0.1325),
    (160,0.1330), (170,0.1336), (180,0.1342), (190,0.1349), (200,0.1355),
    (210,0.1363), (220,0.1370), (230,0.1378), (240,0.1386), (250,0.1394),
    (260,0.1403), (270,0.1412), (280,0.1422), (290,0.1432), (293,0.1435),
    (295,0.1437), (300,0.1442), (310,0.1453), (320,0.1463), (330,0.1475),
    (340,0.1486), (350,0.1498), (375,0.1530),
]
PENG_SI = [
    (1,0.1915), (2,0.1915), (3,0.1915), (4,0.1915), (5,0.1916),
    (6,0.1916), (7,0.1916), (8,0.1917), (9,0.1918), (10,0.1918),
    (15,0.1923), (20,0.1929), (25,0.1937), (30,0.1946), (40,0.1971),
    (50,0.2003), (60,0.2042), (65,0.2064), (70,0.2088), (75,0.2113),
    (77,0.2124), (80,0.2141), (85,0.2170), (90,0.2201), (100,0.2268),
    (110,0.2342), (120,0.2423), (130,0.2511), (140,0.2607), (150,0.2709),
    (160,0.2648), (170,0.2814), (180,0.2979), (190,0.3145), (200,0.3310),
    (210,0.3476), (220,0.3641), (230,0.3807), (240,0.3972), (250,0.4138),
    (260,0.4303), (270,0.4469), (280,0.4634), (290,0.4799), (293,0.4849),
    (295,0.4882), (300,0.4965), (310,0.5130), (320,0.5296), (330,0.5461),
    (340,0.5626), (350,0.5792), (375,0.6205), (400,0.6619), (450,0.7445),
    (500,0.8271), (600,0.9923), (700,1.1573), (800,1.3222), (900,1.4869),
    (1000,1.6514),
]
PENG_GE = [
    (1,0.1341), (2,0.1342), (3,0.1342), (4,0.1342), (5,0.1343),
    (6,0.1343), (7,0.1344), (8,0.1344), (9,0.1345), (10,0.1346),
    (15,0.1352), (20,0.1360), (25,0.1370), (30,0.1382), (40,0.1414),
    (50,0.1455), (60,0.1505), (65,0.1533), (70,0.1564), (75,0.1597),
    (77,0.1611), (80,0.1632), (85,0.1670), (90,0.1844), (100,0.2049),
    (110,0.2254), (120,0.2458), (130,0.2663), (140,0.2868), (150,0.3073),
    (160,0.3278), (170,0.3483), (180,0.3687), (190,0.3892), (200,0.4097),
    (210,0.4302), (220,0.4506), (230,0.4711), (240,0.4915), (250,0.5120),
    (260,0.5325), (270,0.5529), (280,0.5734), (290,0.5939), (293,0.6000),
    (295,0.6041), (300,0.6143), (310,0.6348), (320,0.6552), (330,0.6756),
    (340,0.6961), (350,0.7165), (375,0.7676), (400,0.8187), (450,0.9207),
    (500,1.0227), (600,1.2263), (700,1.4293), (800,1.6318), (900,1.8336),
    (1000,2.0345),
]

# ------------------------------------------------------------------ Sang 2010 (Si)
# Table 1 (T_K, B, eB). 300 K uses the t>160 nm value, NOT the conclusion's 0.5063.
SANG_SI = [(96,0.2707,0.0162),(173,0.3476,0.0171),(300,0.4833,0.0110)]

# --------------------------------------------------------------- Spackman 1986 (Si)
# mean B = 0.4632(11); data at/converted to 293-298 K (use 295 K as nominal RT).
SPACKMAN_SI = [(295,0.4632,0.0011)]

# --------------------------------------- Reid & Pirie 1980 (bond-charge model col)
# Representative model column; the per-temperature multi-model spread is in notes.
RP_C = [(1,0.1277),(5,0.1277),(10,0.1277),(20,0.1278),(40,0.1279),(60,0.1282),
        (80,0.1286),(100,0.1291),(150,0.1312),(200,0.1344),(250,0.1389),
        (295,0.1439),(350,0.1511),(400,0.1586),(500,0.1757),(600,0.1949),
        (700,0.2155),(800,0.2373),(900,0.2598),(1000,0.2828)]
RP_SI = [(1,0.1881),(5,0.1882),(10,0.1883),(20,0.1891),(40,0.1934),(60,0.2030),
         (80,0.2172),(100,0.2349),(150,0.2882),(200,0.3493),(250,0.4147),
         (295,0.4758),(350,0.5522),(400,0.6230),(500,0.7665),(600,0.9117),
         (700,1.0579),(800,1.2047),(900,1.3520),(1000,1.4996)]
RP_GE = [(1,0.1282),(5,0.1282),(10,0.1286),(20,0.1307),(40,0.1434),(60,0.1648),
         (80,0.1910),(100,0.2200),(150,0.2994),(200,0.3837),(250,0.4703),
         (295,0.5493),(350,0.6468),(400,0.7359),(500,0.9150),(600,1.0948),
         (700,1.2750),(800,1.4555),(900,1.6361),(1000,1.8168)]
# room-temperature multi-model spread (all models, 295 K) for the notes field
RP_SPREAD = {"C": "0.1439-0.1451", "Si": "0.4507-0.5192", "Ge": "0.5160-0.6015"}

# ----------------------------- Batterman 1962 (Si/Ge, amplitude Debye temperatures)
M_SI = 28.085         # amu
M_GE = 72.63          # amu
THETA_M_SI = 543.0    # K (Table I); uncertainty +/- 8 K
THETA_M_GE = 290.0    # K (Table I); uncertainty +/- 5 K
BATT_GRID = [80,100,150,200,250,290,293,295,298,300,350,400,500,600,700]


def _phi(x: float) -> float:
    """phi(x) = (1/x) int_0^x t/(e^t-1) dt via fine trapezoid."""
    n = 200_000
    dt = x / n
    s = 0.0
    for i in range(1, n):
        t = i * dt
        s += t / (math.exp(t) - 1.0)
    return (s * dt) / x


def b_from_theta_m(T: float, theta: float, mass_amu: float) -> float:
    x = theta / T
    m = mass_amu * AMU
    val = (6.0 * H**2 / (m * KB * theta)) * (_phi(x) / x + 0.25)
    return val * 1e20  # m^2 -> A^2


HEADER_COMMENT = """\
# Complete literature archive of isotropic Debye-Waller B(T) for the
# diamond-structure controls (C, Si, Ge). Every tabulated temperature point from
# each primary source is included; SI Fig. S1 plots a curated subset.
# Provenance audited 2026-06-03 against the primary-source PDFs.
#
# source_type: diffraction_refinement | cbed_direct | pdos_derived
#            | thetaM_derived | lattice_dynamical_calc
# Conversions: B = 8 pi^2 Uiso (Beyer); thetaM_derived B(T) from
#   Theta_M(Si)=543(8) K or Theta_M(Ge)=290(5) K via
#   B(T)=(6 h^2/(m kB Theta_M))*(phi(x)/x + 1/4), x=Theta_M/T.
#
# flag column: "" for a clean point, or "artifact" for a value retained verbatim
# from the source table that is physically impossible (B must rise with T). These
# are kept for archival fidelity but excluded from Fig. S1; the notes column gives
# the rationale per point.
"""
COLS = ["material","T_K","B_A2","B_unc_A2","source_key","source_label",
        "source_type","page_table","quantity","conversion","flag","notes"]

# Points kept verbatim from a source table but flagged as physically impossible
# (non-monotonic B vs T). Keyed by (material, source_key, T_K). The deposited Peng
# 1996 supplement is a poor-quality microfilm scan; a handful of cells are mis-scanned.
ARTIFACT_POINTS = {
    ("Si", "Peng1996", 160.0):
        "non-monotonic in deposited Peng supplement (150 K=0.2709, 170 K=0.2814); "
        "likely microfilm scan artifact, ~0.276 by interpolation; excluded from Fig. S1",
}


def _flag(material, source_key, T):
    return "artifact" if (material, source_key, float(T)) in ARTIFACT_POINTS else ""


def _artifact_note(material, source_key, T, default):
    return ARTIFACT_POINTS.get((material, source_key, float(T)), default)


def main() -> None:
    rows = []

    # --- Beyer 2023 (C) direct refinement
    for label, T, u, eu in BEYER_C:
        rows.append(dict(
            material="C", T_K=T, B_A2=f"{EIGHT_PI2*u*1e-4:.4f}",
            B_unc_A2=f"{EIGHT_PI2*eu*1e-4:.4f}", source_key="Beyer2023",
            source_label="Beyer et al. 2023 (Acta Cryst A79 41)",
            source_type="diffraction_refinement", page_table="Table 1",
            quantity=f"Uiso={u}({int(round(eu*100))})e-4 A^2", conversion="B=8pi^2 Uiso",
            flag="",
            notes=f"nominal set-point T ({label}); calibrated T in their Table S1"))

    # --- Peng 1996 (pDOS-derived)
    for mat, tbl, data in [("C","Supplement Table 1",PENG_C),
                           ("Si","Supplement Table 1",PENG_SI),
                           ("Ge","Supplement Table 1",PENG_GE)]:
        for T, B in data:
            default = "from experimentally determined phonon DOS; accuracy ~2-3%"
            rows.append(dict(
                material=mat, T_K=T, B_A2=f"{B:.4f}", B_unc_A2="",
                source_key="Peng1996",
                source_label="Peng et al. 1996 (Acta Cryst A52 456)",
                source_type="pdos_derived", page_table=f"{tbl} (B column)",
                quantity=f"B={B:.4f} A^2", conversion="none (direct B)",
                flag=_flag(mat, "Peng1996", T),
                notes=_artifact_note(mat, "Peng1996", T, default)))

    # --- Spackman 1986 (Si) direct refinement
    for T, B, eB in SPACKMAN_SI:
        rows.append(dict(
            material="Si", T_K=T, B_A2=f"{B:.4f}", B_unc_A2=f"{eB:.4f}",
            source_key="Spackman1986",
            source_label="Spackman 1986 (Acta Cryst A42 271)",
            source_type="diffraction_refinement", page_table="Table 1 + text p.279",
            quantity=f"mean B={B}({int(round(eB*1e4))}) A^2", conversion="none (direct B)",
            flag="",
            notes="X-ray pseudoatom refinement; data at/converted to 293-298 K (295 K nominal)"))

    # --- Sang 2010 (Si) CBED direct
    for T, B, eB in SANG_SI:
        note = "CBED four-beam measurement"
        if T == 300:
            note += "; Table 1 value (t>160 nm), NOT the conclusion's ambiguous 0.5063"
        rows.append(dict(
            material="Si", T_K=T, B_A2=f"{B:.4f}", B_unc_A2=f"{eB:.4f}",
            source_key="Sang2010", source_label="Sang et al. 2010 (Acta Cryst A66 685)",
            source_type="cbed_direct", page_table="Table 1",
            quantity=f"B={B}({int(round(eB*1e4))}) A^2", conversion="none (direct B)",
            flag="",
            notes=note))

    # --- Batterman 1962 (Si and Ge) Theta_M-derived
    for mat, theta_m, mass, uncertainty in [
        ("Si", THETA_M_SI, M_SI, 8),
        ("Ge", THETA_M_GE, M_GE, 5),
    ]:
        for T in BATT_GRID:
            B = b_from_theta_m(T, theta_m, mass)
            rows.append(dict(
                material=mat, T_K=T, B_A2=f"{B:.4f}", B_unc_A2="",
                source_key="Batterman1962",
                source_label="Batterman & Chipman 1962 (Phys Rev 127 690)",
                source_type="thetaM_derived",
                page_table=f"Table I (Theta_M={theta_m:.0f}({uncertainty}) K)",
                quantity=f"Theta_M({mat})={theta_m:.0f}({uncertainty}) K",
                conversion="B(T) from Theta_M Debye-Waller formula",
                flag="",
                notes="DERIVED, not direct B; paper reports Theta_M only"))

    # --- Reid & Pirie 1980 (lattice-dynamical model calc, archival)
    for mat, tbl, data in [("C","Table 2 (diamond)",RP_C),
                           ("Si","Table 1",RP_SI),
                           ("Ge","Table 2 (germanium)",RP_GE)]:
        for T, B in data:
            note = "model calc (bond-charge column)"
            if T == 295:
                note += f"; 295K model spread {RP_SPREAD[mat]}"
            rows.append(dict(
                material=mat, T_K=T, B_A2=f"{B:.4f}", B_unc_A2="",
                source_key="ReidPirie1980",
                source_label="Reid & Pirie 1980 (Acta Cryst A36 957)",
                source_type="lattice_dynamical_calc", page_table=f"{tbl}, bond-charge",
                quantity=f"B={B:.4f} A^2 (model)", conversion="none (model)",
                flag="",
                notes=note))

    order = {"C": 0, "Si": 1, "Ge": 2}
    rows.sort(key=lambda r: (order[r["material"]], r["source_key"], float(r["T_K"])))

    with open(OUT, "w", newline="") as fh:
        fh.write(HEADER_COMMENT)
        w = csv.DictWriter(fh, fieldnames=COLS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {OUT} ({len(rows)} points)")


if __name__ == "__main__":
    main()
