"""Vendored plotting/benchmark configuration for the DWF data package.

PROVENANCE
----------
Recovered verbatim from:
    20260305_new_plot.zip :: 20260305_new_plot/dwf_plot_config.py
    (archive dated 2026-03-06; originally outside this project tree)

`rebuild_dwf_revision_package.py` loads this file with `runpy.run_path` and
reads only the `ELEMENTS` dict.  It is the sole external input that the master
tables depend on, which is why it is vendored here: without it,
`descriptor_table.csv`, `bt_curves_full.csv`,
`bt_at_experiment_temperatures.csv`, `benchmark_metrics.csv`,
`low_frequency_phonon_summary.csv`, `material_method_ranking.csv`, and
`Bi_SOC_vs_noSOC_delta.csv` cannot be regenerated.

WHERE THE NUMBERS COME FROM
---------------------------
For each of As, Sb and Bi:

  exp_temps / exp_b              experimental Debye-Waller factors B(T)
  empirical_temps / empirical_b  the empirical Debye-model reference curve

Both series were **read off figure 3 by hand** (manual digitization) from:

    P Fischer, I Sosnowska and M Szymanski,
    "Debye-Waller factor and thermal expansion of arsenic, antimony and
    bismuth", J. Phys. C: Solid State Phys. 11 (1978) 1043.
    Received 8 August 1977, in final form 21 November 1977.

Fischer et al. do not tabulate B(T).  Their text states the values "are shown
in figure 3"; Table 1 gives the hexagonal z parameter, Table 3 gives Debye
temperatures from elastic constants, and Table 2 gives Bi temperature factors
at 293 K and 516 K only.  Table 2 is therefore the only independent
cross-check available -- see ../../data/literature/A7_literature_B_sources.csv.

KNOWN DIGITIZATION ARTIFACTS (retained deliberately -- do not silently fix)
--------------------------------------------------------------------------
  Sb  empirical_temps contains -1.4285714 K (a negative temperature).
      `sanitize_curve_points()` in rebuild_dwf_revision_package.py clamps any
      temperature in (-10, 0) to exactly 0 K and keeps its B value, so this
      point becomes B(0 K) = 0.27142857 A^2 while B(10 K) = 0.17678498 A^2 --
      i.e. a non-monotonic B(0) that is visible in
      Sb_comparison_main_curves_wide.csv.
  As  empirical_temps contains 2.9398968 K with B = 0.27746586 A^2, above the
      36.179334 K value of 0.18471171 A^2.  Positive, so not clamped; the
      non-monotonicity is retained as-is.

Both artifacts lie in the *empirical reference curve*, which the manuscript
benchmarks as one of the compared methods ("empirical digitized reference"),
not in the experimental anchor points.

The numerical content below is unmodified.  Editing it changes published
figures and benchmark tables.
"""

ELEMENTS = {
    "As": {
        "mass_amu": 74.92160,
        "theta_d": 249.135,
        "apl_file": "As_data",
        "exp_temps": [
            5.815831987,
            40.71082391,
            81.42164782,
            130.8562197,
            180.2907916,
            232.6332795,
            290.7915994,
            478.3521809,
            664.4588045,
            820.0323102,
        ],
        "exp_b": [
            0.434402332,
            0.510204082,
            0.457725948,
            0.679300292,
            0.760932945,
            1.069970845,
            1.029154519,
            1.443148688,
            2.131195335,
            2.516034985,
        ],
        "empirical_temps": [
            881.198711,
            740.836013,
            685.594855,
            522.347267,
            355.048553,
            257.553704,
            159.183987,
            120.098039,
            95.493923,
            65.112402,
            36.179334,
            17.363441,
            2.9398968,
        ],
        "empirical_b": [
            2.54335526,
            2.13872832,
            1.92485549,
            1.52023121,
            1.05780347,
            0.77456647,
            0.59867052,
            0.53936835,
            0.32369942,
            0.23694422,
            0.18471171,
            0.16184971,
            0.27746586,
        ],
    },
    "Bi": {
        "mass_amu": 208.98040,
        "theta_d": 115.1,
        "apl_file": "Bi_data",
        "exp_temps": [
            41.89944134,
            82.40223464,
            131.2849162,
            181.5642458,
            230.4469274,
            291.8994413,
            410.6145251,
            474.8603352,
            515.3631285,
            300.2793296,
            94.97206704,
        ],
        "exp_b": [
            0.067039106,
            0.240223464,
            0.30726257,
            0.575418994,
            0.776536313,
            0.983240223,
            1.167597765,
            1.491620112,
            1.357541899,
            1.150837989,
            0.335195531,
        ],
        "empirical_temps": [
            565.7142857,
            524.2857143,
            414.2857143,
            367.1428571,
            288.5714286,
            235.7142857,
            165.7142857,
            104.2857143,
            65.71428571,
            37.14285714,
            5.714285714,
        ],
        "empirical_b": [
            2.488636364,
            2.318181818,
            1.835227273,
            1.636363636,
            1.284090909,
            1.051136364,
            0.755681818,
            0.482954545,
            0.3125,
            0.198863636,
            0.130681818,
        ],
    },
    "Sb": {
        "mass_amu": 121.760,
        "theta_d": 206.693,
        "apl_file": "Sb_data",
        "exp_temps": [
            4.0,
            40.0,
            84.0,
            130.0,
            172.0,
            182.0,
            480.0,
            678.0,
            776.0,
            842.0,
            294.0,
            226.0,
        ],
        "exp_b": [
            0.25,
            0.298387097,
            0.379032258,
            0.60483871,
            0.669354839,
            0.790322581,
            1.758064516,
            2.540322581,
            2.935483871,
            3.201612903,
            1.096774194,
            0.903225806,
        ],
        "empirical_temps": [
            858.571429,
            800.0,
            727.142857,
            644.285714,
            532.857143,
            440.0,
            350.0,
            260.0,
            170.0,
            111.428571,
            55.7142857,
            20.0,
            -1.4285714,
        ],
        "empirical_b": [
            1.92714286,
            1.84,
            1.67428571,
            1.48571429,
            1.22857143,
            1.01714286,
            0.81714286,
            0.60571429,
            0.39428571,
            0.28857143,
            0.16571429,
            0.13142857,
            0.27142857,
        ],
    },
}

METHOD_ORDER = (
    "Empirical",
    "AEL",
    "APL_iso",
    "APL_parallel",
    "APL_perpendicular",
)

METHOD_LABELS = {
    "Empirical": "Empirical",
    "AEL": "AEL",
    "APL_iso": "APL (iso)",
    "APL_parallel": "APL parallel",
    "APL_perpendicular": "APL perpendicular",
}

METHOD_COLORS = {
    "Empirical": "#d97706",
    "AEL": "#2563eb",
    "APL_iso": "#dc2626",
    "APL_parallel": "#059669",
    "APL_perpendicular": "#7c3aed",
}
