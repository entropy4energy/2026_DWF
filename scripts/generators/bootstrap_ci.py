#!/usr/bin/env python3
"""
Monte Carlo confidence intervals on the benchmark RMSE, NRMSE and R^2 (SI Table S2).

Each (material, method) row of benchmark_metrics.csv is re-scored N_DRAWS times,
with the experimental anchor B^exp(T_i) perturbed by independent Gaussian noise
N(0, sigma^2) and the model curve B^model(T_i) held fixed. The 2.5 and 97.5
percentiles of each metric give its 95% interval.

The two sigmas, 0.05 and 0.10 A^2, round outward the only standard uncertainties
Fischer (1978) reports: about 0.06 A^2 for Bi at 293 K and 0.09 A^2 at 516 K.
He gives none for As or Sb.

Why perturb the anchor rather than bootstrap the temperature points: with
N = 10..22 points, a nonparametric bootstrap drops about 37% of them in each
draw, so the R^2 interval would reflect the sample size rather than the
uncertainty of the measurement.

Output
------
  data/processed/benchmark_metrics_with_ci.csv
      benchmark_metrics.csv plus, for each sigma (tags sigma050, sigma100),
      {RMSE,NRMSE,R2}_{lo,hi} and {RMSE,R2}_med.

Reproducibility
---------------
SEED = 20260429, N_DRAWS = 10_000. Before anything is written, every row is
re-scored without noise and checked against benchmark_metrics.csv to within
SANITY_TOL. The SHA-256 of both inputs is printed to the log.

Usage
-----
    python3 scripts/generators/bootstrap_ci.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -- locations ---------------------------------------------------------------
# Both inputs and the output live at data/processed/.

SCRIPT_DIR = Path(__file__).resolve().parent
# Every path resolves through scripts/paths.py, the single source of truth
# for this package.
sys.path.insert(0, str(SCRIPT_DIR.parent))
from paths import PROCESSED  # noqa: E402

CSV_DIR = PROCESSED

INPUT_BT = CSV_DIR / "bt_at_experiment_temperatures.csv"
INPUT_METRICS = CSV_DIR / "benchmark_metrics.csv"

OUTPUT_METRICS_CI = CSV_DIR / "benchmark_metrics_with_ci.csv"

# -- knobs -------------------------------------------------------------------

# Anchor-noise sigmas, A^2: an outward rounding of Fischer's 0.06-0.09 for Bi.
SIGMAS = (0.05, 0.10)
N_DRAWS = 10_000
SEED = 20260429

# Sanity-check tolerance for matching original benchmark_metrics.csv at sigma=0.
SANITY_TOL = 1e-6

# Method-id to bt-column-stub mapping (must match
# rebuild_dwf_revision_package.py METHOD_LABELS / column_stub convention).
METHOD_TO_STUB = {
    "empirical digitized reference":                       "empirical_digitized_reference",
    "elastic-derived Debye":                               "elastic_derived_debye",
    "quasi-harmonic Debye":                                "quasi_harmonic_debye",
    "harmonic phonon spectrum (isotropic)":                "harmonic_phonon_spectrum_isotropic",
}

# The basal-plane and c-axis harmonic components were once scored here, one row
# each after the isotropic harmonic row. Fischer's B is isotropic, so they no
# longer are, but their noise draws are still taken: the random stream, and with
# it every interval in SI Table S2, stays as published.
RETIRED_AFTER = "harmonic phonon spectrum (isotropic)"
RETIRED_ROWS = 2


# -- core --------------------------------------------------------------------


def evaluate_model(exp_values: np.ndarray, predicted_values: np.ndarray) -> dict:
    """Mirror of evaluate_model() in rebuild_dwf_revision_package.py.

    Kept as an exact copy (not import) so this script is independent of the
    rebuild pipeline path layout, and so future drift in the rebuild script
    does not silently change the meaning of CI columns produced here.
    """
    exp_values = np.asarray(exp_values, dtype=float)
    predicted_values = np.asarray(predicted_values, dtype=float)
    mask = np.isfinite(exp_values) & np.isfinite(predicted_values)
    n_points = int(np.sum(mask))
    if n_points < 2:
        return {"N_points": n_points, "RMSE": np.nan, "MAE": np.nan,
                "NRMSE": np.nan, "R2": np.nan}
    exp_values = exp_values[mask]
    predicted_values = predicted_values[mask]
    residual = predicted_values - exp_values
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    exp_range = float(np.max(exp_values) - np.min(exp_values))
    nrmse = rmse / exp_range if exp_range > 0.0 else 0.0
    ss_tot = float(np.sum((exp_values - np.mean(exp_values)) ** 2))
    ss_res = float(np.sum((exp_values - predicted_values) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return {"N_points": n_points, "RMSE": rmse, "MAE": mae,
            "NRMSE": nrmse, "R2": r2}


def mc_ci(exp_values: np.ndarray,
          predicted_values: np.ndarray,
          sigma: float,
          n_draws: int,
          rng: np.random.Generator) -> dict:
    """Monte Carlo: perturb B_exp by N(0, sigma^2); recompute RMSE/NRMSE/R^2."""
    exp_values = np.asarray(exp_values, dtype=float)
    predicted_values = np.asarray(predicted_values, dtype=float)
    n = len(exp_values)
    rmse_d = np.empty(n_draws)
    nrmse_d = np.empty(n_draws)
    r2_d = np.empty(n_draws)
    # Vectorize the noise draw, but loop the metric eval (cheap enough at
    # N=10..22 and N_DRAWS=10000; clarity > perf).
    noise_block = rng.normal(0.0, sigma, size=(n_draws, n))
    for d in range(n_draws):
        m = evaluate_model(exp_values + noise_block[d], predicted_values)
        rmse_d[d] = m["RMSE"]
        nrmse_d[d] = m["NRMSE"]
        r2_d[d] = m["R2"]
    return {
        "RMSE_lo":  float(np.percentile(rmse_d, 2.5)),
        "RMSE_hi":  float(np.percentile(rmse_d, 97.5)),
        "RMSE_med": float(np.median(rmse_d)),
        "NRMSE_lo": float(np.percentile(nrmse_d, 2.5)),
        "NRMSE_hi": float(np.percentile(nrmse_d, 97.5)),
        "R2_lo":    float(np.percentile(r2_d, 2.5)),
        "R2_hi":    float(np.percentile(r2_d, 97.5)),
        "R2_med":   float(np.median(r2_d)),
    }


def sigma_tag(sigma: float) -> str:
    """Encode sigma as 'sigmaXXX' where XXX = sigma * 1000 to 3 digits.

    e.g. 0.05 -> 'sigma050', 0.10 -> 'sigma100'. Avoids float strings in headers.
    """
    return f"sigma{int(round(sigma * 1000)):03d}"


# -- main --------------------------------------------------------------------


def main() -> int:
    print("[bootstrap_ci] start", file=sys.stderr)
    print(f"[bootstrap_ci] script dir   : {SCRIPT_DIR}", file=sys.stderr)
    print(f"[bootstrap_ci] csv dir      : {CSV_DIR}", file=sys.stderr)
    print(f"[bootstrap_ci] N_DRAWS={N_DRAWS}, SIGMAS={SIGMAS}, SEED={SEED}",
          file=sys.stderr)

    if not INPUT_BT.exists():
        print(f"[bootstrap_ci] FATAL: missing input {INPUT_BT}", file=sys.stderr)
        return 2
    if not INPUT_METRICS.exists():
        print(f"[bootstrap_ci] FATAL: missing input {INPUT_METRICS}", file=sys.stderr)
        return 2

    bt_sha = hashlib.sha256(INPUT_BT.read_bytes()).hexdigest()
    metrics_sha = hashlib.sha256(INPUT_METRICS.read_bytes()).hexdigest()
    print(f"[bootstrap_ci] bt SHA-256       : {bt_sha}", file=sys.stderr)
    print(f"[bootstrap_ci] metrics SHA-256  : {metrics_sha}", file=sys.stderr)

    bt = pd.read_csv(INPUT_BT)
    metrics_orig = pd.read_csv(INPUT_METRICS)
    print(f"[bootstrap_ci] bt rows={bt.shape[0]}  cols={bt.shape[1]}",
          file=sys.stderr)
    print(f"[bootstrap_ci] metrics rows={metrics_orig.shape[0]}  "
          f"cols={metrics_orig.shape[1]}", file=sys.stderr)
    print(f"[bootstrap_ci] materials = {sorted(bt['material'].unique())}",
          file=sys.stderr)

    rng = np.random.default_rng(SEED)

    rows_out = []
    sanity_failures: list[str] = []
    point_outside_ci: list[str] = []

    for _, row in metrics_orig.iterrows():
        material = row["material"]
        method = row["method"]
        method_stub = METHOD_TO_STUB.get(method)
        if method_stub is None:
            print(f"[bootstrap_ci] WARN unknown method '{method}' material={material} "
                  f"-- skipping", file=sys.stderr)
            continue
        sub = bt[bt["material"] == material]
        if sub.empty:
            print(f"[bootstrap_ci] WARN no bt rows for material '{material}' "
                  f"-- skipping {method}", file=sys.stderr)
            continue
        col_pred = f"B_{method_stub}_A2"
        if col_pred not in sub.columns:
            print(f"[bootstrap_ci] WARN column '{col_pred}' missing for {material} "
                  f"-- skipping {method}", file=sys.stderr)
            continue

        exp_values = sub["B_exp_A2"].to_numpy(dtype=float)
        pred_values = sub[col_pred].to_numpy(dtype=float)

        # sanity-check vs original benchmark_metrics.csv at sigma=0
        check = evaluate_model(exp_values, pred_values)
        for key in ("N_points", "RMSE", "MAE", "NRMSE", "R2"):
            if key == "N_points":
                if check[key] != int(row[key]):
                    sanity_failures.append(
                        f"{material}/{method}/N_points "
                        f"rebuilt={check[key]} csv={int(row[key])}"
                    )
            else:
                if abs(check[key] - float(row[key])) > SANITY_TOL:
                    sanity_failures.append(
                        f"{material}/{method}/{key} "
                        f"rebuilt={check[key]:.10f} "
                        f"csv={float(row[key]):.10f} "
                        f"diff={abs(check[key]-float(row[key])):.2e}"
                    )

        out_row = {
            "material":   material,
            "soc_status": row["soc_status"],
            "method":     method,
            "N_points":   int(row["N_points"]),
            "RMSE":       float(row["RMSE"]),
            "MAE":        float(row["MAE"]),
            "NRMSE":      float(row["NRMSE"]),
            "R2":         float(row["R2"]),
        }
        for sigma in SIGMAS:
            ci = mc_ci(exp_values, pred_values, sigma, N_DRAWS, rng)
            tag = sigma_tag(sigma)
            out_row[f"RMSE_lo_{tag}"]  = ci["RMSE_lo"]
            out_row[f"RMSE_hi_{tag}"]  = ci["RMSE_hi"]
            out_row[f"RMSE_med_{tag}"] = ci["RMSE_med"]
            out_row[f"NRMSE_lo_{tag}"] = ci["NRMSE_lo"]
            out_row[f"NRMSE_hi_{tag}"] = ci["NRMSE_hi"]
            out_row[f"R2_lo_{tag}"]    = ci["R2_lo"]
            out_row[f"R2_hi_{tag}"]    = ci["R2_hi"]
            out_row[f"R2_med_{tag}"]   = ci["R2_med"]

            if not (ci["RMSE_lo"] <= float(row["RMSE"]) <= ci["RMSE_hi"]):
                point_outside_ci.append(
                    f"{material}/{method} sigma={sigma}: RMSE point "
                    f"{float(row['RMSE']):.4f} outside CI "
                    f"[{ci['RMSE_lo']:.4f}, {ci['RMSE_hi']:.4f}]"
                )
            if not (ci["R2_lo"] <= float(row["R2"]) <= ci["R2_hi"]):
                point_outside_ci.append(
                    f"{material}/{method} sigma={sigma}: R2 point "
                    f"{float(row['R2']):.4f} outside CI "
                    f"[{ci['R2_lo']:.4f}, {ci['R2_hi']:.4f}]"
                )

        rows_out.append(out_row)
        print(
            f"[bootstrap_ci]   {material:8s} {method[:46]:46s}  "
            f"RMSE={float(row['RMSE']):.3f} "
            f"CI@.05=[{out_row['RMSE_lo_sigma050']:.3f},"
            f"{out_row['RMSE_hi_sigma050']:.3f}]  "
            f"R2={float(row['R2']):+.3f} "
            f"CI@.05=[{out_row['R2_lo_sigma050']:+.3f},"
            f"{out_row['R2_hi_sigma050']:+.3f}]",
            file=sys.stderr,
        )

        if method == RETIRED_AFTER:
            for _ in range(RETIRED_ROWS):
                for sigma in SIGMAS:
                    rng.normal(0.0, sigma, size=(N_DRAWS, len(exp_values)))

    if sanity_failures:
        print(f"[bootstrap_ci] FATAL: {len(sanity_failures)} sanity-check failure(s) "
              f"vs original benchmark_metrics.csv (tol={SANITY_TOL:.0e}):",
              file=sys.stderr)
        for msg in sanity_failures:
            print(f"[bootstrap_ci]   - {msg}", file=sys.stderr)
        print("[bootstrap_ci] aborting before writing any output", file=sys.stderr)
        return 1

    if point_outside_ci:
        # Not necessarily fatal -- median of MC distribution can drift away
        # from the unperturbed point estimate, especially for highly nonlinear
        # transforms like R^2 with small N. Just log loudly.
        print(f"[bootstrap_ci] NOTE: {len(point_outside_ci)} (material,method,sigma) "
              f"combos where the unperturbed point estimate falls outside "
              f"the percentile CI:", file=sys.stderr)
        for msg in point_outside_ci:
            print(f"[bootstrap_ci]   * {msg}", file=sys.stderr)

    df_out = pd.DataFrame(rows_out)
    OUTPUT_METRICS_CI.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_METRICS_CI, index=False, float_format="%.8f")
    print(f"[bootstrap_ci] wrote {OUTPUT_METRICS_CI}  "
          f"({df_out.shape[0]} rows x {df_out.shape[1]} cols)",
          file=sys.stderr)

    print("[bootstrap_ci] DONE", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
