#!/usr/bin/env python3
"""Recompute every quantity quoted in the manuscript directly from the raw results.

Usage:
    python scripts/verify_manuscript_numbers.py results/

Prints each quantity beside the value reported in the paper and flags any
disagreement. This exists so that a reviewer can confirm the tables and the text
agree with the measurements without reading the notebook.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Values as printed in the manuscript. Edit only if the manuscript changes.
REPORTED = {
    "runs_total": 1548,
    "gpu_hours": 12.71,
    # E3 was re-run under the corrected step rule tau <= 0.9*min(1/r, 1/beta);
    # these are the values the manuscript reports from that re-run (pass A, K_max=64,
    # deterministic kernels), not the superseded tau <= 0.9*2/r figures.
    "e3_delivered_unnorm_T32": 1.6453,
    "e3_delivered_norm_T32": 27.93,
    "e3_matvec_unnorm_euler_T32": 64.00,
    "e3_matvec_norm_euler_T32": 50.00,
    "e3_matvec_norm_exact_T32": 72.00,
    "horizon_fraction_T32": 0.0514,
    # per-variant at T=32 under normalisation, which the main text quotes separately
    # because the pooled rows average two regimes
    "e3_matvec_T32_positive": 36.00,
    "e3_seconds_T32_positive_euler": 12.46,
    "e3_seconds_T32_positive_exact": 41.12,
    "e3_tau_T32_positive": 0.8889,
    "margin_min": 0.039,
    "cg_residual_max": 1.41e-6,
    "cheb_doubling_max": 2.03e-8,
    "cheb_certified": 132,
    "cheb_uncertified": 48,
    "n_exact_runs": 180,
    "friedman_p_hetero": 0.270,
    "critical_difference_hetero": 6.06,
    "opscale_min": 2.4e1,
    "opscale_max": 2.8e7,
}

TOL_REL = 5e-3


def check(name, computed, reported, fmt="{:.4g}", tol=TOL_REL):
    """tol is relative. Use a looser value for quantities the manuscript reports
    to two significant figures; comparing those at 0.5% would flag correct
    rounding as a discrepancy."""
    if reported is None:
        print(f"  {name:<34s} {fmt.format(computed):>12s}   (no reported value)")
        return True
    if reported == 0:
        ok = abs(computed) < 1e-12
    else:
        ok = abs(computed - reported) / abs(reported) <= tol
    flag = "ok " if ok else "MISMATCH"
    print(f"  {name:<34s} {fmt.format(computed):>12s} vs {fmt.format(reported):>12s}  {flag}")
    return ok


def _is_pass(root: Path, csv_path: Path, want_pass: str, want_cap: int,
             want_det: bool) -> bool:
    """True if the sidecar e3_cfl_summary for this CSV describes the wanted pass."""
    sidecar = root / csv_path.name.replace("e3_solver_", "e3_cfl_summary_").replace(".csv", ".json")
    if not sidecar.exists():
        return False
    s = json.loads(sidecar.read_text())
    if "min(1/r" not in str(s.get("step_rule", "")):
        return False
    if want_det and s.get("deterministic") != "strict":
        return False
    return s.get("pass", "A") == want_pass and int(s.get("max_euler_steps", 64)) == want_cap


def main(root: Path) -> int:
    csvs = {p.stem.split("_MAIN_")[0]: p for p in root.glob("*_MAIN_*.csv")
            if not p.stem.startswith("e3_solver")}
    # Several E3 passes may sit side by side (corrected-CFL pass A and pass B, and the
    # superseded pre-re-run file).  Pin the one the manuscript reports instead of
    # letting the glob order decide, and say so if it is missing.
    e3_pass_a = [p for p in root.glob("e3_solver_MAIN_*.csv")
                 if _is_pass(root, p, want_pass="A", want_cap=64, want_det=True)]
    if len(e3_pass_a) != 1:
        sys.exit(f"expected exactly one deterministic K_max=64 corrected-CFL E3 file, "
                 f"found {len(e3_pass_a)}: {[p.name for p in e3_pass_a]}")
    csvs["e3_solver"] = e3_pass_a[0]
    print(f"E3 source: {e3_pass_a[0].name}")
    if not csvs:
        print(f"No MAIN result files found in {root}. Run the notebook with "
              f"RUN_MODE='MAIN' first.")
        return 2

    e3 = pd.read_csv(csvs["e3_solver"])
    e5 = pd.read_csv(csvs["e5_classification"])
    stats = json.loads((root / next(p.name for p in root.glob("statistics_MAIN_*.json"))).read_text())
    verify = json.loads((root / "verification.json").read_text())

    passed = []
    print("\n== verification suite (all must be at machine precision) ==")
    for k, v in verify.items():
        limit = 1e-2 if "lambda_min" in k else (1e-5 if k.startswith("T7_shift_c") else 1e-9)
        if k == "T7_shift_covers":
            ok = v > 0
            print(f"  {k:<34s} {v:>12.3e}   {'ok ' if ok else 'MISMATCH'}")
        else:
            ok = v < limit
            print(f"  {k:<34s} {v:>12.3e}   {'ok ' if ok else 'MISMATCH'}")
        passed.append(ok)

    print("\n== discretization sweep (E3) ==")
    piv_d = e3.pivot_table(index=["scale_operator", "propagator"], columns="T",
                           values="delivered_T", aggfunc="mean")
    piv_m = e3.pivot_table(index=["scale_operator", "propagator"], columns="T",
                           values="n_matvec", aggfunc="mean")
    T = 32.0
    d_un = piv_d.loc[(False, "euler"), T]
    d_no = piv_d.loc[(True, "euler"), T]
    passed += [
        check("delivered T=32, unnormalised", d_un, REPORTED["e3_delivered_unnorm_T32"]),
        check("delivered T=32, normalised", d_no, REPORTED["e3_delivered_norm_T32"]),
        check("fraction of horizon delivered", d_un / T, REPORTED["horizon_fraction_T32"]),
        check("matvec T=32, norm euler", piv_m.loc[(True, "euler"), T],
              REPORTED["e3_matvec_norm_euler_T32"]),
        check("matvec T=32, norm exact", piv_m.loc[(True, "exact"), T],
              REPORTED["e3_matvec_norm_exact_T32"]),
    ]
    # The pooled explicit rows average a positive-definite and an indefinite variant,
    # which are limited by different arguments of the step rule.  The main text quotes
    # the positive-definite variant on its own, so check that separately.
    pos = e3[(e3.model == "learned_node_gauge") & e3.scale_operator & (e3["T"] == T)]
    passed += [
        check("matvec T=32, positive variant",
              pos[pos.propagator == "euler"]["n_matvec"].mean(),
              REPORTED["e3_matvec_T32_positive"]),
        check("realised step T=32, positive variant",
              pos[pos.propagator == "euler"]["euler_tau"].mean(),
              REPORTED["e3_tau_T32_positive"]),
        check("seconds T=32, positive euler",
              pos[pos.propagator == "euler"]["seconds"].mean(),
              REPORTED["e3_seconds_T32_positive_euler"]),
        check("seconds T=32, positive exact",
              pos[pos.propagator == "exact"]["seconds"].mean(),
              REPORTED["e3_seconds_T32_positive_exact"]),
    ]

    print("\n== stability and numerics (E5 runs) ==")
    margin = e5["margin"].dropna()
    passed += [
        check("minimum stability margin", margin.min(), REPORTED["margin_min"]),
        check("stability violations", float((margin < 0).sum()), 0.0, "{:.0f}"),
        check("max CG residual", e5["cg_residual"].dropna().max(),
              REPORTED["cg_residual_max"], "{:.3e}"),
        check("max Chebyshev doubling", e5["cheb_doubling"].dropna().max(),
              REPORTED["cheb_doubling_max"], "{:.3e}"),
    ]
    conv = e5["cheb_order_converged"].dropna()
    passed += [
        check("orders certified", float((conv == True).sum()),
              REPORTED["cheb_certified"], "{:.0f}"),
        check("orders at cap", float((conv == False).sum()),
              REPORTED["cheb_uncertified"], "{:.0f}"),
    ]

    print("\n== operator scale ==")
    osc = e5["operator_scale"].dropna()
    passed += [
        # reported to two significant figures in the manuscript
        check("Gershgorin radius, minimum", osc.min(), REPORTED["opscale_min"],
              "{:.3g}", tol=0.05),
        check("Gershgorin radius, maximum", osc.max(), REPORTED["opscale_max"],
              "{:.3g}", tol=0.05),
    ]

    print("\n== statistics ==")
    het = stats.get("heterophilic", {})
    passed += [
        check("Friedman p (heterophilic)", het.get("p", float("nan")),
              REPORTED["friedman_p_hetero"], "{:.3f}"),
        check("critical difference", het.get("cd", float("nan")),
              REPORTED["critical_difference_hetero"], "{:.2f}"),
    ]

    print("\n== scale ==")
    skipped = 0
    manifest = root.parent / "manifest.json"
    if manifest.exists():
        runs = json.loads(manifest.read_text())["runs"]
        secs = [v["rec"]["seconds"] for v in runs.values() if "rec" in v]
        passed += [
            check("total runs", float(len(runs)), REPORTED["runs_total"], "{:.0f}"),
            check("training hours", sum(secs) / 3600, REPORTED["gpu_hours"], "{:.2f}"),
        ]
    else:
        # The manifest sits beside results/, so an archive of results/ alone does not
        # carry it. Reconstruct the run total from the row counts instead, and say
        # which check that cannot replace rather than dropping it in silence.
        print(f"  manifest.json not found at {manifest.resolve()};")
        print("  reconstructing the run total from row counts instead")
        cfg = json.loads((root / "config.json").read_text())
        hp_path = next(root.glob("hp_MAIN_*.json"), None)
        n_hp = (len(json.loads(hp_path.read_text()))
                * int(cfg["hp_budget"]) * int(cfg["hp_val_splits"])) if hp_path else 0
        n_e4 = len(pd.read_csv(csvs["e4_stability"])) if "e4_stability" in csvs else 0
        total = len(e5) + n_hp + len(e3) + n_e4
        print(f"  rows: E5 grid {len(e5)}, equal-budget search {n_hp}, "
              f"E3 {len(e3)}, E4 {n_e4}")
        passed += [check("total runs", float(total), REPORTED["runs_total"], "{:.0f}")]
        # Not counted as a pass: without the manifest there is nothing to compare
        # the reported figure against, and a skipped check must not read as a
        # satisfied one.
        print(f"  {'training hours':<34s} {'not checkable':>12s} vs "
              f"{REPORTED['gpu_hours']:12.2f}  SKIPPED (needs manifest.json)")
        skipped += 1

    n_bad = sum(1 for x in passed if not x)
    print(f"\n{len(passed) - n_bad}/{len(passed)} checks passed"
          + (f", {skipped} skipped" if skipped else "")
          + ("" if n_bad == 0 else f"  --  {n_bad} MISMATCH"))
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "results")))
