from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class RunResult:
    dt: float
    csv_path: Path
    df: pd.DataFrame


def _repo_root() -> Path:
    
    return Path(__file__).resolve().parents[1]


def _fmt_dt(dt: float) -> str:
    s = f"{dt:.10f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _run_fit_drag_model(repo: Path, dt: float) -> None:
    """
    Calls the existing module that uses RK4 and produces:
      - data/processed/effective_drag_fit.csv
    DT is passed via environment variable.
    """
    env = os.environ.copy()
    env["DT"] = str(dt)

    cmd = [sys.executable, "-m", "src.fit_drag_model"]
    print(f"\n[RUN] DT={dt:g}  ->  {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(repo), env=env, check=True)


def _load_effective_drag_fit(repo: Path) -> Path:
    p = repo / "data" / "processed" / "effective_drag_fit.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Expected output CSV not found: {p}\n"
            "Make sure `python -m src.fit_drag_model` runs successfully and writes this file."
        )
    return p


def _choose_key_column(df: pd.DataFrame) -> str:
    """
    Pick the x/index column (fin ratio) if present; else fallback to first column.
    """
    cols = list(df.columns)
    lowered = [c.lower() for c in cols]

    for i, c in enumerate(lowered):
        if "fin" in c and "ratio" in c:
            return cols[i]

    for i, c in enumerate(lowered):
        if "fin" in c:
            return cols[i]

    return cols[0]


def _numeric_columns(df: pd.DataFrame, key_col: str) -> List[str]:
    nums = list(df.select_dtypes(include=[np.number]).columns)
    return [c for c in nums if c != key_col]


def _pick_primary_metric(cols: List[str]) -> str:
    """
    Fallback headline metric selection if k_eff_kg_per_m is not present.
    """
    lowered = [c.lower() for c in cols]

    for i, c in enumerate(lowered):
        if "k_eff" in c:
            return cols[i]

    for i, c in enumerate(lowered):
        if "range_sim" in c:
            return cols[i]

    for i, c in enumerate(lowered):
        if "range" in c:
            return cols[i]

    return cols[0] if cols else ""


def _relative_change(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    |b-a| / |a| with safe handling near 0.
    """
    denom = a.abs().replace(0, np.nan)
    return (b - a).abs() / denom


def main() -> None:
    repo = _repo_root()

    dt0 = float(os.environ.get("DT", "0.003"))
    dts = [dt0, dt0 / 2.0, dt0 / 4.0]  # convergence ladder

    out_dir = repo / "figures" / "timestep_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: List[RunResult] = []

    # --- Run the model at each DT and snapshot the resulting effective_drag_fit.csv ---
    for dt in dts:
        _run_fit_drag_model(repo, dt)

        csv_path = _load_effective_drag_fit(repo)
        snap_csv = out_dir / f"effective_drag_fit_dt_{_fmt_dt(dt)}.csv"
        shutil.copy2(csv_path, snap_csv)

        df = pd.read_csv(snap_csv)
        runs.append(RunResult(dt=dt, csv_path=snap_csv, df=df))

    # --- Determine key column + numeric columns ---
    base_df = runs[0].df.copy()
    key_col = _choose_key_column(base_df)

    # Verify key_col exists everywhere
    for r in runs[1:]:
        if key_col not in r.df.columns:
            raise KeyError(
                f"Key column '{key_col}' not found in {r.csv_path.name}. "
                f"Columns: {list(r.df.columns)}"
            )

    numeric_cols = _numeric_columns(base_df, key_col)
    if not numeric_cols:
        raise ValueError(
            f"No numeric columns found to compare in {runs[0].csv_path.name}.\n"
            f"Columns: {list(runs[0].df.columns)}"
        )

    # Primary metric (used as fallback)
    primary_metric = "k_eff_kg_per_m" if "k_eff_kg_per_m" in numeric_cols else _pick_primary_metric(numeric_cols)

    # Metrics we want to print as headline lines (both if available)
    metrics_to_report: List[str] = []
    if "k_eff_kg_per_m" in numeric_cols:
        metrics_to_report.append("k_eff_kg_per_m")
    if "range_sim_m" in numeric_cols:
        metrics_to_report.append("range_sim_m")
    if not metrics_to_report and primary_metric:
        metrics_to_report.append(primary_metric)

    # --- Compare runs[1], runs[2] against baseline runs[0] ---
    base = runs[0].df.set_index(key_col).sort_index()

    summary: Dict[str, Dict[str, float]] = {}
    # dt_key -> {metric_name -> max_rel_change}
    headline: Dict[str, Dict[str, float]] = {}

    for r in runs[1:]:
        other = r.df.set_index(key_col).sort_index()

        common_idx = base.index.intersection(other.index)
        if len(common_idx) == 0:
            raise ValueError(
                f"No overlapping '{key_col}' values between baseline and {r.csv_path.name}. "
                "Check the CSVs are aligned on the same fin_ratio grid."
            )

        b = base.loc[common_idx]
        o = other.loc[common_idx]

        dt_key = f"dt_{_fmt_dt(r.dt)}"

        per_col_max_rel: Dict[str, float] = {}
        per_col_max_abs: Dict[str, float] = {}

        for col in numeric_cols:
            if col not in o.columns or col not in b.columns:
                continue

            rel = _relative_change(b[col], o[col])
            abschg = (o[col] - b[col]).abs()

            per_col_max_rel[col] = float(np.nanmax(rel.to_numpy()))
            per_col_max_abs[col] = float(np.nanmax(abschg.to_numpy()))

        # Store headline metrics for this dt
        headline[dt_key] = {m: per_col_max_rel[m] for m in metrics_to_report if m in per_col_max_rel}

        summary[dt_key] = {
            **{f"max_rel_{col}": v for col, v in per_col_max_rel.items()},
            **{f"max_abs_{col}": v for col, v in per_col_max_abs.items()},
            "n_common_points": int(len(common_idx)),
        }

    # --- Save JSON summary ---
    summary_path = out_dir / "timestep_sensitivity_summary.json"
    payload = {
        "baseline_dt": dt0,
        "tested_dts": dts,
        "key_column": key_col,
        "numeric_columns": numeric_cols,
        "primary_metric": primary_metric,
        "headline_max_relative_change": headline,
        "full_summary": summary,
    }
    summary_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[SAVED] {summary_path}")

    # --- Headline lines for README: explicitly dt0 -> dt0/2 ---
    dt_hi = dts[0]
    dt_lo = dts[1]
    dt_half_key = f"dt_{_fmt_dt(dt_lo)}"

    if dt_half_key not in headline or not headline[dt_half_key]:
        print("⚠️ No headline metrics found for the halved DT comparison (check columns).")
    else:
        for metric in metrics_to_report:
            rel_half = headline[dt_half_key].get(metric)
            if rel_half is None:
                continue
            pct_change = 100.0 * rel_half
            print(
                f"✅ Timestep stability check: halving DT from {dt_hi:g} to {dt_lo:g} "
                f"changed '{metric}' by at most {pct_change:.12f}% "
                f"(worst-case across fin ratios)."
            )

    print("\n✅ Timestep stability check completed. See JSON summary for details.")


if __name__ == "__main__":
    main()

