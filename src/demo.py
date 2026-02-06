"""
Demo CLI for the fin-aero projectile model.

Examples (run from inside fin-aero-ee-repo):
  .venv/bin/python -m src.demo --fin-ratio 0.75
  .venv/bin/python -m src.demo --fin-ratio 0.75 --plot
  DT=0.00075 .venv/bin/python -m src.demo --fin-ratio 1.0 --plot
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import core research functions
from src.fit_drag_model import (
    THETA_DEG,
    ensure_dirs,
    load_corrected_ranges,
    estimate_v0_from_control,
    fit_k_for_target_range,
    simulate_range,
    rk4_step,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_FIG = ROOT / "outputs" / "figures"
OUT_DATA = ROOT / "data" / "processed"
EFFECTIVE_FIT_CSV = OUT_DATA / "effective_drag_fit.csv"


def _fmt_ratio(r: float) -> str:
    s = f"{r:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _find_row_by_ratio(df: pd.DataFrame, ratio: float, col: str = "fin_ratio") -> Optional[pd.Series]:
    if col not in df.columns:
        return None
    arr = df[col].to_numpy(dtype=float)
    idx = np.where(np.isclose(arr, ratio, atol=1e-12, rtol=0.0))[0]
    if idx.size == 0:
        return None
    return df.iloc[int(idx[0])]


def simulate_trajectory(
    v0: float,
    theta_deg: float,
    k_eff: float,
    dt: float,
    t_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return x(t), y(t) until landing (y crosses 0 after being positive).
    """
    import math

    theta = math.radians(theta_deg)
    vx0 = v0 * math.cos(theta)
    vy0 = v0 * math.sin(theta)

    state = np.array([0.0, 0.0, vx0, vy0], dtype=float)

    t = 0.0
    y_was_positive = False

    xs = [float(state[0])]
    ys = [float(state[1])]

    x_prev, y_prev = float(state[0]), float(state[1])

    while t < t_max:
        state = rk4_step(state, dt, k_eff)
        t += dt

        x, y = float(state[0]), float(state[1])
        xs.append(x)
        ys.append(y)

        if y > 0:
            y_was_positive = True

        if y_was_positive and (y_prev > 0.0) and (y <= 0.0):
            break

        x_prev, y_prev = x, y

    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo CLI: show model output for a given fin ratio.")
    p.add_argument("--fin-ratio", type=float, required=True, help="Fin length-to-diameter ratio (e.g., 0.75)")
    p.add_argument("--dt", type=float, default=None, help="Integrator timestep in seconds (overrides DT env var)")
    p.add_argument("--t-max", type=float, default=5.0, help="Max simulation time (s)")
    p.add_argument("--force-refit", action="store_true", help="Ignore effective_drag_fit.csv and refit k_eff from data")
    p.add_argument("--plot", action="store_true", help="Save a trajectory plot to outputs/figures/")
    return p.parse_args()


def main() -> None:
    ensure_dirs()

    args = parse_args()
    fin_ratio = float(args.fin_ratio)

    # dt: CLI > env var > default
    dt_env = os.environ.get("DT", "")
    dt_default = float(dt_env) if dt_env.strip() else 1e-3
    dt = float(args.dt) if args.dt is not None else dt_default

    # Load corrected ranges (observed)
    df = load_corrected_ranges()
    row_obs = _find_row_by_ratio(df, fin_ratio, col="fin_ratio")
    if row_obs is None:
        raise SystemExit(
            f"fin_ratio={fin_ratio} not found in corrected ranges. "
            f"Available: {sorted(df['fin_ratio'].unique().tolist())}"
        )

    R_obs = float(row_obs["avg_range_m_used"])
    v0 = estimate_v0_from_control(df)

    # Reuse effective_drag_fit.csv if available (instant demo)
    k_eff = None
    R_sim = None
    source = "refit"

    if (not args.force_refit) and EFFECTIVE_FIT_CSV.exists():
        fit_df = pd.read_csv(EFFECTIVE_FIT_CSV)
        row_fit = _find_row_by_ratio(fit_df, fin_ratio, col="fin_ratio")
        if row_fit is not None and "k_eff_kg_per_m" in fit_df.columns and "range_sim_m" in fit_df.columns:
            k_eff = float(row_fit["k_eff_kg_per_m"])
            R_sim = float(row_fit["range_sim_m"])
            source = "effective_drag_fit.csv"

    # Otherwise refit for this ratio only
    if k_eff is None or R_sim is None:
        k_eff = float(fit_k_for_target_range(v0, THETA_DEG, R_obs))
        R_sim = float(simulate_range(v0, THETA_DEG, k_eff, dt=dt, t_max=float(args.t_max)))
        source = "on-the-fly refit"

    residual = R_sim - R_obs

    print("\n=== Demo: fin-aero model ===")
    print(f"fin_ratio          = {fin_ratio:g}")
    print(f"v0 (from control)  = {v0:.6f} m/s")
    print(f"DT used            = {dt:.6g} s")
    print(f"k_eff              = {k_eff:.10g} kg/m   ({source})")
    print(f"Observed range     = {R_obs:.6f} m")
    print(f"Simulated range    = {R_sim:.6f} m")
    print(f"Residual (sim-obs) = {residual:.6f} m")

    if args.plot:
        xs, ys = simulate_trajectory(v0, THETA_DEG, k_eff, dt=dt, t_max=float(args.t_max))
        fig_path = OUT_FIG / f"demo_trajectory_fin_{_fmt_ratio(fin_ratio)}.png"

        plt.figure()
        plt.plot(xs, ys)
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title(f"Trajectory demo (fin_ratio={fin_ratio:g}, DT={dt:g}, k_eff={k_eff:.3g})")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

        print(f"Saved: {fig_path}")

    print()


if __name__ == "__main__":
    main()
