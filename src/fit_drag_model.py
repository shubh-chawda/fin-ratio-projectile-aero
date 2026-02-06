"""
Fit an effective quadratic-drag parameter k_eff for each fin ratio by matching measured range.

Run:
    python -m src.fit_drag_model

Outputs:
    outputs/figures/effective_drag_k_vs_fin.png
    outputs/figures/drag_model_range_fit.png
    data/processed/effective_drag_fit.csv
"""
from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT_FIG = ROOT / "outputs" / "figures"
OUT_DATA = ROOT / "data" / "processed"

# Physical constants 
G = 9.81           # m/s^2
MASS = 0.250       # kg (250 g sphere)
THETA_DEG = 45.0   # launch angle fixed in experiment


def ensure_dirs() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)


def load_corrected_ranges() -> pd.DataFrame:
    """
    Load average range per fin ratio and apply the 'outlier additional trials' correction
    for 0.75 and 1.00.
    """
    df_range = pd.read_csv(DATA / "horizontal_range_trials.csv")
    df_outliers = pd.read_csv(DATA / "outlier_additional_trials.csv")[["fin_ratio", "new_avg_range_m"]]

    df = df_range[["fin_ratio", "avg_range_m"]].merge(df_outliers, on="fin_ratio", how="left")
    df["avg_range_m_used"] = df["new_avg_range_m"].fillna(df["avg_range_m"])
    df = df.sort_values("fin_ratio").reset_index(drop=True)
    return df


def estimate_v0_from_control(df: pd.DataFrame) -> float:
    """
    Estimate v0 using the no-drag range formula at 45 degrees:
        R = v0^2 / g  (since sin(2*45°) = 1)

    Use the 0.00x (no-fins) average range as 'closest to ideal'.
    """
    control = df.loc[df["fin_ratio"] == 0.00, "avg_range_m_used"]
    if control.empty:
        raise ValueError("Could not find fin_ratio == 0.00 control row to estimate v0.")
    R0 = float(control.iloc[0])
    v0 = math.sqrt(R0 * G)
    return v0


def deriv(state: np.ndarray, k_eff: float) -> np.ndarray:
    """
    State = [x, y, vx, vy]
    Quadratic drag acceleration: a_drag = -(k_eff/m) * |v| * v_vec
    where k_eff has units kg/m (lumps 0.5*rho*Cd*A into one parameter).

    Returns d/dt [x, y, vx, vy]
    """
    x, y, vx, vy = state
    v = math.sqrt(vx * vx + vy * vy)
    ax = -(k_eff / MASS) * v * vx
    ay = -G - (k_eff / MASS) * v * vy
    return np.array([vx, vy, ax, ay], dtype=float)


def rk4_step(state: np.ndarray, dt: float, k_eff: float) -> np.ndarray:
    k1 = deriv(state, k_eff)
    k2 = deriv(state + 0.5 * dt * k1, k_eff)
    k3 = deriv(state + 0.5 * dt * k2, k_eff)
    k4 = deriv(state + dt * k3, k_eff)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_range(v0: float, theta_deg: float, k_eff: float, dt: float = 1e-3, t_max: float = 5.0) -> float:
    """
    Simulate until projectile hits y=0 again; return horizontal range.
    Uses RK4 integration.

    Note: starts at y=0, so we detect ground-crossing only after y becomes positive first.
    """
    theta = math.radians(theta_deg)
    vx0 = v0 * math.cos(theta)
    vy0 = v0 * math.sin(theta)

    state = np.array([0.0, 0.0, vx0, vy0], dtype=float)

    t = 0.0
    y_was_positive = False
    x_prev, y_prev = state[0], state[1]

    while t < t_max:
        state = rk4_step(state, dt, k_eff)
        t += dt
        x, y = float(state[0]), float(state[1])

        if y > 0:
            y_was_positive = True

        # detect crossing from above to below ground after going up
        if y_was_positive and (y_prev > 0.0) and (y <= 0.0):
            # linear interpolation for x at y=0
            frac = y_prev / (y_prev - y)  # between 0 and 1
            x_cross = x_prev + frac * (x - x_prev)
            return float(x_cross)

        x_prev, y_prev = x, y

    # If no crossing detected (shouldn't happen with these parameters), return last x
    return float(state[0])


def fit_k_for_target_range(v0: float, theta_deg: float, target_range: float) -> float:
    """
    Find k_eff such that simulated_range(k_eff) ~= target_range using monotonic bisection.
    """
    # Ideal (k=0) range is the maximum; if target is at/near ideal, return ~0.
    r0 = simulate_range(v0, theta_deg, k_eff=0.0)
    if target_range >= r0 * 0.999:  # allow tiny numerical tolerance
        return 0.0

    # Find an upper bound where range is <= target
    k_hi = 0.05
    r_hi = simulate_range(v0, theta_deg, k_eff=k_hi)
    while r_hi > target_range and k_hi < 100:
        k_hi *= 2.0
        r_hi = simulate_range(v0, theta_deg, k_eff=k_hi)

    k_lo = 0.0
    # Bisection
    for _ in range(60):
        k_mid = 0.5 * (k_lo + k_hi)
        r_mid = simulate_range(v0, theta_deg, k_eff=k_mid)
        if r_mid > target_range:
            k_lo = k_mid
        else:
            k_hi = k_mid

    return 0.5 * (k_lo + k_hi)


def main() -> None:
    ensure_dirs()
    df = load_corrected_ranges()
    v0 = estimate_v0_from_control(df)

    rows = []
    for _, row in df.iterrows():
        fin = float(row["fin_ratio"])
        R_obs = float(row["avg_range_m_used"])
        k_eff = fit_k_for_target_range(v0, THETA_DEG, R_obs)
        R_sim = simulate_range(v0, THETA_DEG, k_eff)
        rows.append({"fin_ratio": fin, "avg_range_obs_m": R_obs, "k_eff_kg_per_m": k_eff, "range_sim_m": R_sim})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DATA / "effective_drag_fit.csv", index=False)

    # Plot 1: k_eff vs fin ratio
    plt.figure()
    plt.plot(out["fin_ratio"], out["k_eff_kg_per_m"], marker="o")
    plt.xlabel("Fin length-to-diameter ratio (L/D)")
    plt.ylabel("Fitted effective drag parameter k_eff (kg/m)")
    plt.title("Effective quadratic-drag parameter vs fin ratio")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "effective_drag_k_vs_fin.png", dpi=200)
    plt.close()

    # Plot 2: observed vs simulated range
    plt.figure()
    plt.plot(out["fin_ratio"], out["avg_range_obs_m"], marker="o", label="Observed (EE)")
    plt.plot(out["fin_ratio"], out["range_sim_m"], marker="x", label="Drag-model fit")
    plt.xlabel("Fin length-to-diameter ratio (L/D)")
    plt.ylabel("Range (m)")
    plt.title("Range: observed vs drag-model fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG / "drag_model_range_fit.png", dpi=200)
    plt.close()

    print(f"Estimated v0 (from 0.00x control, no-drag): {v0:.3f} m/s")
    print("Saved:")
    print(" - outputs/figures/effective_drag_k_vs_fin.png")
    print(" - outputs/figures/drag_model_range_fit.png")
    print(" - data/processed/effective_drag_fit.csv")


if __name__ == "__main__":
    main()
