"""
Compare linear-drag vs quadratic-drag models by:
1) fitting drag parameter per fin ratio to match observed range
2) predicting average velocity-decay slope from simulated speed vs time
3) comparing to observed velocity decay from the EE table

Run:
    python -m src.model_compare

Outputs:
    outputs/figures/velocity_decay_model_compare.png
    outputs/figures/model_rmse_velocity_decay.png
    data/processed/model_comparison_velocity_decay.csv
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

G = 9.81
MASS = 0.250
THETA_DEG = 45.0


def ensure_dirs() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)


def load_corrected_ranges() -> pd.DataFrame:
    df_range = pd.read_csv(DATA / "horizontal_range_trials.csv")
    df_out = pd.read_csv(DATA / "outlier_additional_trials.csv")[["fin_ratio", "new_avg_range_m"]]
    df = df_range[["fin_ratio", "avg_range_m"]].merge(df_out, on="fin_ratio", how="left")
    df["range_obs_m"] = df["new_avg_range_m"].fillna(df["avg_range_m"])
    return df[["fin_ratio", "range_obs_m"]].sort_values("fin_ratio").reset_index(drop=True)


def load_velocity_decay() -> pd.DataFrame:
    df = pd.read_csv(DATA / "velocity_decay_trials.csv")
    return df[["fin_ratio", "avg_vel_decay_ms2"]].rename(columns={"avg_vel_decay_ms2": "decay_obs_ms2"})


def estimate_v0_from_control(range_df: pd.DataFrame) -> float:
    R0 = float(range_df.loc[range_df["fin_ratio"] == 0.0, "range_obs_m"].iloc[0])
    return math.sqrt(R0 * G)


# ---- Quadratic drag ----
def deriv_quad(state: np.ndarray, k_eff: float) -> np.ndarray:
    x, y, vx, vy = state
    v = math.sqrt(vx * vx + vy * vy)
    ax = -(k_eff / MASS) * v * vx
    ay = -G - (k_eff / MASS) * v * vy
    return np.array([vx, vy, ax, ay], dtype=float)


# ---- Linear drag ----
def deriv_lin(state: np.ndarray, b_eff: float) -> np.ndarray:
    x, y, vx, vy = state
    ax = -(b_eff / MASS) * vx
    ay = -G - (b_eff / MASS) * vy
    return np.array([vx, vy, ax, ay], dtype=float)


def rk4_step(state: np.ndarray, dt: float, param: float, mode: str) -> np.ndarray:
    if mode == "quad":
        f = lambda s: deriv_quad(s, param)
    elif mode == "lin":
        f = lambda s: deriv_lin(s, param)
    else:
        raise ValueError("mode must be 'quad' or 'lin'")

    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_full(v0: float, param: float, mode: str, dt: float = 1e-3, t_max: float = 5.0):
    theta = math.radians(THETA_DEG)
    vx0 = v0 * math.cos(theta)
    vy0 = v0 * math.sin(theta)
    state = np.array([0.0, 0.0, vx0, vy0], dtype=float)

    t = 0.0
    y_was_positive = False

    ts, speeds = [], []
    x_prev, y_prev = state[0], state[1]

    while t < t_max:
        # record
        vx, vy = float(state[2]), float(state[3])
        speed = math.sqrt(vx*vx + vy*vy)
        ts.append(t)
        speeds.append(speed)

        # step
        state = rk4_step(state, dt, param, mode)
        t += dt
        x, y = float(state[0]), float(state[1])

        if y > 0:
            y_was_positive = True

        # hit ground
        if y_was_positive and (y_prev > 0.0) and (y <= 0.0):
            frac = y_prev / (y_prev - y)
            x_cross = x_prev + frac * (x - x_prev)
            return float(x_cross), np.array(ts), np.array(speeds)

        x_prev, y_prev = x, y

    return float(state[0]), np.array(ts), np.array(speeds)


def fit_param_for_range(v0: float, target_range: float, mode: str) -> float:
    # Ideal range for param=0
    r0, _, _ = simulate_full(v0, 0.0, mode)
    if target_range >= r0 * 0.999:
        return 0.0

    hi = 0.05 if mode == "quad" else 0.5
    r_hi, _, _ = simulate_full(v0, hi, mode)
    while r_hi > target_range and hi < 500:
        hi *= 2.0
        r_hi, _, _ = simulate_full(v0, hi, mode)

    lo = 0.0
    for _ in range(55):
        mid = 0.5 * (lo + hi)
        r_mid, _, _ = simulate_full(v0, mid, mode)
        if r_mid > target_range:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def slope_speed_vs_time(ts: np.ndarray, speeds: np.ndarray) -> float:
    # Linear regression slope (m/s per s = m/s^2)
    # We keep it comparable to velocity decay sign (negative slope).
    if len(ts) < 10:
        return float("nan")
    m, b = np.polyfit(ts, speeds, 1)
    return float(m)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    ensure_dirs()

    rng_df = load_corrected_ranges()
    dec_df = load_velocity_decay()
    df = rng_df.merge(dec_df, on="fin_ratio", how="inner").sort_values("fin_ratio").reset_index(drop=True)

    v0 = estimate_v0_from_control(rng_df)

    rows = []
    for _, r in df.iterrows():
        fin = float(r["fin_ratio"])
        R_obs = float(r["range_obs_m"])
        decay_obs = float(r["decay_obs_ms2"])

        kq = fit_param_for_range(v0, R_obs, "quad")
        Rq, tq, sq = simulate_full(v0, kq, "quad")
        decay_q = slope_speed_vs_time(tq, sq)

        bl = fit_param_for_range(v0, R_obs, "lin")
        Rl, tl, sl = simulate_full(v0, bl, "lin")
        decay_l = slope_speed_vs_time(tl, sl)

        rows.append({
            "fin_ratio": fin,
            "range_obs_m": R_obs,
            "decay_obs_ms2": decay_obs,
            "k_eff_quad_kg_per_m": kq,
            "b_eff_lin_kg_per_s": bl,
            "decay_pred_quad_ms2": decay_q,
            "decay_pred_lin_ms2": decay_l,
        })

    out = pd.DataFrame(rows).sort_values("fin_ratio")
    out.to_csv(OUT_DATA / "model_comparison_velocity_decay.csv", index=False)

    # Plot: observed vs predicted velocity decay
    plt.figure(figsize=(10, 6))
    plt.plot(out["fin_ratio"], out["decay_obs_ms2"], marker="o", label="Observed (EE)")
    plt.plot(out["fin_ratio"], out["decay_pred_quad_ms2"], marker="x", label="Predicted (quadratic drag)")
    plt.plot(out["fin_ratio"], out["decay_pred_lin_ms2"], marker="^", label="Predicted (linear drag)")
    plt.xlabel("Fin length-to-diameter ratio (L/D)")
    plt.ylabel("Velocity decay (m/s²)  (slope of speed vs time)")
    plt.title("Model validation: velocity decay (observed vs predicted)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG / "velocity_decay_model_compare.png", dpi=200)
    plt.close()

    # RMSE bar plot
    obs = out["decay_obs_ms2"].to_numpy()
    pred_q = out["decay_pred_quad_ms2"].to_numpy()
    pred_l = out["decay_pred_lin_ms2"].to_numpy()
    rmse_q = rmse(obs, pred_q)
    rmse_l = rmse(obs, pred_l)

    plt.figure(figsize=(7, 5))
    plt.bar(["Quadratic drag", "Linear drag"], [rmse_q, rmse_l])
    plt.ylabel("RMSE vs observed velocity decay (m/s²)")
    plt.title("Which drag model better matches velocity decay?")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "model_rmse_velocity_decay.png", dpi=200)
    plt.close()

    print(f"Estimated v0 from control range: {v0:.3f} m/s")
    print("Saved:")
    print(" - outputs/figures/velocity_decay_model_compare.png")
    print(" - outputs/figures/model_rmse_velocity_decay.png")
    print(" - data/processed/model_comparison_velocity_decay.csv")


if __name__ == "__main__":
    main()
