"""
Bootstrap uncertainty for fitted quadratic-drag parameter k_eff vs fin ratio.

Run:
    python -m src.bootstrap_k_eff

Outputs:
    outputs/figures/k_eff_vs_fin_bootstrap_ci.png
    data/processed/bootstrap_k_eff_ci.csv
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


def _load_trials() -> dict[float, np.ndarray]:
    """
    Return dict fin_ratio -> array of range trials.
    Uses:
      - horizontal_range_trials.csv (trial1..trial5)
      - outlier_additional_trials.csv (trial6..trial10 for 0.75 and 1.00)
    """
    df = pd.read_csv(DATA / "horizontal_range_trials.csv")
    trials = {}
    for _, r in df.iterrows():
        fin = float(r["fin_ratio"])
        arr = np.array([r["trial1"], r["trial2"], r["trial3"], r["trial4"], r["trial5"]], dtype=float)
        trials[fin] = arr

    df_extra = pd.read_csv(DATA / "outlier_additional_trials.csv")
    for _, r in df_extra.iterrows():
        fin = float(r["fin_ratio"])
        extra = np.array([r["trial6"], r["trial7"], r["trial8"], r["trial9"], r["trial10"]], dtype=float)
        if fin in trials:
            trials[fin] = np.concatenate([trials[fin], extra])
        else:
            trials[fin] = extra

    return trials


def deriv_quad(state: np.ndarray, k_eff: float) -> np.ndarray:
    x, y, vx, vy = state
    v = math.sqrt(vx * vx + vy * vy)
    ax = -(k_eff / MASS) * v * vx
    ay = -G - (k_eff / MASS) * v * vy
    return np.array([vx, vy, ax, ay], dtype=float)


def rk4_step(state: np.ndarray, dt: float, k_eff: float) -> np.ndarray:
    k1 = deriv_quad(state, k_eff)
    k2 = deriv_quad(state + 0.5 * dt * k1, k_eff)
    k3 = deriv_quad(state + 0.5 * dt * k2, k_eff)
    k4 = deriv_quad(state + dt * k3, k_eff)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_range_quad(v0: float, theta_deg: float, k_eff: float, dt: float = 1e-3, t_max: float = 5.0) -> float:
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

        if y_was_positive and (y_prev > 0.0) and (y <= 0.0):
            frac = y_prev / (y_prev - y)
            x_cross = x_prev + frac * (x - x_prev)
            return float(x_cross)

        x_prev, y_prev = x, y

    return float(state[0])


def fit_k_for_target_range(v0: float, target_range: float) -> float:
    r0 = simulate_range_quad(v0, THETA_DEG, 0.0)
    if target_range >= r0 * 0.999:
        return 0.0

    k_hi = 0.05
    r_hi = simulate_range_quad(v0, THETA_DEG, k_hi)
    while r_hi > target_range and k_hi < 200:
        k_hi *= 2.0
        r_hi = simulate_range_quad(v0, THETA_DEG, k_hi)

    k_lo = 0.0
    for _ in range(60):
        k_mid = 0.5 * (k_lo + k_hi)
        r_mid = simulate_range_quad(v0, THETA_DEG, k_mid)
        if r_mid > target_range:
            k_lo = k_mid
        else:
            k_hi = k_mid
    return 0.5 * (k_lo + k_hi)


def main() -> None:
    ensure_dirs()
    trials = _load_trials()
    fins = sorted(trials.keys())

    rng = np.random.default_rng(42)
    n_boot = 1200

    # Bootstrap: resample control trials to get v0 each time (propagates v0 uncertainty)
    control_trials = trials[0.0]

    boot_k = {fin: [] for fin in fins}

    for _ in range(n_boot):
        R0 = float(rng.choice(control_trials, size=len(control_trials), replace=True).mean())
        v0 = math.sqrt(R0 * G)

        for fin in fins:
            arr = trials[fin]
            Rb = float(rng.choice(arr, size=len(arr), replace=True).mean())
            k = fit_k_for_target_range(v0, Rb)
            boot_k[fin].append(k)

    rows = []
    for fin in fins:
        ks = np.array(boot_k[fin], dtype=float)
        rows.append({
            "fin_ratio": fin,
            "k_median": float(np.median(ks)),
            "k_mean": float(np.mean(ks)),
            "k_p2_5": float(np.quantile(ks, 0.025)),
            "k_p97_5": float(np.quantile(ks, 0.975)),
        })

    out = pd.DataFrame(rows).sort_values("fin_ratio")
    out.to_csv(OUT_DATA / "bootstrap_k_eff_ci.csv", index=False)

    # Plot with CI error bars
    x = out["fin_ratio"].to_numpy()
    y = out["k_median"].to_numpy()
    yerr_low = y - out["k_p2_5"].to_numpy()
    yerr_high = out["k_p97_5"].to_numpy() - y

    plt.figure(figsize=(10, 6))
    plt.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="o", capsize=4)
    plt.xlabel("Fin length-to-diameter ratio (L/D)")
    plt.ylabel("Fitted k_eff (kg/m), median ± 95% CI")
    plt.title("Bootstrap uncertainty: effective quadratic-drag parameter vs fin ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "k_eff_vs_fin_bootstrap_ci.png", dpi=200)
    plt.close()

    print("Saved:")
    print(" - outputs/figures/k_eff_vs_fin_bootstrap_ci.png")
    print(" - data/processed/bootstrap_k_eff_ci.csv")


if __name__ == "__main__":
    main()
