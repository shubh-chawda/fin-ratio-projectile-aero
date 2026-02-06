from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT_FIG = ROOT / "outputs" / "figures"
OUT_DATA = ROOT / "data" / "processed"

TARGET_RATIO = 0.75  # the “dip/anomaly” ratio we want to test


def ensure_dirs() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)


def _drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """If a CSV produces duplicate column names (e.g., fin_ratio twice), keep first occurrence."""
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def _to_long_trials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a table of trials into long format with columns:
      fin_ratio, range_m

    Supports:
      - already-long format (has a 'range_m' column)
      - wide tables (multiple trial columns)

    Guards against:
      - duplicate columns (e.g., two 'fin_ratio' columns)
      - summary-only files (avg/new_avg columns) being mis-treated as trials
    """
    df = _drop_duplicate_columns(df)

    if "fin_ratio" not in df.columns:
        raise ValueError("Expected a 'fin_ratio' column in trial data CSV.")

    # Already-long format
    if "range_m" in df.columns:
        out = df[["fin_ratio", "range_m"]].copy()
        out["fin_ratio"] = pd.to_numeric(out["fin_ratio"], errors="coerce")
        out["range_m"] = pd.to_numeric(out["range_m"], errors="coerce")
        return out.dropna(subset=["fin_ratio", "range_m"]).reset_index(drop=True)

    # Wide format: melt everything except fin_ratio, but EXCLUDE summary columns
    summary_cols = {"avg_range_m", "avg_range_m_used", "new_avg_range_m"}
    value_cols = [c for c in df.columns if c != "fin_ratio" and c.lower() not in summary_cols]

    if not value_cols:
        raise ValueError(
            "No trial columns found to melt. Your CSV appears to contain only summary columns "
            "(e.g., avg_range_m / new_avg_range_m). This significance test needs raw trial columns."
        )

    melted = df.melt(id_vars=["fin_ratio"], value_vars=value_cols, value_name="range_m")
    melted["fin_ratio"] = pd.to_numeric(melted["fin_ratio"], errors="coerce")
    melted["range_m"] = pd.to_numeric(melted["range_m"], errors="coerce")

    out = (
        melted[["fin_ratio", "range_m"]]
        .dropna(subset=["fin_ratio", "range_m"])
        .reset_index(drop=True)
    )
    return out


def load_all_range_trials() -> pd.DataFrame:
    """
    Load all horizontal range trials.

    We use:
      - horizontal_range_trials.csv   (main trials)
      - outlier_additional_trials.csv (extra trials, if present as trial columns)

    If outlier_additional_trials.csv contains ONLY summary columns (fin_ratio + new_avg_range_m),
    we skip it because it does not add raw trial data.
    """
    base = pd.read_csv(DATA / "horizontal_range_trials.csv")
    base = _drop_duplicate_columns(base)
    base_long = _to_long_trials(base)

    extra_path = DATA / "outlier_additional_trials.csv"
    if extra_path.exists():
        extra = pd.read_csv(extra_path)
        extra = _drop_duplicate_columns(extra)

        cols_lower = [c.lower() for c in extra.columns]
        has_new_avg = "new_avg_range_m" in cols_lower

        # Trial columns are everything except fin_ratio and new_avg_range_m
        trial_cols = [c for c in extra.columns if c.lower() not in {"fin_ratio", "new_avg_range_m"}]

        # If it's ONLY a summary file (fin_ratio + new_avg_range_m), skip it
        if has_new_avg and len(trial_cols) == 0:
            extra_long = pd.DataFrame(columns=["fin_ratio", "range_m"])
        else:
            keep_cols = ["fin_ratio"] + trial_cols  # fin_ratio included exactly once
            extra_long = _to_long_trials(extra[keep_cols])

        all_trials = pd.concat([base_long, extra_long], ignore_index=True)
    else:
        all_trials = base_long

    all_trials["fin_ratio"] = pd.to_numeric(all_trials["fin_ratio"], errors="coerce")
    all_trials["range_m"] = pd.to_numeric(all_trials["range_m"], errors="coerce")
    all_trials = all_trials.dropna(subset=["fin_ratio", "range_m"]).reset_index(drop=True)

    return all_trials


def isotonic_decreasing_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Isotonic regression enforcing y to be monotonically DECREASING in x.
    Uses PAV algorithm on -y as an increasing fit.
    Returns fitted y_hat aligned to sorted(x).
    """
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    z = -y_sorted  # convert to increasing
    blocks: List[Tuple[float, int]] = []
    for val in z:
        blocks.append((float(val), 1))
        while len(blocks) >= 2:
            s1, n1 = blocks[-2]
            s2, n2 = blocks[-1]
            m1 = s1 / n1
            m2 = s2 / n2
            if m1 <= m2:
                break
            blocks = blocks[:-2] + [(s1 + s2, n1 + n2)]

    z_hat: List[float] = []
    for s, n in blocks:
        z_hat.extend([s / n] * n)

    y_hat_sorted = -np.array(z_hat, dtype=float)
    return y_hat_sorted


def predict_linear(x_sorted: np.ndarray, y_sorted: np.ndarray, x0: float) -> float:
    """Linear interpolation of y at x0 on sorted (x_sorted, y_sorted)."""
    if x0 <= x_sorted[0]:
        return float(y_sorted[0])
    if x0 >= x_sorted[-1]:
        return float(y_sorted[-1])

    i = int(np.searchsorted(x_sorted, x0) - 1)
    x1, x2 = float(x_sorted[i]), float(x_sorted[i + 1])
    y1, y2 = float(y_sorted[i]), float(y_sorted[i + 1])

    if x2 == x1:
        return float(y1)

    t = (x0 - x1) / (x2 - x1)
    return float(y1 + t * (y2 - y1))


def main() -> None:
    ensure_dirs()

    N_BOOT = int(os.environ.get("N_BOOT", "5000"))
    SEED = int(os.environ.get("SEED", "123"))

    trials = load_all_range_trials()

    ratios = np.array(sorted(trials["fin_ratio"].unique()), dtype=float)
    if TARGET_RATIO not in ratios:
        raise ValueError(
            f"TARGET_RATIO={TARGET_RATIO} not found in fin ratios in data. "
            f"Available: {ratios.tolist()}"
        )

    idx0 = int(np.where(ratios == TARGET_RATIO)[0][0])

    # Collect arrays per ratio
    data: Dict[float, np.ndarray] = {}
    for r in ratios:
        arr = trials.loc[trials["fin_ratio"] == r, "range_m"].to_numpy(dtype=float)
        if arr.size < 2:
            raise ValueError(f"Not enough trials for fin_ratio={r}. Need >=2, got {arr.size}.")
        data[float(r)] = arr

    # Observed means (from data)
    mean_obs = np.array([data[float(r)].mean() for r in ratios], dtype=float)

    # Null model: monotonic DECREASING fit built from all ratios EXCEPT TARGET_RATIO
    mask = np.ones_like(ratios, dtype=bool)
    mask[idx0] = False
    x_other = ratios[mask]
    y_other = mean_obs[mask]

    yhat_other_sorted = isotonic_decreasing_fit(x_other, y_other)
    x_other_sorted = np.sort(x_other)

    pred_null_at_target = predict_linear(x_other_sorted, yhat_other_sorted, TARGET_RATIO)
    resid_obs = float(mean_obs[idx0] - pred_null_at_target)

    # Bootstrap: resample within each ratio, refit null excluding target, compute residual at target
    rng = np.random.default_rng(SEED)
    means_boot = np.empty((N_BOOT, ratios.size), dtype=float)
    resid_boot = np.empty(N_BOOT, dtype=float)

    for b in range(N_BOOT):
        for j, r in enumerate(ratios):
            arr = data[float(r)]
            sample = rng.choice(arr, size=arr.size, replace=True)
            means_boot[b, j] = float(sample.mean())

        y_other_b = means_boot[b, mask]
        yhat_b = isotonic_decreasing_fit(x_other, y_other_b)

        pred_b = predict_linear(x_other_sorted, yhat_b, TARGET_RATIO)
        resid_boot[b] = float(means_boot[b, idx0] - pred_b)

    # 95% bootstrap CI for residual
    ci_lo, ci_hi = np.quantile(resid_boot, [0.025, 0.975])

    # “p-like” sign probability (two-sided)
    p_sign = 2.0 * min(np.mean(resid_boot <= 0.0), np.mean(resid_boot >= 0.0))
    p_sign = float(min(1.0, p_sign))

    # CI for mean at each ratio for error bars
    ci_means_lo = np.quantile(means_boot, 0.025, axis=0)
    ci_means_hi = np.quantile(means_boot, 0.975, axis=0)

    # Null curve over all ratios for plot
    null_curve = np.array([predict_linear(x_other_sorted, yhat_other_sorted, float(x)) for x in ratios], dtype=float)

    # --- FIGURE 1: Data with 95% CI + null monotonic trend (excluding target) ---
    fig1_path = OUT_FIG / "range_075_null_fit.png"
    plt.figure()
    yerr = np.vstack([mean_obs - ci_means_lo, ci_means_hi - mean_obs])
    plt.errorbar(ratios, mean_obs, yerr=yerr, fmt="o", capsize=3, label="Observed mean ± 95% bootstrap CI")
    plt.plot(ratios, null_curve, linestyle="--", label="Monotonic null fit (excl. 0.75)")
    plt.scatter([TARGET_RATIO], [mean_obs[idx0]], marker="*", s=180, label="0.75× point")
    plt.xlabel("Fin length-to-diameter ratio (L/D)")
    plt.ylabel("Range (m)")
    plt.title("0.75× anomaly test: observed range vs monotonic null")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=200)
    plt.close()

    # --- FIGURE 2: Bootstrap residual distribution ---
    fig2_path = OUT_FIG / "range_075_residual_bootstrap.png"
    plt.figure()
    plt.hist(resid_boot, bins=50)
    plt.axvline(0.0, linestyle="--")
    plt.axvline(resid_obs)
    plt.xlabel("Residual at 0.75× (observed mean − null-predicted mean) [m]")
    plt.ylabel("Bootstrap count")
    plt.title("Bootstrap residual distribution for 0.75× anomaly")
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=200)
    plt.close()

    # Save JSON summary
    out_json = {
        "target_ratio": TARGET_RATIO,
        "n_boot": N_BOOT,
        "seed": SEED,
        "observed_mean_at_target_m": float(mean_obs[idx0]),
        "null_pred_at_target_m": float(pred_null_at_target),
        "observed_residual_m": float(resid_obs),
        "bootstrap_ci_95_residual_m": [float(ci_lo), float(ci_hi)],
        "bootstrap_two_sided_sign_p": float(p_sign),
        "criterion": "Significant if 95% bootstrap CI for residual excludes 0.",
        "figures": {
            "null_fit_plot": str(fig1_path.relative_to(ROOT)),
            "residual_hist": str(fig2_path.relative_to(ROOT)),
        },
    }
    out_path = OUT_DATA / "range_075_significance.json"
    out_path.write_text(json.dumps(out_json, indent=2))

    # Print a clean summary line
    print("\n=== 0.75× anomaly significance test ===")
    print(f"N_BOOT = {N_BOOT}")
    print(f"Observed mean at 0.75× = {mean_obs[idx0]:.6f} m")
    print(f"Null-predicted mean at 0.75× = {pred_null_at_target:.6f} m")
    print(f"Residual (obs - null) = {resid_obs:.6f} m")
    print(f"95% bootstrap CI for residual = [{ci_lo:.6f}, {ci_hi:.6f}] m")
    print(f"Bootstrap two-sided sign p ≈ {p_sign:.6f}")
    print(f"Saved: {fig1_path}")
    print(f"Saved: {fig2_path}")
    print(f"Saved: {out_path}")
    print("Decision rule: significant if CI excludes 0.\n")


if __name__ == "__main__":
    main()
