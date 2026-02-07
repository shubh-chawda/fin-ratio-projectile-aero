"""
Residual diagnostics: where does the range-vs-fin trend "fail"?

We build a monotonic DECREASING null trend (isotonic regression) and do
leave-one-out (LOO) predictions at each fin_ratio. Then we compute residuals:

    residual = observed_mean(fin_ratio) - null_predicted_mean(fin_ratio)

We bootstrap within each fin_ratio to get 95% CIs for:
- observed mean range
- residuals (LOO)

Run:
    N_BOOT=2000 .venv/bin/python -m src.residual_diagnostics
(or N_BOOT=5000 for a stronger result)

Outputs:
    outputs/figures/loo_null_fit_means.png
    outputs/figures/loo_null_fit_residuals.png
    data/processed/loo_null_residuals.csv
    data/processed/loo_null_residuals_summary.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT_FIG = ROOT / "outputs" / "figures"
OUT_DATA = ROOT / "data" / "processed"


# columns we should NOT treat as trial values if present in CSVs
IGNORE_VALUE_COLS = {
    "avg_range_m",
    "avg_range",
    "mean_range_m",
    "std_range_m",
    "new_avg_range_m",
    "avg_range_m_used",
    "notes",
    "comment",
    "remarks",
}


def ensure_dirs() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)


def _read_csv_unique_cols(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # guard against duplicated columns (causes melt() errors)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def _to_long_trials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert trials table into long format with columns: fin_ratio, range_m
    Supports:
      - already-long format (has 'range_m')
      - wide format (trial columns)
    """
    if "fin_ratio" not in df.columns:
        raise ValueError("Expected a 'fin_ratio' column in trial data CSV.")

    
    if "range_m" in df.columns:
        out = df[["fin_ratio", "range_m"]].copy()
        out["fin_ratio"] = pd.to_numeric(out["fin_ratio"], errors="coerce")
        out["range_m"] = pd.to_numeric(out["range_m"], errors="coerce")
        return out.dropna(subset=["fin_ratio", "range_m"]).reset_index(drop=True)

    # wide format: melt likely trial columns
    value_cols = [
        c
        for c in df.columns
        if c != "fin_ratio" and (c.lower() not in IGNORE_VALUE_COLS)
    ]

    if not value_cols:
        raise ValueError(
            "Could not find trial columns to melt. "
            f"Columns seen: {list(df.columns)}"
        )

    melted = df.melt(
        id_vars=["fin_ratio"],
        value_vars=value_cols,
        value_name="range_m",
        var_name="trial_col",
    )

    melted["fin_ratio"] = pd.to_numeric(melted["fin_ratio"], errors="coerce")
    melted["range_m"] = pd.to_numeric(melted["range_m"], errors="coerce")

    out = melted[["fin_ratio", "range_m"]].dropna(subset=["fin_ratio", "range_m"]).reset_index(drop=True)
    return out


def load_all_range_trials() -> pd.DataFrame:
    """
    Load all horizontal range trial values into long format.
    Uses:
      - data/raw/horizontal_range_trials.csv
      - data/raw/outlier_additional_trials.csv (if it actually contains trial columns)
    """
    base_path = DATA / "horizontal_range_trials.csv"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing: {base_path}")

    base = _read_csv_unique_cols(base_path)
    base_long = _to_long_trials(base)

    extra_path = DATA / "outlier_additional_trials.csv"
    if extra_path.exists():
        extra = _read_csv_unique_cols(extra_path)

        # If it's only a summary file (e.g., fin_ratio + new_avg_range_m), skip it
        lower_cols = {c.lower() for c in extra.columns}
        trial_like_cols = [
            c for c in extra.columns
            if c.lower() not in {"fin_ratio", "new_avg_range_m", *IGNORE_VALUE_COLS}
        ]

        if ("new_avg_range_m" in lower_cols) and (len(trial_like_cols) == 0):
            extra_long = pd.DataFrame(columns=["fin_ratio", "range_m"])
        else:
            # drop summary column if present; keep fin_ratio + trial-like cols
            keep = ["fin_ratio"] + trial_like_cols
            extra_long = _to_long_trials(extra[keep])

        trials = pd.concat([base_long, extra_long], ignore_index=True)
    else:
        trials = base_long

    trials["fin_ratio"] = pd.to_numeric(trials["fin_ratio"], errors="coerce")
    trials["range_m"] = pd.to_numeric(trials["range_m"], errors="coerce")
    trials = trials.dropna(subset=["fin_ratio", "range_m"]).reset_index(drop=True)

    return trials


def isotonic_decreasing_fit(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isotonic regression enforcing y to be monotonically DECREASING in x.
    PAV on -y (increasing). Returns (x_sorted, yhat_sorted).
    """
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    z = -ys  # increasing target

    blocks: List[Tuple[float, int]] = []
    for val in z:
        blocks.append((float(val), 1))
        while len(blocks) >= 2:
            s1, n1 = blocks[-2]
            s2, n2 = blocks[-1]
            m1 = s1 / n1
            m2 = s2 / n2
            # increasing constraint: m1 <= m2
            if m1 <= m2:
                break
            blocks = blocks[:-2] + [(s1 + s2, n1 + n2)]

    zhat: List[float] = []
    for s, n in blocks:
        zhat.extend([s / n] * n)

    yhat = -np.array(zhat, dtype=float)
    return xs, yhat


def predict_linear(x_sorted: np.ndarray, y_sorted: np.ndarray, x0: float) -> float:
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


def loo_null_predictions(ratios: np.ndarray, means: np.ndarray) -> np.ndarray:
    """
    For each ratio i: fit isotonic decreasing on all j != i, predict at i.
    Returns pred array same length as ratios.
    """
    preds = np.empty_like(means, dtype=float)
    n = ratios.size
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        x_other = ratios[mask]
        y_other = means[mask]
        xs, yhat = isotonic_decreasing_fit(x_other, y_other)
        preds[i] = predict_linear(xs, yhat, float(ratios[i]))
    return preds


def main() -> None:
    ensure_dirs()

    N_BOOT = int(os.environ.get("N_BOOT", "2000"))
    SEED = int(os.environ.get("SEED", "123"))

    trials = load_all_range_trials()

    ratios = np.array(sorted(trials["fin_ratio"].unique()), dtype=float)

    # collect raw arrays per ratio
    data: Dict[float, np.ndarray] = {}
    for r in ratios:
        arr = trials.loc[trials["fin_ratio"] == r, "range_m"].to_numpy(dtype=float)
        if arr.size < 2:
            raise ValueError(f"Not enough trial values for fin_ratio={r}. Need >=2, got {arr.size}.")
        data[float(r)] = arr

    # observed mean
    mean_obs = np.array([data[float(r)].mean() for r in ratios], dtype=float)

    # bootstrap mean per ratio
    rng = np.random.default_rng(SEED)
    means_boot = np.empty((N_BOOT, ratios.size), dtype=float)

    for b in range(N_BOOT):
        for j, r in enumerate(ratios):
            arr = data[float(r)]
            sample = rng.choice(arr, size=arr.size, replace=True)
            means_boot[b, j] = float(sample.mean())

    # 95% CI for observed mean at each ratio
    ci_mean_lo = np.quantile(means_boot, 0.025, axis=0)
    ci_mean_hi = np.quantile(means_boot, 0.975, axis=0)

    # LOO null predictions on observed means
    pred_null = loo_null_predictions(ratios, mean_obs)
    resid_obs = mean_obs - pred_null

    # bootstrap CIs for residuals (LOO at each bootstrap)
    resid_boot = np.empty((N_BOOT, ratios.size), dtype=float)
    for b in range(N_BOOT):
        pred_b = loo_null_predictions(ratios, means_boot[b, :])
        resid_boot[b, :] = means_boot[b, :] - pred_b

    ci_resid_lo = np.quantile(resid_boot, 0.025, axis=0)
    ci_resid_hi = np.quantile(resid_boot, 0.975, axis=0)

    # significance flag: residual CI excludes 0
    sig = (ci_resid_lo > 0.0) | (ci_resid_hi < 0.0)

    # save a tidy table
    out_df = pd.DataFrame({
        "fin_ratio": ratios,
        "mean_obs_m": mean_obs,
        "mean_ci_lo_m": ci_mean_lo,
        "mean_ci_hi_m": ci_mean_hi,
        "null_pred_loo_m": pred_null,
        "residual_loo_m": resid_obs,
        "resid_ci_lo_m": ci_resid_lo,
        "resid_ci_hi_m": ci_resid_hi,
        "resid_ci_excludes_0": sig.astype(bool),
    })

    csv_path = OUT_DATA / "loo_null_residuals.csv"
    out_df.to_csv(csv_path, index=False)

    # summary json
    biggest_idx = int(np.argmax(np.abs(resid_obs)))
    summary = {
        "n_boot": N_BOOT,
        "seed": SEED,
        "ratios": ratios.tolist(),
        "most_anomalous_ratio": float(ratios[biggest_idx]),
        "most_anomalous_residual_m": float(resid_obs[biggest_idx]),
        "most_anomalous_residual_ci_95_m": [float(ci_resid_lo[biggest_idx]), float(ci_resid_hi[biggest_idx])],
        "count_ratios_with_ci_excluding_0": int(sig.sum()),
        "outputs": {
            "table_csv": str(csv_path.relative_to(ROOT)),
            "fig_means": "outputs/figures/loo_null_fit_means.png",
            "fig_residuals": "outputs/figures/loo_null_fit_residuals.png",
        },
        "decision_rule": "A ratio is 'anomalous' if the 95% bootstrap CI of its LOO residual excludes 0.",
    }

    json_path = OUT_DATA / "loo_null_residuals_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # ---- FIG 1: observed means + CI + LOO null predictions (dashed) ----
    fig1 = OUT_FIG / "loo_null_fit_means.png"
    plt.figure()
    yerr = np.vstack([mean_obs - ci_mean_lo, ci_mean_hi - mean_obs])
    plt.errorbar(ratios, mean_obs, yerr=yerr, fmt="o", capsize=3, label="Observed mean ± 95% bootstrap CI")
    plt.plot(ratios, pred_null, linestyle="--", label="LOO monotonic null prediction")
    plt.xlabel("Fin length-to-diameter ratio (L/D)")
    plt.ylabel("Range (m)")
    plt.title("Observed range vs leave-one-out monotonic null")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1, dpi=200)
    plt.close()

    # ---- FIG 2: residuals + CI, highlight anomalies ----
    fig2 = OUT_FIG / "loo_null_fit_residuals.png"
    plt.figure()
    resid_yerr = np.vstack([resid_obs - ci_resid_lo, ci_resid_hi - resid_obs])
    plt.errorbar(ratios, resid_obs, yerr=resid_yerr, fmt="o", capsize=3, label="Residual (obs − null) ± 95% CI")
    plt.axhline(0.0, linestyle="--")
    # mark points whose CI excludes 0
    if sig.any():
        plt.scatter(ratios[sig], resid_obs[sig], marker="*", s=160, label="CI excludes 0")
    plt.xlabel("Fin length-to-diameter ratio (L/D)")
    plt.ylabel("Residual (m)")
    plt.title("Residual diagnostics: where monotonic trend fails")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2, dpi=200)
    plt.close()

    print("\n=== Residual diagnostics (LOO monotonic null) ===")
    print(f"N_BOOT = {N_BOOT}")
    print(f"Saved: {fig1}")
    print(f"Saved: {fig2}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Most anomalous ratio = {summary['most_anomalous_ratio']}, residual = {summary['most_anomalous_residual_m']:.6f} m")
    print("Decision rule: anomalous if 95% CI excludes 0.\n")


if __name__ == "__main__":
    main()
