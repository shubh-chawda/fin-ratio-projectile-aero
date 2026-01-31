"""
Generate figures for the fin-length-to-diameter ratio EE dataset.

Run:
    python -m src.make_figures

Outputs:
    outputs/figures/*.png
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "figures"


def _ensure_outdir() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def _load_data() -> dict[str, pd.DataFrame]:
    df_range = pd.read_csv(DATA / "horizontal_range_trials.csv")
    df_range_std = pd.read_csv(DATA / "horizontal_range_stddev.csv")
    df_outliers = pd.read_csv(DATA / "outlier_additional_trials.csv")
    df_vel = pd.read_csv(DATA / "velocity_decay_trials.csv")
    df_vmax = pd.read_csv(DATA / "max_velocity_trials.csv")
    return {
        "range": df_range,
        "range_std": df_range_std,
        "outliers": df_outliers,
        "vel": df_vel,
        "vmax": df_vmax,
    }


def plot_range_vs_fin(d: dict[str, pd.DataFrame]) -> None:
    df = d["range"].merge(d["range_std"], on="fin_ratio", how="left")
    # Create a "corrected" average for 0.75 and 1.00 using the extra trials table
    corr = d["outliers"][["fin_ratio", "new_avg_range_m"]]
    df = df.merge(corr, on="fin_ratio", how="left")
    df["avg_range_corrected_m"] = df["new_avg_range_m"].fillna(df["avg_range_m"])

    x = df["fin_ratio"].to_numpy()
    y = df["avg_range_corrected_m"].to_numpy()
    yerr = df["stddev_m"].to_numpy()

    _ensure_outdir()
    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=3)
    # Best-fit line (simple linear regression)
    m, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    plt.plot(xx, m * xx + b)
    plt.xlabel("Fin length-to-diameter ratio (L/D)")
    plt.ylabel("Average horizontal range (m)")
    plt.title("Horizontal range vs fin ratio")
    plt.tight_layout()
    plt.savefig(OUT / "range_vs_fin.png", dpi=200)
    plt.close()


def plot_velocity_decay_vs_fin(d: dict[str, pd.DataFrame]) -> None:
    df = d["vel"].copy()
    x = df["fin_ratio"].to_numpy()
    y = df["avg_vel_decay_ms2"].to_numpy()
    yerr = df["uncertainty_ms2"].to_numpy()

    _ensure_outdir()
    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=3)
    m, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    plt.plot(xx, m * xx + b)
    plt.xlabel("Fin length-to-diameter ratio (L/D)")
    plt.ylabel("Average velocity decay (m/s²)")
    plt.title("Velocity decay vs fin ratio")
    plt.tight_layout()
    plt.savefig(OUT / "velocity_decay_vs_fin.png", dpi=200)
    plt.close()


def spearman_heatmap(d: dict[str, pd.DataFrame]) -> None:
    # Build a compact dataset and compute Spearman correlation
    df_range = d["range"].copy()
    corr = d["outliers"][["fin_ratio", "new_avg_range_m"]]
    df_range = df_range.merge(corr, on="fin_ratio", how="left")
    df_range["avg_range_m_used"] = df_range["new_avg_range_m"].fillna(df_range["avg_range_m"])

    df = (
        df_range[["fin_ratio", "avg_range_m_used"]]
        .merge(d["vel"][["fin_ratio", "avg_vel_decay_ms2"]], on="fin_ratio")
        .merge(d["vmax"][["fin_ratio", "avg_vmax_ms1"]], on="fin_ratio")
        .rename(columns={
            "avg_range_m_used": "avg_range_m",
            "avg_vel_decay_ms2": "avg_vel_decay_ms2",
            "avg_vmax_ms1": "avg_vmax_ms1",
        })
    )

    # If you want to exclude specific outliers, uncomment:
    # df = df[~df["fin_ratio"].isin([0.75, 1.00])].copy()

    corr = df.corr(method="spearman")

    _ensure_outdir()
    plt.figure()
    plt.imshow(corr.to_numpy(), aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=30, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar()
    plt.title("Spearman correlation heatmap")
    plt.tight_layout()
    plt.savefig(OUT / "spearman_heatmap.png", dpi=200)
    plt.close()


def main() -> None:
    d = _load_data()
    plot_range_vs_fin(d)
    plot_velocity_decay_vs_fin(d)
    spearman_heatmap(d)
    print("Done. Figures saved to outputs/figures/")


if __name__ == "__main__":
    main()
