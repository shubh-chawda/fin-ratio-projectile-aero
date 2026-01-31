"""
Generate EE-style figures for the fin-length-to-diameter ratio dataset.

Run:
    python3 -m src.make_figures

Outputs:
    outputs/figures/*.png
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
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
    """
    EE Graph 9.1 style:
    Best-Fit:      R = -0.256 L + 2.00
    Max slope:     R = -0.235 L + 1.98
    Min slope:     R = -0.325 L + 2.06
    """
    df = d["range"].merge(d["range_std"], on="fin_ratio", how="left")

    # Use corrected averages for the two outlier ratios (0.75 and 1.00)
    corr = d["outliers"][["fin_ratio", "new_avg_range_m"]]
    df = df.merge(corr, on="fin_ratio", how="left")
    df["avg_range_used_m"] = df["new_avg_range_m"].fillna(df["avg_range_m"])

    x = df["fin_ratio"].to_numpy()
    y = df["avg_range_used_m"].to_numpy()
    yerr = df["stddev_m"].to_numpy()

    _ensure_outdir()
    plt.figure(figsize=(10, 6))

    # Lines (use EE equations exactly)
    xx = np.linspace(0.0, 2.0, 200)
    y_best = -0.256 * xx + 2.00
    y_max  = -0.235 * xx + 1.98
    y_min  = -0.325 * xx + 2.06

    plt.plot(xx, y_best, label="Best-fit line")
    plt.plot(xx, y_max,  linestyle="--", label="Max slope line")
    plt.plot(xx, y_min,  linestyle="--", label="Min slope line")

    # Data points (error bars)
    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=4, label="Data points")

    plt.title("Average Horizontal Range vs. Fin-Length-to-Diameter Ratio")
    plt.xlabel("Fin-Length-to-Diameter Ratio / L")
    plt.ylabel("Average Horizontal Range / R / m")

    plt.xticks(np.arange(0.0, 2.01, 0.25))
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(OUT / "range_vs_fin.png", dpi=200)
    plt.close()


def plot_velocity_decay_vs_fin(d: dict[str, pd.DataFrame]) -> None:
    """
    EE Graph 9.4 style:
    Best-Fit:      a = -1.70 L - 1.55
    Max slope:     a = -1.69 L - 1.54
    Min slope:     a = -1.88 L - 1.36
    """
    df = d["vel"].copy()
    x = df["fin_ratio"].to_numpy()
    y = df["avg_vel_decay_ms2"].to_numpy()
    yerr = df["uncertainty_ms2"].to_numpy()

    _ensure_outdir()
    plt.figure(figsize=(10, 6))

    xx = np.linspace(0.0, 2.0, 200)
    y_best = -1.70 * xx - 1.55
    y_max  = -1.69 * xx - 1.54
    y_min  = -1.88 * xx - 1.36

    plt.plot(xx, y_best, label="Best-Fit Line")
    plt.plot(xx, y_max,  linestyle="--", label="Max Slope")
    plt.plot(xx, y_min,  linestyle="--", label="Min Slope")

    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=4, label="Data points")

    plt.title("Average Velocity Decay vs. Fin-Length-to-Diameter Ratio")
    plt.xlabel("Fin-Length-to-Diameter Ratio / L")
    plt.ylabel(r"Average Velocity Decay / a / m s$^{-2}$")

    plt.xticks(np.arange(0.0, 2.01, 0.25))
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(OUT / "velocity_decay_vs_fin.png", dpi=200)
    plt.close()


def spearman_heatmap(d: dict[str, pd.DataFrame]) -> None:
    """
    EE Figure 9.4 heatmap style (3x3, annotated values):
    Variables:
      - Fin Length Ratio
      - Experimental Range (m)
      - Velocity Decay (m/s^2)
    """
    # Build dataset using corrected range for 0.75 and 1.00 (this matches the EE heatmap values)
    df_range = d["range"].copy()
    corr = d["outliers"][["fin_ratio", "new_avg_range_m"]]
    df_range = df_range.merge(corr, on="fin_ratio", how="left")
    df_range["avg_range_used_m"] = df_range["new_avg_range_m"].fillna(df_range["avg_range_m"])

    df = (
        df_range[["fin_ratio", "avg_range_used_m"]]
        .merge(d["vel"][["fin_ratio", "avg_vel_decay_ms2"]], on="fin_ratio")
        .rename(columns={
            "fin_ratio": "Fin Length Ratio",
            "avg_range_used_m": "Experimental Range (m)",
            "avg_vel_decay_ms2": r"Velocity Decay (m/s$^2$)",
        })
    )

    corr = df.corr(method="spearman")

    labels = list(corr.columns)
    n = len(labels)

    _ensure_outdir()
    plt.figure(figsize=(9, 6))
    im = plt.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")

    plt.title("Spearman Rank Correlation Heatmap")
    plt.xticks(np.arange(n), labels, rotation=0, ha="center")
    plt.yticks(np.arange(n), labels)

    # Draw white grid lines between cells (EE-style)
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate each cell with the correlation value (2 dp)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="white")

    plt.colorbar(im)
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
