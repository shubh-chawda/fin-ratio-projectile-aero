# Aerodynamic Regime Shifts in Finned Projectiles

[![CI](https://github.com/shubh-chawda/fin-ratio-projectile-aero/actions/workflows/ci.yml/badge.svg)](https://github.com/shubh-chawda/fin-ratio-projectile-aero/actions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18434071.svg)](https://doi.org/10.5281/zenodo.18434071)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the source code and data for a computational study on the aerodynamics of spherical projectiles with varying fin-length-to-diameter ratios ($L/D$).

Using an **inverse modelling approach**, this project infers effective drag parameters from experimental range data. The study identifies a non-monotonic relationship between fin length and drag, specifically highlighting a regime where fins of length $L \approx D$ significantly reduce drag compared to shorter fins ($L \approx 0.75D$).

The pipeline is designed for **reproducibility**, featuring automated unit tests for physical laws, bootstrap uncertainty quantification, and numerical stability audits.

---

## Key Results

### 1. The "Splitter Plate" Effect
Experimental data shows that drag does not increase monotonically with fin surface area. Our inverse solver identified a distinct drop in the effective drag parameter ($k_{eff}$) at the critical ratio $L/D=1.0$.

* **Observation:** Increasing fin length from $0.75D$ to $1.0D$ resulted in a **~56% reduction** in the fitted drag parameter ($0.029 \to 0.013$ kg/m).
* **Hypothesis:** This suggests a wake suppression mechanism where the fins act as a splitter plate, preventing the interaction of shear layers behind the sphere.

<p align="center">
  <img src="outputs/effective_drag_k_vs_fin.png" width="70%">
  <br>
  <em>Figure 1: The effective drag parameter inferred from range data. Note the sharp reduction at L/D=1.0.</em>
</p>

### 2. Statistical Significance
To determine if the range drop at $L/D=0.75$ (and subsequent rebound) was a statistical artifact, we tested the data against a **Monotonic Null Model** (Isotonic Regression).

* **Method:** We performed a bootstrap analysis ($N=5000$) to compare the observed range at $L/D=0.75$ against the range predicted by a strictly monotonic trend.
* **Result:** The observed residual was negative and statistically significant ($p < 2 \times 10^{-4}$), with the 95% confidence interval entirely excluding zero.

<p align="center">
  <img src="outputs/range_075_null_fit.png" width="70%">
  <br>
  <em>Figure 2: Observed range vs. the Monotonic Null Model. The observation at 0.75x (star) deviates significantly from the null hypothesis.</em>
</p>

### 3. Model Limitations (Velocity Decay)
While the quadratic drag model ($F_d \propto v^2$) fits the range data well, a comparison against instantaneous velocity decay measurements reveals discrepancies.

* **The Paradox:** The model accurately predicts total range but underestimates the deceleration for the bare sphere ($L/D=0$).
* **Implication:** This suggests that the "vacuum baseline" assumption used to calibrate initial velocity ($v_0$) may mask baseline drag effects. Future work should implement multi-objective optimization to fit both range and decay simultaneously.

<p align="center">
  <img src="outputs/velocity_decay_model_compare.png" width="70%">
  <br>
  <em>Figure 3: Discrepancy between model predictions (orange/green) and observed velocity decay (blue).</em>
</p>

---

## Computational Methodology

To infer the aerodynamic properties from the raw range data, we solved an **Inverse Problem**: finding the optimal drag coefficient that minimizes the residual between simulated and observed trajectories.

### 1. The Inverse Solver
We define the range function $\mathcal{R}(k_{eff})$ as the horizontal distance traveled by the projectile until ground impact ($y=0$). Since aerodynamic drag monotonically reduces range ($\frac{d\mathcal{R}}{dk} < 0$), we use a **Bisection Root-Finding Algorithm** to solve:

$$
\text{Find } k^* \text{ such that } |\mathcal{R}(k^*) - R_{obs}| < \epsilon
$$

* **Nuisance Parameters:** The initial launch velocity $v_0$ is unobserved. We infer it from the control group ($L/D=0$) assuming a ballistic baseline ($k \approx 0$), solving $v_0 = \sqrt{R_{control} \cdot g / \sin(2\theta)}$.

### 2. Forward Simulation (RK4)
The trajectory is propagated using a custom implementation of the **Runge-Kutta 4th Order (RK4)** method for the coupled system of ODEs:

$$
\mathbf{a}(t) = \mathbf{g} - \frac{k_{eff}}{m} \|\mathbf{v}\| \mathbf{v}
$$

* **Event Detection:** Standard time-stepping often overshoots the ground ($y < 0$). To ensure precision, we implement a **Linear Interpolation** step at the zero-crossing boundary to determine the exact impact coordinate $x_{impact}$.

---

## Reproducibility & Verification

This project adheres to **Research Software Engineering (RSE)** standards. The entire pipeline—from raw data to final figures—is automated and verified.

### 1. Automated Pipeline (Make)
We use a `Makefile` to define a Directed Acyclic Graph (DAG) of build targets, ensuring that all results are traceable to the source code.

| Command | Description | Artifact |
| :--- | :--- | :--- |
| `make core` | Runs the inverse solver for all fin ratios. | `data/processed/effective_drag_fit.csv` |
| `make bootstrap` | Runs the statistical significance tests ($N=5000$). | `data/processed/bootstrap_k_eff_ci.csv` |
| `make timestep` | Runs the numerical stability audit. | `timestep_sensitivity_summary.json` |

### 2. Physics Verification (Unit Tests)
Before processing experimental data, the physics engine is validated against analytical laws using `pytest`.

* **Vacuum Limit Test:** Verifies that when drag is zero ($k=0$), the numerical solver matches the analytical projectile equation ($R = v^2/g$) within **1% tolerance**.
* **Monotonicity Test:** Asserts that increasing the drag parameter always results in a shorter range, preventing logical errors in the drag force direction.

### 3. Numerical Stability Audit
To rule out numerical artifacts, we performed a **Timestep Sensitivity Analysis**. We refined the integration step $dt$ from $3.0\text{ms} \to 1.5\text{ms}$ and measured the drift in the inferred parameter $k_{eff}$.

* **Result:** The relative change in $k_{eff}$ was **< 0.007%**, confirming that the observed "drag dip" (a ~50% effect) is physically robust and not a discrete integration error.

---

## Usage

### 1. Installation
To reproduce the findings, first clone the repository and install the dependencies. We use `requirements.lock.txt` to guarantee bit-for-bit reproducibility of the scientific environment.

```
# Clone the repository
git clone [https://github.com/shubh-chawda/fin-ratio-projectile-aero.git](https://github.com/shubh-chawda/fin-ratio-projectile-aero.git)
cd fin-ratio-projectile-aero

# Create and activate a virtual environment (Recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install exact dependencies
pip install -r requirements.lock.txt

## 2. Running the "Drag Crisis" Demo
```

### 2. Demo
You can visually compare the trajectory of the **Roughness Regime** *(L/D = 0.75)* vs. the **Splitter Plate Regime** *(L/D = 1.00)* using the CLI demo.

```
# Simulate the high-drag case (Roughness Regime)
python -m src.demo --fin-ratio 0.75 --plot

# Simulate the low-drag case (Splitter Plate Regime)
# Note: Range increases significantly despite the longer fin
python -m src.demo --fin-ratio 1.00 --plot
```

---

## Citation

If you utilize this dataset, please cite this software artifact:

```bibtex
@software{Chawda_Fin_Aero_2026,
  author    = {Chawda, Shubh},
  title     = {{Aerodynamic Regime Shifts in Finned Projectiles: A Reproducible Computational Study}},
  year      = {2026},
  publisher = {GitHub},
  journal   = {GitHub repository},
  url       = {https://github.com/shubh-chawda/fin-ratio-projectile-aero},
  doi       = {10.5281/zenodo.18434071},
  version   = {1.0.0}
}
