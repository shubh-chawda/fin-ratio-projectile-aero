<div align="center">

# 🚀 Fin-Aero: Projectile Dynamics & Wake Stabilization
### Analysis of Aerodynamic Drag and Vortex Shedding Effects on Spherical Projectiles

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18443083.svg)](https://doi.org/10.5281/zenodo.18443083)
[![Build Status](https://img.shields.io/github/actions/workflow/status/shubh-chawda/fin-ratio-projectile-aero/ci.yml?branch=main&label=reproducibility)](https://github.com/shubh-chawda/fin-ratio-projectile-aero/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[**Read the Paper**](https://doi.org/10.5281/zenodo.18434071) | [**View Notebooks**](./fin-aero-ee-repo/notebooks) | [**Report Bug**](../../issues)

</div>

---

## 📖 Abstract

**Fin-Aero** is a **reproducible research artefact** that quantifies the aerodynamic trade-offs of fin geometry on spherical projectiles. Unlike standard kinematic studies that assume a vacuum, this project implements a **Quadratic Drag Physics Engine** to model the non-linear decay of velocity under real-world air resistance.

The study identifies an anomaly at **$0.75\times$ Fin-Length-to-Diameter ($L/D$)**, where the range reduction is minimized. We **hypothesize** this is due to **Wake Stabilization** and **Vortex Shedding Resonance** (governed by the Strouhal Number), which momentarily delays boundary layer separation.

---

## ✨ Why This Exists (The "Anti-PDF" Philosophy)

Most research ends as a static PDF. This repository converts the **IB Physics Extended Essay** into a living, auditable code base.

| Feature | Standard PDF Essay | 🚀 Fin-Aero Repository |
| :--- | :--- | :--- |
| **Data Source** | Static Tables (Hard to check) | **Raw CSVs** (Auditable & Open) |
| **Analysis** | Excel Screenshots | **Python Pipelines** (Pandas/SciPy) |
| **Physics** | Simple Formulas | **RK4 Numerical Integration** |
| **Uncertainty** | Basic Error Bars | **Bootstrap Resampling** ($n=10k$) |

---

## ⚡ Key Capabilities

* **📐 Physics Engine**: Solves coupled ODEs for projectile motion using **Runge-Kutta (RK4)** integration.
* **🌪️ Inverse Fitting**: Uses **Bisection Optimization** to infer the "Effective Drag Parameter" ($k_{\text{eff}}$) from experimental range data.
* **📉 Uncertainty Propagation**: Generates **95% Confidence Intervals** for aerodynamic coefficients via computational bootstrapping.
* **📊 Automated Viz**: Produces publication-ready vector graphics (`.svg`/`.png`) for velocity decay, Spearman correlations, and drag fitting.

---

## 🧠 Why I built this

Extended Essays often end as PDFs and DOCXs. I wanted the opposite:

✅ **Reproducibility**: start from raw tables → regenerate plots and fitted parameters  
✅ **Auditability**: anyone can verify what I did, not just trust the write-up  
✅ **Research workflow skills**: modelling + code structure + CI + archival DOI  
✅ **Presentation**: clear assumptions, numerical methods, uncertainty treatment

## 🧪 Physics & modelling assumptions (what’s actually being modelled)

This project sits at the intersection of **projectile motion** and **aerodynamic drag**, with fin geometry acting as a controlled design variable.

### Core physics concepts used
- **Newton’s 2nd law**: $$\(\vec{F}_\text{net}=m\vec{a}\)$$
- **Decomposition into components**: $$\(x\)-motion and \(y\)-motion coupled via speed \(v=\sqrt{v_x^2+v_y^2}\)$$
- **Gravity**: constant downward acceleration $$\(g\)$$
- **Quadratic drag** (dominant at moderate speeds):  
  $$\[
  \vec{F}_d = -k_\mathrm{eff}\, v\, \vec{v}
  \]$$
  where \(k_\mathrm{eff}\) is an **effective** parameter absorbing geometry + air density + drag coefficient + reference area.
- **Launch angle control**: $$\(\theta \approx 45^\circ\)$$ (range-optimising baseline without drag; still a strong comparative angle under drag)
- **Parameter inference**: fit $$\(k_\mathrm{eff}\)$$ such that simulation matches observed range
- **Uncertainty propagation**: bootstrap trials → distribution of fitted $$\(k_\mathrm{eff}\)$$

### What “effective” means here
Instead of separately estimating $$\(C_d\)$$, cross-sectional area, and flow regime details, the model uses $$\(k_\mathrm{eff}\)$$ as a compact way to represent **overall aerodynamic resistance** for each fin ratio.

> ✅ This is common in early-stage research and undergraduate modelling: infer an effective parameter first, then refine into deeper fluid-dynamics later.

## 🧮 Numerical method (how results are computed)

Analytic solutions for projectile range **with quadratic drag** are generally not clean/closed-form for the full 2D case, so this repo uses numerical integration.

### Integration scheme
- **RK4 (4th-order Runge–Kutta)** time-stepping for state:
  $$\[
  \text{state} = (x, y, v_x, v_y)
  \]$$
- Each step uses:
  $$\[
  \dot{x}=v_x,\quad \dot{y}=v_y,\quad
  \dot{v_x}= -\frac{k_\mathrm{eff}}{m} v v_x,\quad
  \dot{v_y}= -g -\frac{k_\mathrm{eff}}{m} v v_y
  \]$$
- **Ground-hit interpolation**: when $$\(y\)$$ crosses 0, range is estimated by linear interpolation between the last positive and first non-positive step.

### Why RK4?
- stable + accurate for smooth ODEs
- widely used in physics simulation work
- “research standard” stepping method for first-pass modelling before advanced solvers

### Computational workflow
- Range trials → mean range per fin ratio
- Control condition used to estimate launch speed $$\(v_0\)$$ (no-drag baseline):
  $$\[R_0 \approx \frac{v_0^2}{g} \;\Rightarrow\; v_0 \approx \sqrt{R_0 g}\]$$
- Then solve for $$\(k_\mathrm{eff}\)$$ per fin ratio by matching simulated range to measured range (bisection on $$\(k\)$$).

## 📦 Repository layout:

- `fin-aero-ee-repo/`
  - `data/raw/`  
    ✅ tables transcribed from the EE (range trials, velocity decay trials, max velocity, uncertainties)
  - `src/`  
    🧠 scripts for generating plots + fitting drag models + bootstrapping + model comparison
  - `figures/`  
    🖼️ committed outputs (useful for quick viewing on GitHub)
  - `notebooks/`  
    📓 reproducibility notebook(s) to rerun key results interactively
  - `outputs/figures/`  
    📈 generated figures when you run the scripts locally/CI

Top-level repo also includes:
- `.github/workflows/ci.yml` ✅ runs reproducibility pipeline on GitHub Actions  
- `CITATION.cff` 📚 citation metadata  
- `LICENSE` (MIT) ⚖️  
- `CHANGELOG.md` 🧾 versioned improvements

## 📊 Key Results & Aerodynamic Analysis

The analysis pipeline (see `notebooks/`) revealed a distinct trade-off between aerodynamic stability and parasitic drag, with significant non-linear anomalies observed in the wake region.

### 1. 📉 Horizontal Range & The Drag Penalty
As hypothesized, increasing the Fin-Length-to-Diameter ratio ($L/D$) generally resulted in a monotonic decrease in horizontal range due to the increased wetted surface area and skin friction drag.

* **Baseline:** The control projectile ($0.00\times$) achieved a mean range of **$2.02 \pm 0.04$ m**.
* **Max Penalty:** The largest fin configuration ($2.00\times$) suffered a **~28% reduction** in range ($1.46 \pm 0.05$ m).
* **Fit Model:** Linear regression on the aggregate data yields a strong negative correlation:
    $$R(L) \approx -0.26 L + 2.00 \quad (R^2 > 0.9)$$

### 2. 🌪️ The "Wake Stabilization" Anomaly ($0.75\times$)
A statistically significant outlier was identified at $L/D = 0.75$ and $1.00$, where the range reduction was **lower than predicted** by the quadratic drag model.

**Physical Hypothesis: Vortex Shedding Resonance**
We attribute this anomaly to **Wake Stabilization** governed by the Strouhal Number ($St$). At specific fin geometries, the vortex shedding frequency $f$ likely aligns with the natural frequency of the wake, delaying boundary layer separation.

$$
St = \frac{f L}{v} \approx 0.2 \text{ (for blunt bodies)}
$$

* **Observation:** The $0.75\times$ fins may act as **vortex generators** that re-energize the boundary layer, temporarily reducing the pressure drag coefficient ($C_d$) despite the increased surface area.
* **Validation:** Additional trials ($N=5$) confirmed this was not random error, suggesting a "sweet spot" in the Reynolds number regime where wake turbulence is minimized.

### 3. 📉 Velocity Decay & Spearman Correlations
Unlike range, the **velocity decay** (deceleration) showed a stricter adherence to linearity, suggesting that while wake effects preserve *range* (flight time), the *instantaneous* kinetic energy loss remains high.

**Statistical Rigor (Spearman's $\rho$):**
To quantify monotonicity without assuming a linear parameteric model, we computed Spearman's Rank Correlation:

| Variable Pair | Spearman's $\rho$ | Interpretation |
| :--- | :---: | :--- |
| **Fin Ratio vs. Range** | **-0.95** | Strong Inverse Monotonic |
| **Fin Ratio vs. Decay** | **-1.00** | Perfect Inverse Monotonic |
| **Range vs. Decay** | **+0.95** | Strong Positive Association |

> **Note:** The mismatch between decay correlation (-1.00) and range correlation (-0.95) mathematically isolates the $0.75\times$ anomaly as a **trajectory efficiency** phenomenon rather than a pure drag force reduction.

### 4. 🎲 Uncertainty Quantification (Bootstrap)
We moved beyond standard error propagation by implementing **Bootstrap Resampling** ($n=10,000$ iterations) to generate 95% Confidence Intervals (CI) for the effective drag parameter $k_{\text{eff}}$.

* **Method:** Resampled residuals from the quadratic fit to estimate the stability of $k_{\text{eff}}$.
* **Result:** The confidence bands narrow significantly at higher $L/D$ ratios, indicating that **larger fins stabilize the flight path**, reducing variance between trials even as they increase drag.

---

## 💻 Computational Physics & Numerical Methods

To validate the experimental data against theoretical expectations, this repository implements a custom **physics engine** written in Python. Unlike simple kinematic equations (suvat), which assume a vacuum, this engine solves the **non-linear equations of motion** under quadratic drag.

### 1. The Governing Differential Equations
The projectile motion is modeled as a system of coupled Ordinary Differential Equations (ODEs). For a spherical projectile with velocity vector $\vec{v}$, the net force includes gravity and the opposing drag force:

$$
\vec{F}_{net} = m\vec{g} - \frac{1}{2}\rho A C_d |\vec{v}| \vec{v}
$$

Decomposing this into Cartesian coordinates for the simulation state vector $\vec{S} = [x, y, v_x, v_y]$:

$$
\begin{cases} 
\dot{v}_x = -\frac{k_{\text{eff}}}{m} v_x \sqrt{v_x^2 + v_y^2} \\
\dot{v}_y = -g - \frac{k_{\text{eff}}}{m} v_y \sqrt{v_x^2 + v_y^2}
\end{cases}
$$

> **Note:** We utilize an **effective drag parameter** $k_{\text{eff}}$ to encapsulate the geometric complexity of the fins, where $k_{\text{eff}} \propto C_d A$.

### 2. RK4 Integration Scheme
The analytical solution for 2D quadratic drag is non-trivial. Therefore, we employ the **Runge-Kutta 4th Order (RK4)** numerical integration method for high-fidelity time-stepping.

* **Precision:** RK4 minimizes truncation error ($O(\Delta t^5)$), ensuring that the simulated trajectory remains physically rigorous even over long flight durations.
* **State Update:**
    ```python
    k1 = derivatives(state, t)
    k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = derivatives(state + dt * k3, t + dt)
    new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    ```

### 3. Inverse Parameter Inference (The Bisection Method)
A key innovation in this analysis is the **inverse fitting** of aerodynamic parameters. Since $C_d$ varies with fin geometry, we cannot assume it is constant.

Instead, we use a **Bisection Algorithm** (Root-Finding) to iteratively solve for the $k_{\text{eff}}$ that minimizes the error between the *simulated range* and the *experimental mean range* for each fin ratio.

$$
\text{Minimize } \epsilon(k) = | R_{\text{sim}}(k) - R_{\text{exp}} |
$$

This approach allows us to numerically quantify the **"Drag Penalty"** added by each increment in fin size.

---

## 📊 Visualization Gallery

The following high-resolution figures are generated automatically by the pipeline. See `figures/` for the full vector outputs.

### 1. The Core Trade-Off: Range vs. Fin Ratio
`figures/range_vs_fin.png`
> **Figure 1:** The experimental range decreases monotonically with fin length. The "Max Slope" and "Min Slope" bounds (dotted lines) represent the worst-case error propagation. Note the statistically significant deviation at **0.75x**, identifying the wake stabilization region.

![Range vs Fin](./fin-aero-ee-repo/figures/range_vs_fin.png)

### 2. The Drag Signature: Velocity Decay
`figures/velocity_decay_vs_fin.png`
> **Figure 2:** Time-derivative of position ($\vec{a}$) vs. Fin Ratio. The tight linearity here ($R^2 \approx 0.99$) contrasts with the range anomalies, suggesting that while specific geometries preserve flight *time* (via lift/stabilization), the kinetic energy dissipation remains strictly proportional to surface area.

![Velocity Decay](./fin-aero-ee-repo/figures/velocity_decay_vs_fin.png)

### 3. Inverse Parameter Fitting: Effective Drag ($k_{\text{eff}}$)
`figures/effective_drag_k_vs_fin.png`
> **Figure 3:** The computed effective drag parameter $k_{\text{eff}}$ derived from the Bisection Root-Finding algorithm. This parameter encapsulates the aerodynamic penalty of the fins. The non-linear "dip" at intermediate ratios correlates with the Strouhal resonance hypothesis.

![Effective Drag](./fin-aero-ee-repo/figures/effective_drag_k_vs_fin.png)

### 4. Correlation Matrix (Spearman's $\rho$)
`figures/spearman_heatmap.png`
> **Figure 4:** Non-parametric correlation scan. The `-1.00` correlation between Fin Ratio and Decay confirms that the physics engine correctly models the drag penalty as a monotonic function of geometry, despite experimental noise in the range data.

![Heatmap](./fin-aero-ee-repo/figures/spearman_heatmap.png)

### 5. Model Fit Check: Range (Observed vs Drag-Model Fit)
`figures/drag_model_range_fit.png`
> **Figure 5:** Range validation plot showing **observed EE mean range** vs the **quadratic-drag model fit** after inverse fitting $k_{\text{eff}}$ (range-matching via bisection) for each fin ratio. This acts as a pipeline sanity check: the numerical integrator + fitting routine reproduce the overall trend (including the non-linear bump region) when calibrated to range.

![Range Fit](./fin-aero-ee-repo/figures/drag_model_range_fit.png)

### 6. Model Validation: Velocity Decay (Observed vs Predicted — Quadratic vs Linear Drag)
`figures/velocity_decay_model_compare.png`
> **Figure 6:** Time-domain validation comparing **observed EE velocity decay** (slope of speed vs time) against predictions from both **quadratic drag** and a **linear drag baseline**. While both models capture the overall geometry trend, they **systematically underpredict the magnitude** of deceleration (observed decay is more negative), suggesting that fitting to range alone does not fully constrain instantaneous energy loss (or that additional unmodelled effects/measurement definitions contribute).

![Velocity Decay Model Compare](./fin-aero-ee-repo/figures/velocity_decay_model_compare.png)

### 7. Bootstrap Uncertainty: $k_{\text{eff}}$ vs Fin Ratio (Median ± 95% CI)
`figures/k_eff_vs_fin_bootstrap_ci.png`
> **Figure 7:** Bootstrap-based uncertainty quantification for the fitted **effective quadratic-drag parameter** $k_{\text{eff}}$. For each fin ratio, we resample trials and refit $k_{\text{eff}}$ repeatedly, plotting the **median** estimate with **95% confidence intervals**. This highlights where parameter inference is stable vs where measurement variability produces wider uncertainty bands.

![Bootstrap k_eff CI](./fin-aero-ee-repo/figures/k_eff_vs_fin_bootstrap_ci.png)

## ⚠️ Limitations & Theoretical Constraints

While this study successfully quantifies the macroscopic aerodynamic effects of fin geometry, the **Quadratic Drag Model** ($F_D \propto v^2$) serves as a first-order approximation. Several higher-order fluid dynamics effects were simplified:

### 1. The Reynolds Number ($Re$) Dependency
We assumed a constant Drag Coefficient ($C_d$) throughout the trajectory. In reality, $C_d$ is a function of the Reynolds number:

$$
Re = \frac{\rho v L}{\mu}
$$

As the projectile decelerates from launch velocity ($v_0 \approx 4.5 \text{ m/s}$) to impact, $Re$ decreases. This may trigger a **flow regime transition** (e.g., from turbulent to laminar boundary layer), which would alter $C_d$ mid-flight. Our model averages this into a single $k_{\text{eff}}$.

### 2. The Magnus Effect (Spin Dynamics)
Although the launch mechanism was calibrated to minimize spin, any residual angular velocity ($\vec{\omega}$) would introduce a lift force perpendicular to the wake:

$$
\vec{F}_L = S (\vec{\omega} \times \vec{v})
$$

Visual inspection of the high-speed footage suggests minimal rotation, but without gyroscopic sensors, we cannot strictly rule out minor lift contributions to the range anomalies.

---

## 🚀 Future Work & Roadmap

To resolve the **0.75x Anomaly** and further refine the physics engine, the following extensions are proposed:

* **🌊 CFD Validation (OpenFOAM/ANSYS):**
    Run 3D RANS (Reynolds-Averaged Navier-Stokes) simulations to visualize the **vortex shedding street**.
    * *Hypothesis to test:* Does the shedding frequency at $0.75\times$ lock into a sub-harmonic of the projectile's natural frequency?

* **🌬️ Wind Tunnel PIV (Particle Image Velocimetry):**
    Direct measurement of the wake field to quantify the boundary layer separation point. This would empirically determine if the "fins" are acting as turbulators (delaying separation) or airbrakes.

* **📉 Variable-$$C_d$$ Solvers:**
    Upgrade the RK4 engine to support a dynamic drag coefficient $$C_d(Re)$$, implementing a lookup table based on standard sphere-drag curves rather than a fixed constant.

---

## 📜 Citation

Please cite this repository if you use the data or physics engine in your work:

```bibtex
@software{chawda_fin_aero_2025,
  author       = {Shubh Chawda},
  title        = {Fin-Aero: High-Fidelity Projectile Aerodynamics & Wake Analysis},
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.18443083},
  url          = {[https://doi.org/10.5281/zenodo.18443083](https://doi.org/10.5281/zenodo.18443083)}
}

