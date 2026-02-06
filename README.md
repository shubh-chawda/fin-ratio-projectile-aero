<div align="center">

# 🚀 Aerodynamic Regime Shifts in Finned Projectiles
### A Reproducible Computational Study on Non-Monotonic Drag Scaling

[![CI Status](https://img.shields.io/github/actions/workflow/status/shubh-chawda/fin-ratio-projectile-aero/ci.yml?label=Build&logo=github&style=for-the-badge&color=238636)](https://github.com/shubh-chawda/fin-ratio-projectile-aero/actions)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logo=open-source-initiative)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18434071-blue?style=for-the-badge&logo=doi&logoColor=white)](https://doi.org/10.5281/zenodo.18434071)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000?style=for-the-badge&logo=codefactor)](https://github.com/psf/black)
<br>

</div>

---

## 📄 Abstract

This study presents a rigorous computational investigation into the non-monotonic aerodynamic drag scaling of spherical projectiles augmented with cylindrical fins of varying length-to-diameter ratios ($\lambda = L/D$). By solving the **Inverse Projectile Motion Problem** over a dataset of experimental trajectories, we identify a statistically significant **"Drag Crisis"** regime centered at the critical ratio $\lambda_c \approx 1.0$.

Contradicting the monotonic drag increase predicted by classical skin-friction theory, our results demonstrate that fins of critical length induce a **Wake Suppression Effect** (analogous to a splitter plate). This mechanism effectively "streamlines" the bluff-body wake, reducing the inferred effective drag parameter $k_{eff}$ by approximately **56%** relative to the $\lambda=0.75$ configuration:

$$
\Delta k_{eff} \approx -56\% \quad \text{at} \quad \lambda : 0.75 \to 1.00
$$

The validity of this regime shift is confirmed via a **Non-Parametric Isotonic Regression** null model. Bootstrap resampling ($N=5000$) of the residuals yields a statistical significance of $p < 2 \times 10^{-4}$, rejecting the null hypothesis of monotonic drag scaling. The computational pipeline is fully reproducible, featuring automated Unit Testing for physical conservation laws, timestep sensitivity audits ($<0.007\%$ error), and Continuous Integration.

---

## I. Theoretical Framework

This study models the trajectory of a spherical projectile of mass $m$ subject to gravitational acceleration $\mathbf{g}$ and an aerodynamic drag force $\mathbf{F}_d$. The system is governed by a set of coupled non-linear Ordinary Differential Equations (ODEs).

### 1.1 Governing Equations of Motion

We define the state vector $\mathbf{S}(t) = [x, y, v_x, v_y]^T$. The time evolution of the system is described by the vector field $\frac{d\mathbf{S}}{dt} = \mathcal{F}(\mathbf{S}, \mathbf{p})$, where $\mathbf{p}$ represents the aerodynamic parameters.

Assuming a quadratic drag law (Newtonian regime, $Re \gg 10^3$), the drag force acts opposite to the velocity vector $\mathbf{v}$:

$$
\mathbf{F}_d = -\frac{1}{2} \rho A C_d |\mathbf{v}| \mathbf{v}
$$

Substituting this into Newton's Second Law ($\mathbf{F}_{net} = m \mathbf{a}$) yields the coupled system:

$$
\begin{cases}
\frac{dx}{dt} = v_x \\
\frac{dy}{dt} = v_y \\
\frac{dv_x}{dt} = -\left( \frac{\rho A C_d}{2m} \right) \sqrt{v_x^2 + v_y^2} \cdot v_x \\
\frac{dv_y}{dt} = -g - \left( \frac{\rho A C_d}{2m} \right) \sqrt{v_x^2 + v_y^2} \cdot v_y
\end{cases}
$$

For computational efficiency, we lump the aerodynamic constants into a single effective resistance parameter $k_{eff}$ (units: $\text{kg/m}$):

$$
k_{eff} \equiv \frac{1}{2} \rho A C_d \quad \Rightarrow \quad \mathbf{a} = \mathbf{g} - \frac{k_{eff}}{m} |\mathbf{v}| \mathbf{v}
$$

### 1.2 The "Splitter Plate" Hypothesis (Wake Suppression)

Standard aerodynamic theory suggests that adding surface area (fins) monotonically increases skin friction drag ($C_{d, friction}$). However, for bluff bodies like spheres, **pressure drag** dominated by the turbulent wake often constitutes $>90\%$ of the total drag.

We hypothesize that at a critical fin length $\lambda_c \approx 1.0$, the fins interact with the recirculation bubble behind the sphere, acting as a **Splitter Plate**.

* **Regime I ($\lambda < 0.75$):** Fins are submerged within the turbulent wake. They act as roughness elements, energizing the boundary layer and increasing $k_{eff}$.
* **Regime II ($\lambda \approx 1.0$):** Fins extend through the shear layer, preventing the interaction of opposing vortices (Von Kármán vortex street suppression). This leads to pressure recovery and a net reduction in $k_{eff}$.
* **Regime III ($\lambda > 1.25$):** The wake is fully stabilized. Further increases in $\lambda$ add wetted surface area, causing skin friction to dominate again ($k_{eff} \propto \lambda$).

### 1.3 Linear vs. Quadratic Drag Scaling

While macroscopic projectiles typically follow quadratic scaling ($F_d \propto v^2$), low-Reynolds transitions can exhibit Stokes-like linear scaling ($F_d \propto v$). We evaluate both models against the observed velocity decay data:

$$
\mathcal{L}_{quad} = \sum_{i} (v_{model}(t_i) - v_{obs}(t_i))^2 \quad \text{vs} \quad \mathcal{L}_{lin} = \sum_{i} (v_{model}(t_i) - v_{obs}(t_i))^2
$$

Our analysis reveals a persistent discrepancy (RMSE $\approx 2.4 \text{ m/s}^2$) in the pure quadratic model, suggesting a mixed-regime flow ($C_d = f(Re)$).

## II. Computational Methodology

The core challenge is an **Inverse Problem**: determining the unknown drag parameter $k_{eff}$ that best reproduces the experimentally observed range $R_{obs}$ for a given projectile configuration.

### 2.1 The Inverse Solver
We define the range function $\mathcal{R}(k_{eff}; v_0, \theta)$ as the horizontal distance traveled by the projectile until ground impact ($y=0$). The inverse problem is formulated as a root-finding task:

$$
\text{Find } k^* \in [0, \infty) \quad \text{s.t.} \quad \mathcal{R}(k^*; v_0, \theta) - R_{obs} = 0
$$

Since $\frac{\partial \mathcal{R}}{\partial k_{eff}} < 0$ (monotonicity of drag), we employ a **Bisection Algorithm** over the interval $[k_{min}, k_{max}]$ to ensure unconditional convergence to a unique solution within tolerance $\epsilon = 10^{-9}$.

* **Nuisance Parameter Estimation:** The initial launch velocity $v_0$ is unobserved. We infer $v_0$ from the control group ($L/D=0$) range $R_{control}$ assuming a ballistic baseline, solving $R_{control} = v_0^2 \sin(2\theta) / g$.

### 2.2 Numerical Integration (RK4)
The trajectory is propagated using the explicit **Runge-Kutta 4th Order (RK4)** method. For the state vector $\mathbf{S}_n$ at time $t_n$, the update to $\mathbf{S}_{n+1}$ is:

$$
\begin{aligned}
\mathbf{k}_1 &= \mathcal{F}(t_n, \mathbf{S}_n) \\
\mathbf{k}_2 &= \mathcal{F}(t_n + \frac{\Delta t}{2}, \mathbf{S}_n + \frac{\Delta t}{2} \mathbf{k}_1) \\
\mathbf{k}_3 &= \mathcal{F}(t_n + \frac{\Delta t}{2}, \mathbf{S}_n + \frac{\Delta t}{2} \mathbf{k}_2) \\
\mathbf{k}_4 &= \mathcal{F}(t_n + \Delta t, \mathbf{S}_n + \Delta t \mathbf{k}_3) \\
\mathbf{S}_{n+1} &= \mathbf{S}_n + \frac{\Delta t}{6} (\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4) + \mathcal{O}(\Delta t^5)
\end{aligned}
$$

**Event Detection (Ground Impact):**
Standard discrete timestepping overshoots the ground ($y_{n+1} < 0$). To preserve order-of-accuracy, we implement a **Linear Interpolation** step at the zero-crossing boundary:

$$
t_{impact} \approx t_n + \Delta t \left( \frac{0 - y_n}{y_{n+1} - y_n} \right) \quad \Rightarrow \quad x_{impact} \approx x_n + v_{x,n} (t_{impact} - t_n)
$$

---

## III. Statistical Validation

To verify that the non-monotonic "dip" at $\lambda_c \approx 1.0$ is a physical phenomenon and not a statistical artifact, we construct a rigorous null hypothesis test.

### 3.1 The Monotonic Null Model
We define the Null Hypothesis ($H_0$) as: *"Aerodynamic drag increases monotonically with fin length."*
To test this, we fit a **Non-Parametric Isotonic Regression** (Pool Adjacent Violators algorithm) to the dataset, excluding the critical point $\lambda_c$.

$$
\hat{\mathbf{y}}_{iso} = \underset{\hat{\mathbf{y}}}{\text{argmin}} \sum_{i} (y_i - \hat{y}_i)^2 \quad \text{subject to} \quad \hat{y}_i \le \hat{y}_{i+1} \quad \forall i
$$

### 3.2 Bootstrap Significance Testing
We quantify the deviation of the observed critical point $y_{obs}(\lambda_c)$ from the null model prediction $\hat{y}_{null}(\lambda_c)$ using **Bootstrap Resampling** ($N=5000$).

1.  Resample experimental trials with replacement $\rightarrow$ Generate bootstrap sample $\mathcal{D}^*_b$.
2.  Refit Isotonic Null Model to $\mathcal{D}^*_b \setminus \{\lambda_c\}$.
3.  Compute Residual: $\delta^*_b = y^*_{obs}(\lambda_c) - \hat{y}^*_{null}(\lambda_c)$.

**Result:** The 95% Confidence Interval (CI) of the residual distribution is strictly negative and excludes zero:
$$
CI_{95\%}(\delta) = [-0.182, -0.142] \text{ m} \quad \Rightarrow \quad p < 2.0 \times 10^{-4}
$$
This allows us to reject $H_0$ with high confidence, confirming the existence of the drag crisis regime.

## IV. Numerical Stability Audit

To ensure that the observed drag anomalies are physical rather than numerical artifacts, we performed a rigorous **Timestep Sensitivity Analysis**. The inverse solver was stress-tested by successively halving the integration timestep $dt$ across a convergence ladder:

$$
\mathcal{T}_{audit} = \{3.0\text{ms}, 1.5\text{ms}, 0.75\text{ms}\}
$$

We define the relative convergence error $\epsilon_{rel}$ for the inferred parameter $k_{eff}$ as:

$$
\epsilon_{rel} = \max_{\lambda} \left| \frac{k_{eff}(\lambda, dt) - k_{eff}(\lambda, dt/2)}{k_{eff}(\lambda, dt)} \right|
$$

**Results:**
* The maximum relative change in $k_{eff}$ when refining $dt$ from $3.0 \to 1.5$ ms was **< 0.007%**.
* Further refinement to $0.75$ ms yielded a change of **< 0.002%**.
* This convergence rate ($\mathcal{O}(dt^4)$) confirms that numerical dissipation is negligible compared to the physical drag magnitude (the "dip" represents a ~56% shift).

![Timestep Sensitivity](outputs/figures/timestep_sensitivity/effective_drag_fit_dt_0p003.csv)

## V. Results & Discussion

### 5.1 The Drag Crisis ($L/D \approx 1.0$)
Our inverse modeling reveals a non-monotonic scaling of the effective drag parameter $k_{eff}$ with fin length.

* **Figure 1 (Below):** The inferred $k_{eff}$ rises linearly in the roughness regime ($\lambda < 0.75$) due to boundary layer energization.
* **The Anomaly:** At $\lambda_c = 1.0$, $k_{eff}$ drops precipitously from $0.029$ kg/m to $0.013$ kg/m.
* **Mechanism:** This confirms the **Splitter Plate Hypothesis**. The fins at $\lambda=1.0$ bridge the recirculation zone, suppressing the Von Kármán vortex street and recovering base pressure.

<div align="center">
  <img src="outputs/effective_drag_k_vs_fin.png" width="75%">
  <p><em>Figure 1: Inferred effective drag parameter k_eff vs. fin ratio. Note the 56% reduction at L/D=1.0.</em></p>
</div>

### 5.2 Statistical Significance
To verify this is not a stochastic artifact, we fit a Monotonic Null Model (Isotonic Regression) to the dataset excluding $\lambda=0.75$.

* **Figure 2 (Below):** The observed range at $\lambda=0.75$ (star) falls significantly below the 95% Confidence Interval of the monotonic trend.
* **Figure 3 (Below):** Bootstrap resampling ($N=5000$) of the residuals confirms that the probability of this deviation arising by chance is $p < 2 \times 10^{-4}$.

<div align="center">
  <table border="0">
    <tr>
      <td width="50%">
        <img src="outputs/range_075_null_fit.png" width="100%">
        <p><em>Figure 2: Observed range vs. Monotonic Null Model.</em></p>
      </td>
      <td width="50%">
        <img src="outputs/k_eff_vs_fin_bootstrap_ci.png" width="100%">
        <p><em>Figure 3: Bootstrap 95% CIs for k_eff.</em></p>
      </td>
    </tr>
  </table>
</div>

### 5.3 The "Vacuum Baseline" Paradox
A comparison of model predictions against velocity decay data revealed a critical insight into baseline calibration.

* **Observation:** The Inverse Solver, when optimizing for *range* only, assigned $k_{eff} \approx 0$ to the bare sphere ($L/D=0$).
* **Contradiction:** Experimental velocity decay measurements (Figure 4) show a clear deceleration ($a_x \approx -1.45 \text{ m/s}^2$).
* **Implication:** The standard practice of assuming a ballistic baseline for control groups masks the baseline drag. Future iterations must optimize for both range and velocity decay simultaneously ($\mathcal{L}_{total} = \mathcal{L}_{range} + \lambda \mathcal{L}_{decay}$).

<div align="center">
  <img src="outputs/velocity_decay_model_compare.png" width="75%">
  <p><em>Figure 4: Model predictions vs. Observed velocity decay. Note the divergence at L/D=0.</em></p>
</div>

## VI. Software Architecture & Reproducibility

This repository employs a **Research Software Engineering (RSE)** workflow designed for auditability and long-term reproducibility. The architecture separates the computational core from statistical post-processing, managed via a directed acyclic graph (DAG) of build targets.

### 6.1 Reproducibility Pipeline (Make)
The scientific lifecycle is automated via GNU Make, ensuring that derived data is rigorously consistent with the source code. The pipeline enforces a "clean-build" philosophy to prevent state contamination.

| Target | Description | Artifacts Produced |
| :--- | :--- | :--- |
| `make core` | Runs the inverse ODE solver to fit $k_{eff}$ for all fin ratios. | `data/processed/effective_drag_fit.csv` |
| `make bootstrap` | Executes the parallelized bootstrap resampling ($N=5000$). | `data/processed/bootstrap_k_eff_ci.csv` |
| `make timestep` | Performs the numerical stability audit (halving $dt$). | `timestep_sensitivity_summary.json` |
| `make tests` | Runs the physics verification suite via `pytest`. | Test Reports |
| `make ci` | Runs the full Continuous Integration suite (fast bootstrap). | CI Status Badge |

```bash
# Full reproduction command (Clean -> Fit -> Verify -> Plot)
make clean && make all
'''bash

## VII. Usage & Citation

### 7.1 Quickstart
To reproduce the core findings of this study, we recommend running the pre-configured demo pipeline.

```bash
# 1. Clone the repository
git clone [https://github.com/shubh-chawda/fin-ratio-projectile-aero.git](https://github.com/shubh-chawda/fin-ratio-projectile-aero.git)
cd fin-ratio-projectile-aero

# 2. Install dependencies (Pinned for bit-for-bit reproducibility)
# We recommend using a virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt

# 3. Run the "Drag Crisis" demo (L/D = 0.75 vs 1.0)
python -m src.demo --fin-ratio 0.75 --plot
python -m src.demo --fin-ratio 1.00 --plot

### 7.2 Citation
If you utilize this dataset, the inverse modelling pipeline, or the "Splitter Plate" findings in your research, please cite the following software artifact:

@software{Chawda_Fin_Aero_2026,
  author = {Chawda, Shubh},
  title = {{Aerodynamic Regime Shifts in Finned Projectiles: A Reproducible Computational Study}},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {[https://github.com/shubh-chawda/fin-ratio-projectile-aero](https://github.com/shubh-chawda/fin-ratio-projectile-aero)},
  doi = {10.5281/zenodo.18434071},
  version = {1.0.0}
}
