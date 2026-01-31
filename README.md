<p align="left">
  <a href="https://github.com/shubh-chawda/fin-ratio-projectile-aero/actions/workflows/ci.yml">
    <img src="https://github.com/shubh-chawda/fin-ratio-projectile-aero/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://doi.org/10.5281/zenodo.18443083">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.18443083.svg" alt="DOI (code)">
  </a>
</p>

# ☄️ Aerodynamic effects of fin length-to-diameter ratio on spherical projectile range & velocity decay

A **reproducible mini-research package** (📦 data + 🧠 physics + 🧪 modelling + 📈 figures + ✅ CI) based on my **IB Physics Extended Essay (May 2025)**.

**Preprint (Zenodo):** https://doi.org/10.5281/zenodo.18434071  
**Code archive (Zenodo):** https://doi.org/10.5281/zenodo.18443083

## ✨ What this repo does

This project studies how changing **fin length-to-diameter ratio** (**L/D**) on a spherical projectile influences:

- 🧭 **Horizontal range** vs fin ratio  
- 🌬️ **Velocity decay** vs fin ratio  
- 🧮 **Effective quadratic drag strength** (*inferred parameter* \(k_\mathrm{eff}\)) vs fin ratio  
- 🔗 **Spearman rank correlations** across key variables (non-parametric relationship scan)  
- 🎲 **Bootstrap uncertainty** to quantify confidence intervals in fitted parameters

> 🎯 Core idea: turn an EE into something closer to **research practice** — structured data, reproducible code, numerical modelling, uncertainty propagation, and automated verification.

## 🧠 Why I built this

Extended Essays often end as PDFs and DOCXs. I wanted the opposite:

✅ **Reproducibility**: start from raw tables → regenerate plots and fitted parameters  
✅ **Auditability**: anyone can verify what I did, not just trust the write-up  
✅ **Research workflow skills**: modelling + code structure + CI + archival DOI  
✅ **Presentation**: clear assumptions, numerical methods, uncertainty treatment

## 🧪 Physics & modelling assumptions (what’s actually being modelled)

This project sits at the intersection of **projectile motion** and **aerodynamic drag**, with fin geometry acting as a controlled design variable.

### Core physics concepts used
- **Newton’s 2nd law**: \(\vec{F}_\text{net}=m\vec{a}\)
- **Decomposition into components**: \(x\)-motion and \(y\)-motion coupled via speed \(v=\sqrt{v_x^2+v_y^2}\)
- **Gravity**: constant downward acceleration \(g\)
- **Quadratic drag** (dominant at moderate speeds):  
  \[
  \vec{F}_d = -k_\mathrm{eff}\, v\, \vec{v}
  \]
  where \(k_\mathrm{eff}\) is an **effective** parameter absorbing geometry + air density + drag coefficient + reference area.
- **Launch angle control**: \(\theta \approx 45^\circ\) (range-optimising baseline without drag; still a strong comparative angle under drag)
- **Parameter inference**: fit \(k_\mathrm{eff}\) such that simulation matches observed range
- **Uncertainty propagation**: bootstrap trials → distribution of fitted \(k_\mathrm{eff}\)

### What “effective” means here
Instead of separately estimating \(C_d\), cross-sectional area, and flow regime details, the model uses \(k_\mathrm{eff}\) as a compact way to represent **overall aerodynamic resistance** for each fin ratio.

> ✅ This is common in early-stage research and undergraduate modelling: infer an effective parameter first, then refine into deeper fluid-dynamics later.

## 🧮 Numerical method (how results are computed)

Analytic solutions for projectile range **with quadratic drag** are generally not clean/closed-form for the full 2D case, so this repo uses numerical integration.

### Integration scheme
- **RK4 (4th-order Runge–Kutta)** time-stepping for state:
  \[
  \text{state} = (x, y, v_x, v_y)
  \]
- Each step uses:
  \[
  \dot{x}=v_x,\quad \dot{y}=v_y,\quad
  \dot{v_x}= -\frac{k_\mathrm{eff}}{m} v v_x,\quad
  \dot{v_y}= -g -\frac{k_\mathrm{eff}}{m} v v_y
  \]
- **Ground-hit interpolation**: when \(y\) crosses 0, range is estimated by linear interpolation between the last positive and first non-positive step.

### Why RK4?
- stable + accurate for smooth ODEs
- widely used in physics simulation work
- “research standard” stepping method for first-pass modelling before advanced solvers

### Computational workflow
- Range trials → mean range per fin ratio
- Control condition used to estimate launch speed \(v_0\) (no-drag baseline):
  \[R_0 \approx \frac{v_0^2}{g} \;\Rightarrow\; v_0 \approx \sqrt{R_0 g}\]
- Then solve for \(k_\mathrm{eff}\) per fin ratio by matching simulated range to measured range (bisection on \(k\)).

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
