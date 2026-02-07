# Changelog
All notable changes to this project will be documented in this file.

## [v1.0.0] - 2026-01-31
### Added
- Reproducible analysis pipeline for fin-ratio projectile study
- Data processing scripts and figure generation
- Quadratic-drag effective parameter fitting
- Bootstrap uncertainty estimation
- CI workflow for reproducibility checks
- Zenodo + CITATION metadata

## [v2.0.0] - 2026-02-07 
This release marks the transition from experimental observation to rigorous statistical validation. We have added a comprehensive diagnostic suite to confirm the **"Splitter Plate" regime shift** at $L/D \approx 1.0$ is a physical anomaly, not a stochastic artifact.

### 📊 Key Additions
* **Leave-One-Out (LOO) Diagnostics:** New outlier analysis confirming the $L/D=0.75$ drag peak is statistically significant with $p < 2 \times 10^{-4}$.
* **Spearman Rank Correlation:** Added heatmap visualization proving a perfect negative correlation ($\rho = -1.00$) between fin ratio and velocity decay.
* **Model Selection Audit:** Empirical RMSE comparison justifying the use of the Quadratic Drag model ($F_d \propto v^2$) over Linear Drag ($F_d \propto v$).

### 🛠️ Engineering & Reproducibility
* **Documentation:** Complete overhaul of `README.md` to meet Research Software Engineering (RSE) standards.
* **Pipeline:** Standardized `Makefile` DAG for automated reproduction of all figures.
* **Dependency Locking:** Pinned `requirements.lock.txt` for bit-for-bit reproducibility.

### 📦 Full Changelog
* Feat: Add bootstrap residual histograms for null model testing.
* Docs: Add theoretical framework (ODEs) and citation metadata.
* Test: Verify vacuum limit convergence in `test_physics.py` (<1% error).
