# Contributing to Projectile Aerodynamics Simulation

First off, thank you for considering contributing to this project! We welcome feedback, bug reports, and improvements to the simulation logic or documentation.

## 🛠️ Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/shubh-chawda/fin-ratio-projectile-aero.git](https://github.com/shubh-chawda/fin-ratio-projectile-aero.git)
cd fin-ratio-projectile-aero
```
### 2. Set Up the Environment
We use **Conda** (or Python `venv`) to manage dependencies. This ensures the physics engine and visualization tools (like Manim) work correctly.

#### Option A: Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate projectile-sim
```

#### Option B: Using Pip
```bash
pip install -r requirements.txt
```

### 3. Verify the Installation
Run the main reproduction script to ensure the physics engine is working:

```bash
python -m src.reproduce_key_results
```

## 📁 Project Structure

- `src/`: Core physics engine and simulation scripts.
- `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA).
- `data/`: Raw experimental data CSVs.
- `outputs/`: Generated figures, animations, and regression tables.

## 🐛 Found a Bug?

If you find a physics error or a code crash:

1. **Check existing issues** to see if it has already been reported.
2. **Open a new issue** including:
   - Steps to reproduce the error.
   - The command you ran.
   - Screenshots or error logs.

## 🚀 Submitting Changes

1. **Fork the repo** and create your branch from `main`.
2. **Make your changes.** If you modify the physics engine (`src/fit_drag_model.py`), please run the validation script to ensure results align with experimental data.
3. **Test your changes:**
```bash
# Run the simulation suite
python -m src.reproduce_key_results
```
4. **Open a Pull Request** (PR). Describe what you changed and why.

## 🎨 Code Style

- We follow **PEP 8** guidelines for Python code.
- Please add comments to complex mathematical functions (e.g., Runge–Kutta integration steps).
- Ensure generated figures are saved to `outputs/figures/`.

<p align="center"><b><span style="font-size: 1.2em;">❤️ Thank you for helping improve our project!</span></b></p>
