# Symbolic Regression of Empirical Mass–Radius Relations for Exoplanets

**An Irradiation-Dependent Law for Hot Jupiters**

*F. D. Cancello — Independent Researcher, Rome, Italy*
*Submitted to Monthly Notices of the Royal Astronomical Society (MNRAS), 2026*

---

## Overview

This repository contains the complete analysis pipeline for the study described in Cancello (2026, submitted). We apply symbolic regression via the [PySR](https://github.com/MilesCranmer/PySR) framework to 734 hot Jupiters drawn from the NASA Exoplanet Archive, searching for interpretable empirical mass–radius relations that incorporate stellar irradiation.

The main result is a novel empirical law for hot Jupiters:

$$\ln R_p = \ln M_p \left(0.0031 \ln F + 0.85\right)^{\ln M_p}$$

where $R_p$, $M_p$, and $F$ are planetary radius, mass, and stellar irradiation flux in Earth units. This relation achieves $R^2 = 0.669$ on a held-out test set, outperforming both a power-law baseline ($R^2 = 0.486$) and the three-regime formula of Chen & Kipping (2017) ($R^2 = 0.441$) on the same planets.

All scripts, trained model checkpoints, and derived data tables needed to reproduce every figure and table in the paper are provided here.

---

## Repository Structure

```
exoplanet-symbolic-regression/
│
├── data/
│   ├── pscomppars_2026-03-15.csv      # NASA Exoplanet Archive snapshot
│   └── metadata.json                  # MD5 checksum + download timestamp
│
├── scripts/
│   ├── 01_download_and_filter.py      # Data retrieval and quality cuts
│   ├── 02_feature_engineering.py      # Log-transforms and feature matrix
│   ├── 03_run_pysr.py                 # Symbolic regression (all groups, 3 seeds)
│   ├── 04_evaluate_models.py          # R², bootstrap CIs, permutation test
│   ├── 05_temporal_robustness.py      # Post-2022 robustness check
│   └── 06_make_figures.py             # Reproduce all paper figures
│
├── models/
│   ├── hot_jupiter_seed0.pkl          # PySR Pareto front, seed 0
│   ├── hot_jupiter_seed42.pkl         # PySR Pareto front, seed 42
│   ├── hot_jupiter_seed123.pkl        # PySR Pareto front, seed 123
│   ├── sub_neptune_seed*.pkl
│   ├── rocky_seed*.pkl
│   └── subterrestrial_seed*.pkl
│
├── figures/
│   ├── bootstrap_gas.png              # Figure 1 — bootstrap distributions
│   └── comparison_literature.png      # Figure 2 — model comparison
│
├── paper/
│   └── paper_final.tex                # Submission-ready LaTeX source
│
├── environment.yml                    # Conda environment specification
├── requirements.txt                   # pip-compatible dependency list
└── README.md
```

---

## Reproducing the Results

### 1. Clone the repository

```bash
git clone https://github.com/Fr4ntastic/exoplanet-symbolic-regression.git
cd exoplanet-symbolic-regression
```

### 2. Set up the environment

Using conda (recommended):

```bash
conda env create -f environment.yml
conda activate exoplanet-sr
```

Or using pip:

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
python scripts/01_download_and_filter.py
python scripts/02_feature_engineering.py
python scripts/03_run_pysr.py          # Warning: computationally intensive (~2–4 h on 8 cores)
python scripts/04_evaluate_models.py
python scripts/05_temporal_robustness.py
python scripts/06_make_figures.py
```

To skip the PySR search and use the pre-trained model checkpoints in `/models`, set `USE_CACHED_MODELS=True` in `scripts/03_run_pysr.py`. This reduces total runtime to under 5 minutes.

---

## Software Versions

| Package    | Version |
|------------|---------|
| Python     | 3.12    |
| PySR       | 1.5.9   |
| NumPy      | ≥ 1.26  |
| SciPy      | ≥ 1.13  |
| pandas     | ≥ 2.2   |
| matplotlib | ≥ 3.9   |

PySR requires a working Julia installation (≥ 1.9). On first run, Julia dependencies are installed automatically.

---

## Data

The input catalogue was downloaded from the NASA Exoplanet Archive Planetary Systems Composite Parameters table (`pscomppars`) via the TAP interface on **2026 March 15 (UTC)**:

```
https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars&format=csv
```

The MD5 checksum of the input CSV is recorded in `data/metadata.json`. Because the archive is updated continuously, using a different snapshot may yield slightly different sample statistics; the main conclusions are not expected to change.

After quality filtering (see Section 2 of the paper), the working sample contains **1112 planets**, of which **734 are hot Jupiters** ($R_p \geq 4\,R_\oplus$).

---

## Citation

If you use this code or the derived data products, please cite the accompanying paper:

```bibtex
@article{Cancello2026,
  author  = {Cancello, Francesco Domenico},
  title   = {Symbolic Regression of Empirical Mass--Radius Relations for Exoplanets:
             An Irradiation-Dependent Law for Hot Jupiters},
  year    = {2026},
  note    = {Submitted}
}
```

This work also makes use of:
- NASA Exoplanet Archive — [exoplanetarchive.ipac.caltech.edu](https://exoplanetarchive.ipac.caltech.edu)
- PySR — Cranmer M., 2023, [github.com/MilesCranmer/PySR](https://github.com/MilesCranmer/PySR)

---

## License

The source code in this repository is released under the **MIT License**.
Derived data products (filtered catalogues, model outputs) are released under **CC BY 4.0**.
Raw NASA Exoplanet Archive data is subject to its own [terms of use](https://exoplanetarchive.ipac.caltech.edu/docs/acknowledge.html).

---

## Contact

Francesco Domenico Cancello — fcancell06@gmail.com
