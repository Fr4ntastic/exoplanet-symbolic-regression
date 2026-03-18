# Symbolic Regression of Empirical Mass–Radius Relations for Exoplanets

**An Irradiation-Dependent Law for Hot Jupiters**

*F. D. Cancello — Independent Researcher, Naples, Italy*  
*Submitted to Monthly Notices of the Royal Astronomical Society (MNRAS), 2026*

---

## Overview

This repository contains the complete analysis pipeline for the study described in Cancello (2026, submitted). Symbolic regression via the [PySR](https://github.com/MilesCranmer/PySR) framework is applied to 734 hot Jupiters drawn from the NASA Exoplanet Archive, searching for interpretable empirical mass–radius relations that incorporate stellar irradiation.

The main result is a novel empirical law for hot Jupiters:

$$\ln R_p = \ln M_p \left(0.0031 \ln F + 0.85\right)^{\ln M_p}$$

where $R_p$, $M_p$, and $F$ are planetary radius, mass, and stellar irradiation flux in Earth units. This relation achieves $R^2 = 0.669$ on a held-out test set, outperforming both a power-law baseline ($R^2 = 0.486$) and the three-regime formula of Chen & Kipping (2017) ($R^2 = 0.441$) on the same planets.

---

## Repository Structure

```
exoplanet-symbolic-regression/
│
├── checkpoints_pysr/              # Trained PySR models (one folder per group/seed)
│   ├── gas_seed0/                 # Best seed for hot Jupiters
│   ├── sub_nep_seed0/             # Best seed for sub-Neptunes
│   ├── sub_terr_seed0/            # Best seed for sub-terrestrials
│   ├── rocky_SE_seed123/          # Best seed for rocky/super-Earths
│
├── run_sr.py                      # Main pipeline: data download, filtering,
│                                  # feature engineering, PySR search (3 seeds),
│                                  # bootstrap evaluation, permutation test
│
├── analyze_results.py             # Post-run analysis: bootstrap CIs (residual,
│                                  # pair, BCa, wild), domain checks, physical
│                                  # diagnostics, temporal robustness check,
│                                  # Chen & Kipping (2017) comparison
│
├── summary_finale.csv             # Final results table (R², CIs, formulas)
├── comparison_literature.csv      # Model comparison vs baselines
├── comparison_literature.png      # Figure 2 — bar chart model comparison
├── bootstrap_gas.png              # Figure 1 — bootstrap distributions (hot Jupiters)
├── metadata.json                  # MD5 checksum + download timestamp
├── requirements_run.txt           # Exact pip freeze from the analysis environment
└── README.md
```

---

## Reproducing the Results

### 1. Clone the repository

```bash
git clone https://github.com/Fr4ntastic/exoplanet-symbolic-regression.git
cd exoplanet-symbolic-regression
```

### 2. Install dependencies

```bash
pip install -r requirements_run.txt
```

PySR requires a working Julia installation (≥ 1.9). On first run, Julia dependencies are installed automatically via `juliacall`.

### 3. Run the symbolic regression pipeline

```bash
python3 run_sr.py
```

This downloads the NASA Exoplanet Archive catalogue, applies quality filters, runs PySR with 3 random seeds for each of the 4 planetary groups, evaluates all models on held-out test sets, and saves trained model checkpoints to `checkpoints_pysr/`.

> ⚠️ **Runtime warning**: the full pipeline takes approximately one day on 8 CPU cores. Set `PILOT = True` at the top of `run_sr.py` for a fast smoke test (≈ 5 minutes, reduced iterations).

The script supports checkpointing: if interrupted, it resumes from the last completed seed.

### 4. Run the post-hoc analysis

```bash
python3 analyze_results.py
```

This loads the trained models from `checkpoints_pysr/`, computes bootstrap confidence intervals, domain checks, physical diagnostics, and the temporal robustness check on post-2022 discoveries. All outputs are saved to `analyze_outputs/`.

To run with fewer bootstrap iterations (faster):

```bash
python3 analyze_results.py --n-boot 200
```

---

## Software Versions

The analysis was run on Ubuntu 24.04 with the following key packages:

| Package      | Version |
|--------------|---------|
| Python       | 3.12.3  |
| PySR         | 1.5.9   |
| NumPy        | 2.4.3   |
| SciPy        | 1.17.1  |
| pandas       | 2.3.3   |
| scikit-learn | 1.8.0   |
| matplotlib   | 3.10.8  |
| sympy        | 1.14.0  |

Full dependency list (including all transitive dependencies) is in `requirements_run.txt`.

---

## Data

The input catalogue was downloaded from the NASA Exoplanet Archive Planetary Systems Composite Parameters table (`pscomppars`) via the TAP interface on **2026 March 15 (UTC)**:

```
https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars&format=csv
```

The MD5 checksum of the input CSV is recorded in `metadata.json`. Because the archive is updated continuously, a different snapshot may yield slightly different sample statistics; the main conclusions are not expected to change.

After quality filtering (see Section 2 of the paper), the working sample contains **1112 planets**, of which **734 are hot Jupiters** ($R_p \geq 4\,R_\oplus$).

---

## Citation

If you use this code or the derived data products, please cite the accompanying paper:

```bibtex
@article{Cancello2026,
  author = {Cancello, Francesco Domenico},
  title  = {Symbolic Regression of Empirical Mass--Radius Relations for Exoplanets:
            An Irradiation-Dependent Law for Hot Jupiters},
  year   = {2026},
  note   = {Submitted}
}
```

This work also makes use of:
- NASA Exoplanet Archive — [exoplanetarchive.ipac.caltech.edu](https://exoplanetarchive.ipac.caltech.edu)
- PySR — Cranmer M., 2023, [github.com/MilesCranmer/PySR](https://github.com/MilesCranmer/PySR)
- Chen & Kipping (2017), ApJ, 834, 17 — comparison baseline

---

## License

The source code in this repository is released under the **MIT License**.  
Derived data products (filtered catalogues, model outputs) are released under **CC BY 4.0**.  
Raw NASA Exoplanet Archive data is subject to its own [terms of use](https://exoplanetarchive.ipac.caltech.edu/docs/acknowledge.html).

---

## Contact

Francesco Domenico Cancello — fcancell06@gmail.com
