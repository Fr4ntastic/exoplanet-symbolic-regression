# Exoplanet Symbolic Regression

**A symbolic regression mass–radius–irradiation relation for transiting gas giants**

Francesco Domenico Cancello — Independent Researcher, Naples, Italy
fcancell06@gmail.com

---

## Overview

This repository contains all analysis scripts, trained model checkpoints,
and derived data tables for the paper:

> **"A symbolic regression mass–radius–irradiation relation for transiting
> gas giants: temporal validation and comparison with the state of the art"**
> F. D. Cancello, submitted to *Astronomy & Astrophysics* (2026)

**Main result:** The symbolic regression formula

```
ln(Rp/R⊕) = 0.0567 * [ln(F/F⊕) - ln(Mp/M⊕) * (ln(Mp/M⊕) - 12.64)]
```

achieves R²=0.642 on 236 gas giants discovered after 2022
(never used in any stage of model development), outperforming all
existing mass-only empirical relations under a consistent calibration
protocol.

**Valid domain:** 0.3 ≲ Mp/MJup ≲ 30, 4 ≤ Rp/R⊕ ≤ 24,
irradiation 1 ≲ F/F⊕ ≲ 10⁵.

---

## Repository Structure

```
├── run_gassosi_pre2022.py       # Main PySR run (gas giants, pre-2022 training)
├── analisi_residui.py           # Residual analysis
├── test_nuova_legge.py          # Hybrid model test
├── valuta_formule_pre2022.py    # Evaluate all HoF formulas on post-2022 test
├── confronto_letteratura.py     # Comparison with literature models
├── bootstrap_coeffs.py          # Bootstrap CI for coefficients
├── collect_data.py              # NASA Archive data collection
├── results_gas_pre2022.csv      # PySR results (3 seeds)
├── tutte_formule_pre2022_r2.csv # R² for all Hall-of-Fame formulas
├── confronto_letteratura.csv    # R² comparison with literature
├── bootstrap_results.csv        # Bootstrap samples
├── requirements_gassosi.txt     # Dependencies
└── README.md
```

---

## How to Reproduce

### 1. Install dependencies

```bash
pip install pysr numpy pandas scipy matplotlib sympy scikit-learn
```

Julia is required by PySR: https://julialang.org/downloads/

### 2. Run PySR (takes ~14h on 8 CPUs)

```bash
python3 run_gassosi_pre2022.py
```

Downloads data from NASA Exoplanet Archive automatically,
splits at disc_year=2022 (511 training, 236 test), runs 3 seeds.

### 3. Evaluate formulas on post-2022 test set

```bash
python3 valuta_formule_pre2022.py
```

### 4. Compare with literature

```bash
python3 confronto_letteratura.py
```

### 5. Bootstrap coefficient uncertainties (~30 seconds)

```bash
python3 bootstrap_coeffs.py
```

---

## Main Formula

```python
import numpy as np

def predict_radius(Mp_earth, F_solar):
    """
    Predict ln(Rp/R_earth) for gas giant exoplanets.

    Parameters
    ----------
    Mp_earth : float or array — planetary mass in Earth masses
    F_solar  : float or array — irradiation in units of Earth's flux

    Returns
    -------
    logRp : ln(Rp/R_earth)

    Valid: 0.3 ≲ Mp/MJup ≲ 30,  F/F_earth ≲ 1e5
    """
    alpha = 0.056739  # 95% CI: [0.0539, 0.0607]
    beta  = 12.642    # 95% CI: [12.21, 13.12]
    logMp = np.log(Mp_earth)
    logF  = np.log(F_solar)
    return alpha * (logF - logMp * (logMp - beta))

# Example: Jupiter (318 M_earth) at 0.05 AU, F ~ 400 F_earth
print(f"Predicted: {np.exp(predict_radius(318, 400)):.2f} R_earth")
```

---

## Data

Retrieved from [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
(pscomppars table) on 2026 March 30 (UTC).

**Caveat:** pscomppars may include updated parameters for pre-2022 planets
from post-2022 publications. The temporal split is a robustness check,
not a strict causal separation. See Section 2.2 of the paper.

---

## Citation

```bibtex
@article{Cancello2026gasgiant,
  author  = {Cancello, Francesco Domenico},
  title   = {A symbolic regression mass--radius--irradiation relation
             for transiting gas giants},
  journal = {Astronomy \& Astrophysics},
  year    = {2026},
  note    = {submitted, arXiv:XXXX.XXXXX}
}
```

---

## License

MIT. See LICENSE file.
