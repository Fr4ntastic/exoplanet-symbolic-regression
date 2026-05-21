# Exoplanet Symbolic Regression

**A symbolic regression mass–radius–irradiation relation for transiting gas giants**

Francesco Domenico Cancello — Independent Researcher, Naples, Italy
fcancell06@gmail.com

---

## Paper

> **"A symbolic regression mass–radius–irradiation relation for transiting
> gas giants: temporal validation and comparison with the state of the art"**
> F. D. Cancello, submitted to *Astronomy & Astrophysics* (2026)

### Main result

```
ln(Rp/R⊕) = 0.0567 × [ln(F/F⊕) − ln(Mp/M⊕) × (ln(Mp/M⊕) − 12.64)]
```

- **R² = 0.642** on 236 gas giants discovered after 2022 (never seen during training)
- **ΔR² = +0.526** vs best existing model (Sousa et al. 2024)
- Generalisation gap **Δg = −0.016** (no overfitting)
- Bootstrap 95% CI: α = 0.0571⁺⁰·⁰⁰³⁵₋₀.₀₀₃₂, β = 12.65⁺⁰·⁴⁷₋₀.₄₄

**Valid domain:** 0.3 ≲ Mp/MJup ≲ 30, 4 ≤ Rp/R⊕ ≤ 24, 1 ≲ F/F⊕ ≲ 10⁵

---

## Repository contents

| File | Description |
|------|-------------|
| `run_gassosi_pre2022.py` | **Main script** — PySR run on pre-2022 gas giants, evaluated on post-2022 test set |
| `analisi_residui.py` | Residual analysis of baseline model |
| `test_nuova_legge.py` | Hybrid model test (PySR + linear correction) |
| `valuta_formule_pre2022.py` | Evaluate all Hall-of-Fame formulas on post-2022 test set |
| `confronto_letteratura.py` | Comparison with Müller+2024, Sousa+2024, CK17, Bashi+2017 |
| `bootstrap_coeffs.py` | Bootstrap confidence intervals for α and β |
| `collect_data.py` | NASA Exoplanet Archive data collection |
| `results_gas_pre2022.csv` | PySR results (3 seeds, pre-2022 training) |
| `tutte_formule_pre2022_r2.csv` | R² for all Hall-of-Fame formulas on post-2022 test |
| `confronto_letteratura.csv` | R² comparison with literature models |
| `bootstrap_results.csv` | Bootstrap samples for α, β, R² |
| `requirements_gassosi.txt` | Python/Julia dependencies |

---

## How to reproduce

### 1. Install dependencies

```bash
pip install pysr numpy pandas scipy matplotlib sympy scikit-learn
```

Julia is required by PySR: https://julialang.org/downloads/

### 2. Run PySR (~14h on 8 CPUs)

```bash
python3 run_gassosi_pre2022.py
```

Downloads data from NASA Exoplanet Archive automatically,
splits at `disc_year = 2022` (511 training, 236 test), runs 3 seeds.

### 3. Evaluate all Hall-of-Fame formulas

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

## Use the formula

```python
import numpy as np

def predict_radius(Mp_earth, F_solar):
    """
    Predict ln(Rp/R_earth) for transiting gas giants.

    Parameters
    ----------
    Mp_earth : float or array  — planetary mass in Earth masses
    F_solar  : float or array  — irradiation flux in Earth flux units (F⊕ ≈ 1361 W/m²)

    Returns
    -------
    logRp : ln(Rp/R_earth)

    Valid: 0.3 ≲ Mp/MJup ≲ 30,  F/F⊕ ≲ 1e5
    """
    alpha = 0.056739  # 95% CI: [0.0539, 0.0607]
    beta  = 12.642    # 95% CI: [12.21, 13.12]
    logMp = np.log(Mp_earth)
    logF  = np.log(F_solar)
    return alpha * (logF - logMp * (logMp - beta))

# Example: Jupiter-mass planet (318 M⊕) at F = 400 F⊕
Rp = np.exp(predict_radius(318, 400))
print(f"Predicted radius: {Rp:.2f} R⊕")
```

---

## Data

Raw data retrieved from the
[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
(`pscomppars` table) on 2026 March 30 (UTC)
([Akeson et al. 2013](https://doi.org/10.1086/672273);
[Christiansen et al. 2025](https://arxiv.org/abs/2506.03299)).

**Important caveat:** `pscomppars` combines parameters from multiple
references. The temporal split (`disc_year < 2022` for training) is a
*robustness check*, not a strict causal separation, as updated parameters
for pre-2022 planets may appear in post-2022 publications.

---

## Citation

```bibtex
@article{Cancello2026gasgiant,
  author  = {Cancello, Francesco Domenico},
  title   = {A symbolic regression mass--radius--irradiation relation
             for transiting gas giants},
  journal = {Astronomy \& Astrophysics},
  year    = {2026},
  note    = {submitted}
}
```

---

## License

MIT — see `LICENSE` file.
