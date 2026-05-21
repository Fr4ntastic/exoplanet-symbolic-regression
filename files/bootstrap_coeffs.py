"""
bootstrap_coeffs.py — Bootstrap degli intervalli di confidenza per i
coefficienti della formula PySR stabile:

  ln(Rp/R_earth) = alpha * [ln(F/F_earth) - ln(Mp/M_earth) * (ln(Mp/M_earth) - beta)]

dove alpha e beta vengono riffittati su 1000 campioni bootstrap del
training set pre-2022.

Uso: python3 ~/bootstrap_coeffs.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ============================================================
# Carica dati
# ============================================================
df = pd.read_csv('nasa_with_discyear.csv')
df = df.dropna(subset=['pl_masse','pl_masseerr1','pl_masseerr2',
                        'pl_rade','pl_radeerr1','pl_insol','disc_year'])
df = df[
    (df['pl_masse'] > 0) & (df['pl_masse'] < 4000) &
    (df['pl_rade']  >= 4.0) &
    (df['pl_insol'] > 0)
].copy()
df['logMp'] = np.log(df['pl_masse'])
df['logF']  = np.log(df['pl_insol'])
df['logRp'] = np.log(df['pl_rade'])
df['err_rel_max'] = (df[['pl_masseerr1','pl_masseerr2']]
                     .abs().max(axis=1) / df['pl_masse'])
df = df[df['err_rel_max'] < 0.30].copy()

# Training: solo pre-2022
train = df[df['disc_year'] < 2022].copy()
test  = df[df['disc_year'] >= 2022].copy()
print(f"Training: {len(train)}  Test: {len(test)}")

X_tr = np.column_stack([train['logMp'].values, train['logF'].values])
y_tr = train['logRp'].values
X_te = np.column_stack([test['logMp'].values,  test['logF'].values])
y_te = test['logRp'].values

# ============================================================
# Formula da fittare: logRp = alpha * (logF - logMp*(logMp - beta))
# Parametri: alpha, beta
# ============================================================
def formula(X, alpha, beta):
    logMp, logF = X[:, 0], X[:, 1]
    return alpha * (logF - logMp * (logMp - beta))

def safe_r2(y, yp):
    ss = np.sum((y - yp)**2)
    st = np.sum((y - y.mean())**2)
    return float(1 - ss/st) if st > 0 else float('nan')

# Fit su training completo
popt, pcov = curve_fit(formula, X_tr, y_tr, p0=[0.057, 12.64])
alpha_fit, beta_fit = popt
print(f"\nFit su training completo:")
print(f"  alpha = {alpha_fit:.6f}")
print(f"  beta  = {beta_fit:.6f}")
yp_te = formula(X_te, *popt)
print(f"  R²_test = {safe_r2(y_te, yp_te):.4f}")

# ============================================================
# Bootstrap: 1000 campioni con replacement dal training
# ============================================================
N_BOOT = 1000
rng    = np.random.default_rng(42)
alphas = np.zeros(N_BOOT)
betas  = np.zeros(N_BOOT)
r2s    = np.zeros(N_BOOT)

print(f"\nBootstrap ({N_BOOT} campioni)...")
for i in range(N_BOOT):
    idx    = rng.integers(0, len(X_tr), size=len(X_tr))
    X_b    = X_tr[idx]
    y_b    = y_tr[idx]
    try:
        p, _ = curve_fit(formula, X_b, y_b, p0=[0.057, 12.64])
        alphas[i] = p[0]
        betas[i]  = p[1]
        r2s[i]    = safe_r2(y_te, formula(X_te, *p))
    except Exception:
        alphas[i] = np.nan
        betas[i]  = np.nan
        r2s[i]    = np.nan

# Rimuovi fallimenti
ok = np.isfinite(alphas) & np.isfinite(betas)
alphas, betas, r2s = alphas[ok], betas[ok], r2s[ok]
print(f"  Campioni validi: {ok.sum()}/{N_BOOT}")

# ============================================================
# Risultati
# ============================================================
print(f"\n{'='*55}")
print("RISULTATI BOOTSTRAP (1000 campioni, intervalli 95%)")
print(f"{'='*55}")

for name, vals, fitted in [('alpha', alphas, alpha_fit),
                             ('beta',  betas,  beta_fit)]:
    lo, hi = np.percentile(vals, [2.5, 97.5])
    std    = np.std(vals)
    print(f"\n  {name} = {fitted:.6f}")
    print(f"    std bootstrap:      {std:.6f}")
    print(f"    95% CI:             [{lo:.6f}, {hi:.6f}]")
    print(f"    CI width / value:   {(hi-lo)/abs(fitted)*100:.1f}%")

lo_r2, hi_r2 = np.percentile(r2s, [2.5, 97.5])
print(f"\n  R²_test (bootstrap su coefficienti):")
print(f"    mean = {r2s.mean():.4f}  std = {r2s.std():.4f}")
print(f"    95% CI: [{lo_r2:.4f}, {hi_r2:.4f}]")

# ============================================================
# Testo LaTeX per il paper
# ============================================================
print(f"\n{'='*55}")
print("TESTO LATEX DA INSERIRE NEL PAPER:")
print(f"{'='*55}")
a_lo, a_hi = np.percentile(alphas, [2.5, 97.5])
b_lo, b_hi = np.percentile(betas,  [2.5, 97.5])
print(f"""
Bootstrap resampling (1000 samples) of the pre-2022 training
set yields 95\\% confidence intervals of
$\\alpha = {alpha_fit:.4f}^{{+{a_hi-alpha_fit:.4f}}}_{{-{alpha_fit-a_lo:.4f}}}$
and $\\beta = {beta_fit:.3f}^{{+{b_hi-beta_fit:.3f}}}_{{-{beta_fit-b_lo:.3f}}}$,
confirming that both coefficients are well-constrained by the data.
The out-of-sample $R^2$ evaluated on bootstrap-refitted coefficients
is ${r2s.mean():.3f} \\pm {r2s.std():.3f}$ (mean $\\pm$ std over 1000 samples),
consistent with the value of $0.642$ obtained with the PySR-fitted
coefficients.
""")

# Salva CSV
pd.DataFrame({'alpha': alphas, 'beta': betas, 'r2_test': r2s}).to_csv(
    'bootstrap_results.csv', index=False)
print("Salvato: bootstrap_results.csv")
