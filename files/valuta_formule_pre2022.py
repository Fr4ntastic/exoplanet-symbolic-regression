"""
valuta_formule_pre2022.py — Valuta TUTTE le formule della Hall of Fame
di run_gassosi_pre2022 sul test set post-2022 (out-of-sample reale).

Uso: python3 ~/valuta_formule_pre2022.py
"""

import numpy as np
import pandas as pd
import sympy as sp
from sklearn.linear_model import LinearRegression

# ============================================================
# Carica dati e prepara test set post-2022
# ============================================================
df = pd.read_csv('nasa_with_discyear.csv')
df = df.dropna(subset=['pl_masse','pl_masseerr1','pl_masseerr2','pl_rade',
                        'pl_radeerr1','pl_insol','st_met','st_mass','st_rad',
                        'pl_orbper','disc_year'])
df = df[
    (df['pl_masse'] > 0) & (df['pl_masse'] < 4000) &
    (df['pl_rade']  >= 4.0) &
    (df['pl_insol'] > 0) & (df['st_mass'] > 0) &
    (df['st_rad']   > 0) & (df['pl_orbper'] > 0)
].copy()

df['logMp']       = np.log(df['pl_masse'])
df['logF']        = np.log(df['pl_insol'])
df['FeH']         = df['st_met']
df['logMs']       = np.log(df['st_mass'])
df['logRs']       = np.log(df['st_rad'])
df['logP']        = np.log(df['pl_orbper'])
df['logRp']       = np.log(df['pl_rade'])
df['log_rhostar'] = df['logMs'] - 3*df['logRs']
df['ecc']         = df['pl_orbeccen'].fillna(0.0).clip(0,0.99) \
                    if 'pl_orbeccen' in df.columns else 0.0
df['log1p_ecc']   = np.log1p(df['ecc'])
df['err_rel_max'] = df[['pl_masseerr1','pl_masseerr2']].abs().max(axis=1) \
                    / df['pl_masse']
df = df[df['err_rel_max'] < 0.30].copy()

SPLIT_YEAR = 2022
df_train = df[df['disc_year'] <  SPLIT_YEAR].copy()
df_test  = df[df['disc_year'] >= SPLIT_YEAR].copy()
print(f"Train (pre-{SPLIT_YEAR}): {len(df_train)}  |  Test (post-{SPLIT_YEAR}): {len(df_test)}")

y_te   = df_test['logRp'].values
y_tr   = df_train['logRp'].values

# ============================================================
# Simboli sympy
# ============================================================
logMp_s, logF_s, FeH_s, logP_s, logRhoStar_s, log1p_ecc_s = sp.symbols(
    'logMp logF FeH logP logRhoStar log1p_ecc')
syms = (logMp_s, logF_s, FeH_s, logP_s, logRhoStar_s, log1p_ecc_s)

def get_vals(df_sub):
    return (df_sub['logMp'].values, df_sub['logF'].values,
            df_sub['FeH'].values,   df_sub['logP'].values,
            df_sub['log_rhostar'].values, df_sub['log1p_ecc'].values)

vals_te = get_vals(df_test)
vals_tr = get_vals(df_train)

def safe_r2(y, yp):
    yp = np.asarray(yp, dtype=float).ravel()
    if yp.size == 1: yp = np.full(len(y), float(yp[0]))
    if not np.all(np.isfinite(yp)): return float('nan')
    ss = np.sum((y-yp)**2); st = np.sum((y-y.mean())**2)
    return float(1-ss/st) if st > 0 else float('nan')

def eval_f(expr_str, vals):
    try:
        expr = sp.sympify(expr_str)
        f    = sp.lambdify(syms, expr, 'numpy')
        yp   = f(*vals)
        return np.asarray(yp, dtype=float).ravel()
    except Exception as e:
        return None

# ============================================================
# TUTTE le formule dai tre seed (dalla Hall of Fame nei log)
# ============================================================
formulas = {
    # ----- SEED 0 -----
    ('s0', 1):  '2.524',
    ('s0', 3):  'logMp**0.52378',
    ('s0', 4):  'log(logMp + logF)',
    ('s0', 5):  '(logF + logMp)**0.37608',
    ('s0', 7):  '0.87108**logMp*(logMp - log1p_ecc)',
    ('s0', 8):  '-2.7914/logMp + log(logF + 14.664)',
    ('s0', 9):  '(logF - (logMp + (-12.642))*logMp)*0.056739',
    ('s0', 11): '(1.2939**logF - (logMp + (-12.4))*logMp)*0.059803',
    ('s0', 13): 'logMp*(1.3552**logF*(-0.0010312) + (-0.061921))*(logMp + (-12.297))',
    ('s0', 15): '(1.258**logF - (logRhoStar + (logMp + (-12.596))*(logMp - log1p_ecc)))*0.059514',
    ('s0', 17): '(1.2567**logF - (FeH + ((logMp + (-12.591))*(logMp - log1p_ecc) + logRhoStar)))*0.059766',
    ('s0', 19): '((((logMp - FeH) + (-12.296))*(logMp - (FeH + log1p_ecc)) + logRhoStar) - 1.2513**logF)*(-0.062456)',
    # ----- SEED 42 -----
    ('s42', 1):  '2.524',
    ('s42', 3):  'logMp**0.52376',
    ('s42', 4):  'log(logF + logMp)',
    ('s42', 5):  '(logF + logMp)**0.37608',
    ('s42', 7):  '0.87108**logMp*(logMp - log1p_ecc)',
    ('s42', 8):  '-2.7921/logMp + log(logF + 14.666)',
    ('s42', 9):  '((logMp + (-12.642))*logMp - logF)*(-0.05674)',
    ('s42', 11): '(logMp*(logMp + (-12.4)) - 1.2939**logF)*(-0.059801)',
    ('s42', 13): '(((logMp + (-12.427))*logMp + logRhoStar) - 1.2759**logF)*(-0.059687)',
    ('s42', 15): '(((logMp - log1p_ecc)*(logMp + (-12.595)) + logRhoStar) - 1.258**logF)*(-0.059515)',
    ('s42', 17): '((((logMp + (-12.589))*(logMp - log1p_ecc) + FeH) + logRhoStar) - 1.2566**logF)*(-0.059778)',
    ('s42', 19): '(1.2513**logF - (((logMp - log1p_ecc) - FeH)*((logMp + (-12.297)) - FeH) + logRhoStar))*0.062445',
    # ----- SEED 123 -----
    ('s123', 1):  '2.524',
    ('s123', 3):  'logMp**0.52377',
    ('s123', 4):  'log(logMp + logF)',
    ('s123', 5):  '(logMp + logF)**0.37608',
    ('s123', 7):  '0.87108**logMp*(logMp - log1p_ecc)',
    ('s123', 8):  'log(logF + 14.666) + (-2.7922/logMp)',
    ('s123', 9):  '(0.84773**logMp)*(1.0249**logF)*logMp',
    ('s123', 10): '(log(logMp)*(0.91546**logMp))**logF + 1.2503',
    ('s123', 11): '(1.0027**logF)**logF * (0.85361**logMp)*logMp',
    ('s123', 12): '((0.78638**logMp + 0.35916)*log(logMp))**logF + 1.1001',
    ('s123', 14): '((0.78925**logMp + 0.35741)*log(logMp))**logF + 1.0347**logP',
    ('s123', 16): '1.04**logP + (((log(logMp)*(0.77914**logMp + 0.36579))**logF)**1.2959)',
    ('s123', 19): '((log(logMp - log1p_ecc/log(logMp))*(0.35931 + 0.7886**logMp))**logF) + 1.0452**logP',
    ('s123', 20): '((log(log1p_ecc/(1.8735 - logF) + logMp)*(0.35916 + 0.78827**logMp))**logF) + 1.0448**logP',
}

# ============================================================
# Baseline e paper originale
# ============================================================
lr = LinearRegression().fit(df_train['logMp'].values.reshape(-1,1), y_tr)
r2_base  = safe_r2(y_te, lr.predict(df_test['logMp'].values.reshape(-1,1)))
r2_paper = safe_r2(y_te, eval_f('logMp*(logF*0.00308879 + 0.85000783)**logMp', vals_te))

# ============================================================
# Valuta tutte le formule
# ============================================================
results = []
for (seed, comp), expr in formulas.items():
    yp = eval_f(expr, vals_te)
    r2 = safe_r2(y_te, yp) if yp is not None else float('nan')
    # Calcola anche R² sul train per confronto (overfitting check)
    yp_tr = eval_f(expr, vals_tr)
    r2_tr = safe_r2(y_tr, yp_tr) if yp_tr is not None else float('nan')
    results.append({
        'seed': seed, 'complexity': comp,
        'r2_test_oos': r2,    # out-of-sample reale
        'r2_train':    r2_tr,
        'overfit':     r2_tr - r2 if not (np.isnan(r2) or np.isnan(r2_tr)) else float('nan'),
        'formula': expr[:80]
    })

df_res = pd.DataFrame(results).sort_values('r2_test_oos', ascending=False)

# ============================================================
# Stampa risultati
# ============================================================
print(f"\n{'='*90}")
print("TUTTE LE FORMULE — R² OUT-OF-SAMPLE (post-2022, mai visti da PySR)")
print(f"  Baseline power-law:        R²={r2_base:.4f}")
print(f"  Chen & Kipping 2017:       R²=0.441  (reference)")
print(f"  Paper originale (seed0):   R²={r2_paper:.4f}  (NON out-of-sample)")
print(f"{'='*90}")
print(f"{'Seed':>6} {'Comp':>5} {'R²_oos':>8} {'R²_train':>9} {'Overfit':>8}  Formula")
print("-"*90)

for _, row in df_res.iterrows():
    if np.isnan(row['r2_test_oos']):
        continue
    marker = ""
    if row['r2_test_oos'] > 0.66:
        marker = " ◄ MIGLIORE PAPER OOS"
    elif row['r2_test_oos'] > r2_base + 0.15:
        marker = " ◄ BATTE BASELINE"
    print(f"{row['seed']:>6} {int(row['complexity']):>5} "
          f"{row['r2_test_oos']:>8.4f} {row['r2_train']:>9.4f} "
          f"{row['overfit']:>8.4f}  {row['formula'][:55]}{marker}")

best = df_res.iloc[0]
print(f"\n{'='*90}")
print(f"MIGLIOR FORMULA OUT-OF-SAMPLE:")
print(f"  Seed={best['seed']}  Complessità={int(best['complexity'])}")
print(f"  R²_oos={best['r2_test_oos']:.4f}  R²_train={best['r2_train']:.4f}  Overfit={best['overfit']:.4f}")
print(f"  {formulas[(best['seed'], int(best['complexity']))]}")

# Formula più stabile (comp9, presente in seed 0 e 42 identica)
print(f"\nFORMULA STABILE (comp9, seed 0 e 42 identica):")
r2_stab = safe_r2(y_te, eval_f('(logF - (logMp + (-12.642))*logMp)*0.056739', vals_te))
print(f"  R²_oos={r2_stab:.4f}")
print(f"  logRp = 0.05674·logF + 0.717·logMp - 0.05674·logMp²")
print(f"  Espansa: logRp = α·logF + β·logMp - α·logMp²  (α=0.05674, β=0.717)")

df_res.to_csv('tutte_formule_pre2022_r2.csv', index=False)
print(f"\nSalvato: tutte_formule_pre2022_r2.csv")
