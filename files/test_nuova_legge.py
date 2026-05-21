"""
test_nuova_legge.py — Testa se i termini fisici trovati nell'analisi
dei residui migliorano il modello gas senza riaddestrare PySR.

Strategia: modello ibrido = formula PySR attuale + correzione lineare
sui residui con i termini significativi (log_rhostar, FeH, logF², logF*FeH).
Se R² migliora significativamente → vale la pena fare il re-run PySR.

Uso: python3 ~/test_nuova_legge.py
"""

import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score

CHECKPOINTS_DIR = 'checkpoints_pysr_paper'
DATA_CSV        = 'exoplanets_filtrati.csv'
GROUP           = 'gas'
SEED_BEST       = 0
OUT_DIR         = 'residui_analysis'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. Carica modello e dati
# ============================================================
pkl = os.path.join(CHECKPOINTS_DIR, f'{GROUP}_seed{SEED_BEST}', 'model.pkl')
with open(pkl, 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv(DATA_CSV)
df['logMp']       = np.log(df['pl_masse'])
df['logF']        = np.log(df['pl_insol'])
df['FeH']         = df['st_met']
df['logMs']       = np.log(df['st_mass'])
df['logRs']       = np.log(df['st_rad'])
df['logP']        = np.log(df['pl_orbper'])
df['logRp']       = np.log(df['pl_rade'])
df['log_rhostar'] = df['logMs'] - 3*df['logRs']
if 'pl_orbeccen' in df.columns:
    df['ecc'] = df['pl_orbeccen'].fillna(0.0).clip(0, 0.99)
else:
    df['ecc'] = 0.0

bins   = [0, 1.5, 2.0, 4.0, 100]
labels = ['sub_terr', 'rocky_SE', 'sub_nep', 'gas']
df['group'] = pd.cut(df['pl_rade'], bins=bins, labels=labels)
df_hot = df[df['group'] == 'gas'].dropna(
    subset=['logMp','logF','FeH','logP','log_rhostar','logRp']
).copy()

X_hot = np.column_stack([
    df_hot['logMp'].values,
    df_hot['logF'].values,
    df_hot['FeH'].values,
    df_hot['logP'].values,
    df_hot['log_rhostar'].values,
])

y     = df_hot['logRp'].values
y_pred_pysr = model.predict(X_hot)
residui     = y - y_pred_pysr

r2_pysr = float(1 - np.sum(residui**2) / np.sum((y - y.mean())**2))
print(f"R² modello PySR originale: {r2_pysr:.4f}")

# ============================================================
# 2. Costruisci i termini fisici dai residui
# ============================================================
logF        = df_hot['logF'].values
FeH         = df_hot['FeH'].values
logMp       = df_hot['logMp'].values
log_rhostar = df_hot['log_rhostar'].values

# Termini significativi dall'analisi residui
terms = {
    'log_rhostar': log_rhostar,
    'FeH':         FeH,
    'logF^2':      logF**2,
    'logF*FeH':    logF * FeH,
    'logMp*logF':  logMp * logF,
}

# ============================================================
# 3. Modello ibrido: PySR + correzione lineare sui residui
# ============================================================
print("\n" + "="*60)
print("MODELLO IBRIDO: PySR + correzione lineare dei residui")
print("="*60)

# Matrice con tutti i termini
X_corr = np.column_stack(list(terms.values()))

# Ridge regression sui residui (evita overfitting)
ridge = Ridge(alpha=0.01)
ridge.fit(X_corr, residui)
correzione = ridge.predict(X_corr)

y_pred_hybrid = y_pred_pysr + correzione
r2_hybrid = float(1 - np.sum((y - y_pred_hybrid)**2) /
                  np.sum((y - y.mean())**2))
print(f"R² modello ibrido (PySR + correzione): {r2_hybrid:.4f}")
print(f"Miglioramento ΔR²: {r2_hybrid - r2_pysr:+.4f}")

print("\nCoefficienti della correzione:")
for name, coef in zip(terms.keys(), ridge.coef_):
    print(f"  {name:<15}: {coef:+.4f}")

# Cross-validation 5-fold per verificare che non sia overfitting
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=0.01))])
cv_r2 = cross_val_score(pipe, X_corr, residui, cv=5, scoring='r2')
print(f"\nCross-validation R² sui residui: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print("(Se questo è positivo, la correzione è reale, non overfitting)")

# ============================================================
# 4. Test su subset: split 80/20 per simulare train/test reale
# ============================================================
print("\n" + "="*60)
print("SIMULAZIONE TRAIN/TEST (80/20, random_state=1)")
print("="*60)

from sklearn.model_selection import train_test_split
idx = np.arange(len(y))
idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=1)

# PySR sul test set
r2_pysr_te = float(1 - np.sum((y[idx_te] - y_pred_pysr[idx_te])**2) /
                   np.sum((y[idx_te] - y[idx_te].mean())**2))

# Ibrido: addestra correzione solo su train, valuta su test
ridge_te = Ridge(alpha=0.01)
ridge_te.fit(X_corr[idx_tr], residui[idx_tr])
corr_te  = ridge_te.predict(X_corr[idx_te])
y_hyb_te = y_pred_pysr[idx_te] + corr_te
r2_hyb_te = float(1 - np.sum((y[idx_te] - y_hyb_te)**2) /
                  np.sum((y[idx_te] - y[idx_te].mean())**2))

print(f"R² PySR    su test set: {r2_pysr_te:.4f}")
print(f"R² ibrido  su test set: {r2_hyb_te:.4f}")
print(f"ΔR² test:               {r2_hyb_te - r2_pysr_te:+.4f}")

if r2_hyb_te - r2_pysr_te > 0.02:
    print("\n>>> MIGLIORAMENTO REALE sul test set: vale la pena fare re-run PySR <<<")
elif r2_hyb_te - r2_pysr_te > 0:
    print("\n>>> Miglioramento modesto: i termini aggiuntivi aiutano ma poco <<<")
else:
    print("\n>>> Nessun miglioramento sul test set: i pattern sono noise sui residui <<<")

# ============================================================
# 5. Figura: confronto predizioni PySR vs ibrido
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Confronto modello PySR vs ibrido (hot Jupiters)', fontsize=13)

Rp_obs  = np.exp(y)
Rp_pysr = np.exp(np.clip(y_pred_pysr, -5, 5))
Rp_hyb  = np.exp(np.clip(y_pred_hybrid, -5, 5))

# Plot 1: PySR
ax = axes[0]
ax.scatter(Rp_pysr, Rp_obs, s=12, alpha=0.5, color='steelblue')
lim = [min(Rp_obs.min(), Rp_pysr.min())*0.9,
       max(Rp_obs.max(), Rp_pysr.max())*1.1]
ax.plot(lim, lim, 'k--', lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('$R_p$ predetto'); ax.set_ylabel('$R_p$ osservato')
ax.set_title(f'PySR originale\nR²={r2_pysr:.3f}')

# Plot 2: ibrido
ax = axes[1]
ax.scatter(Rp_hyb, Rp_obs, s=12, alpha=0.5, color='tomato')
ax.plot(lim, lim, 'k--', lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('$R_p$ predetto'); ax.set_ylabel('$R_p$ osservato')
ax.set_title(f'Ibrido (PySR + correzione)\nR²={r2_hybrid:.3f}')

# Plot 3: residui PySR vs residui ibrido
ax = axes[2]
res_hyb = y - y_pred_hybrid
ax.scatter(residui, res_hyb, s=12, alpha=0.5, color='purple')
ax.plot([-1,1], [-1,1], 'k--', lw=1, label='nessun cambio')
ax.axhline(0, color='grey', lw=0.5)
ax.axvline(0, color='grey', lw=0.5)
ax.set_xlabel('Residui PySR')
ax.set_ylabel('Residui ibrido')
ax.set_title('Riduzione residui\n(punti sotto la diagonale = migliorati)')
ax.legend(fontsize=8)

plt.tight_layout()
out_fig = os.path.join(OUT_DIR, 'confronto_pysr_vs_ibrido.png')
plt.savefig(out_fig, dpi=150)
plt.close(fig)
print(f"\nGrafico salvato: {out_fig}")

# ============================================================
# 6. Raccomandazione finale
# ============================================================
print("\n" + "="*60)
print("RACCOMANDAZIONE")
print("="*60)
delta = r2_hyb_te - r2_pysr_te
if delta > 0.05:
    print(f"ΔR² test = {delta:+.4f} → FORTE segnale.")
    print("Consiglio: fai re-run PySR con log_rhostar, FeH, logF² espliciti.")
    print("Aspettati R² ~ {:.2f} sul test set.".format(r2_pysr_te + delta))
elif delta > 0.01:
    print(f"ΔR² test = {delta:+.4f} → segnale MODESTO ma reale.")
    print("Consiglio: aggiungi log_rhostar come feature nel prossimo run.")
    print("I termini logF² e logF*FeH sono meno prioritari.")
else:
    print(f"ΔR² test = {delta:+.4f} → segnale DEBOLE.")
    print("I pattern nei residui sono reali ma il modello attuale è già vicino al massimo")
    print("estraibile da questi dati. Non vale la pena fare un re-run costoso.")
print("="*60)
