"""
analisi_residui.py — Analisi dei residui del modello gas (hot Jupiters).

Carica il modello PySR già addestrato, calcola i residui su tutto il
gruppo gas, e testa correlazioni lineari e quadratiche con ogni variabile
di input. Se trova pattern significativi, li stampa e salva i grafici.

Uso: python3 ~/analisi_residui.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression as _fr

# ============================================================
# CONFIG — modifica solo questi percorsi se necessario
# ============================================================
CHECKPOINTS_DIR = 'checkpoints_pysr'
DATA_CSV        = 'exoplanets_filtrati.csv'   # dataset originale del paper
GROUP           = 'gas'
SEED_BEST       = 0                            # seed migliore per gas
OUT_DIR         = 'residui_analysis'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. Carica modello
# ============================================================
pkl_path = os.path.join(CHECKPOINTS_DIR, f'{GROUP}_seed{SEED_BEST}', 'model.pkl')
if not os.path.exists(pkl_path):
    raise FileNotFoundError(f"Modello non trovato: {pkl_path}\n"
                            f"Assicurati di essere nella home Ubuntu (~/) e "
                            f"che il run sia completato.")

with open(pkl_path, 'rb') as f:
    model = pickle.load(f)
print(f"Modello caricato: {pkl_path}")
try:
    print(f"Formula: {model.sympy()}")
except:
    print("Formula: (non disponibile)")

# ============================================================
# 2. Carica dataset e prepara features
# ============================================================
if not os.path.exists(DATA_CSV):
    raise FileNotFoundError(f"Dataset non trovato: {DATA_CSV}")

df = pd.read_csv(DATA_CSV)

# Feature engineering identico a prep() in run_sr.py
df['logMp']       = np.log(df['pl_masse'])
df['logF']        = np.log(df['pl_insol'])
df['FeH']         = df['st_met']
df['logMs']       = np.log(df['st_mass'])
df['logRs']       = np.log(df['st_rad'])
df['logP']        = np.log(df['pl_orbper'])
df['logRp']       = np.log(df['pl_rade'])
df['log_rhostar'] = df['logMs'] - 3*df['logRs']

# Eccentricità: usa log1p se presente, altrimenti 0
if 'pl_orbeccen' in df.columns:
    df['ecc'] = df['pl_orbeccen'].fillna(0.0).clip(0, 0.99)
else:
    df['ecc'] = 0.0

# Filtra solo gas giants
bins   = [0, 1.5, 2.0, 4.0, 100]
labels = ['sub_terr', 'rocky_SE', 'sub_nep', 'gas']
df['group'] = pd.cut(df['pl_rade'], bins=bins, labels=labels)
df_hot = df[df['group'] == 'gas'].dropna(
    subset=['logMp','logF','FeH','logP','log_rhostar','logRp']
).copy()
print(f"\nHot Jupiters nel dataset: {len(df_hot)}")

# Matrice X nello stesso ordine di VAR_NAMES in run_sr.py
X_hot = np.column_stack([
    df_hot['logMp'].values,
    df_hot['logF'].values,
    df_hot['FeH'].values,
    df_hot['logP'].values,
    df_hot['log_rhostar'].values,
    np.log1p(df_hot['ecc'].values),
])

# ============================================================
# 3. Calcola residui
# ============================================================
y_pred = model.predict(X_hot)
df_hot['lnRp_pred'] = y_pred
df_hot['residuo']   = df_hot['logRp'] - df_hot['lnRp_pred']

r2_obs = float(1 - np.sum(df_hot['residuo']**2) /
               np.sum((df_hot['logRp'] - df_hot['logRp'].mean())**2))
mae    = float(np.mean(np.abs(df_hot['residuo'])))
print(f"R² modello gas: {r2_obs:.4f}")
print(f"MAE residui:    {mae:.4f} (in log-spazio)")

# ============================================================
# 4. Analisi correlazioni residui vs ogni variabile
# ============================================================
vars_to_test = ['logMp', 'logF', 'FeH', 'logP', 'log_rhostar', 'ecc']
# Aggiungi interazioni a due termini
interactions = {
    'logF*FeH':       df_hot['logF'] * df_hot['FeH'],
    'logMp*logF':     df_hot['logMp'] * df_hot['logF'],
    'logF^2':         df_hot['logF']**2,
    'logMp^2':        df_hot['logMp']**2,
    'FeH^2':          df_hot['FeH']**2,
}
for k, v in interactions.items():
    df_hot[k] = v.values
all_vars = vars_to_test + list(interactions.keys())

results = {}
y = df_hot['residuo'].values

for var in all_vars:
    x = df_hot[var].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        continue
    xm, ym = x[mask], y[mask]

    # Correlazione lineare
    r_lin, p_lin = stats.pearsonr(xm, ym)
    r_spe, p_spe = stats.spearmanr(xm, ym)

    # Modello quadratico: y ~ a*x + b*x^2
    X_quad = np.column_stack([xm, xm**2])
    lr = LinearRegression().fit(X_quad, ym)
    r2_quad = lr.score(X_quad, ym)
    b_quad  = lr.coef_[1]
    # p-value modello quadratico via F-test
    f_stat, p_quad = _fr(X_quad, ym)
    p_quad = float(p_quad[0])  # p del modello completo

    results[var] = {
        'r_lin':   float(r_lin),
        'p_lin':   float(p_lin),
        'r_spe':   float(r_spe),
        'p_spe':   float(p_spe),
        'r2_quad': float(r2_quad),
        'b_quad':  float(b_quad),
        'p_quad':  float(p_quad),
        'n':       int(mask.sum()),
    }

# Correzione Bonferroni
alpha_bonf = 0.01 / len(results)
print(f"\n{'='*60}")
print(f"CORRELAZIONI RESIDUI (soglia Bonferroni p < {alpha_bonf:.4f})")
print(f"{'='*60}")
print(f"{'Variabile':<18} {'r_lin':>7} {'p_lin':>10} {'r²_quad':>8} {'b_quad':>9} {'p_quad':>10}")
print("-" * 65)

sig_lin  = []
sig_quad = []
for var, res in sorted(results.items(), key=lambda x: x[1]['p_lin']):
    sig_l = res['p_lin']  < alpha_bonf
    sig_q = res['p_quad'] < alpha_bonf
    flag  = " *** " if sig_l or sig_q else ""
    print(f"{var:<18} {res['r_lin']:>7.3f} {res['p_lin']:>10.3e} "
          f"{res['r2_quad']:>8.3f} {res['b_quad']:>9.4f} "
          f"{res['p_quad']:>10.3e}{flag}")
    if sig_l:  sig_lin.append(var)
    if sig_q:  sig_quad.append(var)

print(f"\nVariabili con correlazione LINEARE significativa:   {sig_lin if sig_lin else 'nessuna'}")
print(f"Variabili con effetto QUADRATICO significativo:     {sig_quad if sig_quad else 'nessuna'}")

# ============================================================
# 5. Grafici residui vs variabili più importanti
# ============================================================
# Ordina per |r_lin| e prendi le top 4
top_vars = sorted(results.items(), key=lambda x: abs(x[1]['r_lin']), reverse=True)[:4]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Residui modello gas vs variabili input', fontsize=13)

for ax, (var, res) in zip(axes.flat, top_vars):
    x = df_hot[var].values
    y = df_hot['residuo'].values
    mask = np.isfinite(x) & np.isfinite(y)

    # Colora per FeH per vedere interazioni
    feh = df_hot['FeH'].values
    scatter = ax.scatter(x[mask], y[mask], c=feh[mask], cmap='RdYlBu_r',
                         s=15, alpha=0.6, vmin=-0.5, vmax=0.5)
    plt.colorbar(scatter, ax=ax, label='[Fe/H]')

    # Fit lineare
    xf  = np.linspace(x[mask].min(), x[mask].max(), 100)
    lr1 = LinearRegression().fit(x[mask].reshape(-1,1), y[mask])
    ax.plot(xf, lr1.predict(xf.reshape(-1,1)), 'k-', lw=1.5,
            label=f'lineare (r={res["r_lin"]:.2f})')

    # Fit quadratico
    lr2 = LinearRegression().fit(
        np.column_stack([x[mask], x[mask]**2]), y[mask])
    ax.plot(xf, lr2.predict(np.column_stack([xf, xf**2])), 'r--', lw=1.5,
            label=f'quadratico (R²={res["r2_quad"]:.2f})')

    ax.axhline(0, color='grey', lw=0.8, ls=':')
    ax.set_xlabel(var, fontsize=10)
    ax.set_ylabel('Residuo (logRp_obs - logRp_pred)', fontsize=9)
    ax.set_title(f'{var}  |  r={res["r_lin"]:.3f}  p={res["p_lin"]:.2e}',
                 fontsize=10)
    ax.legend(fontsize=8)

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'residui_vs_variabili.png')
plt.savefig(fig_path, dpi=150)
plt.close(fig)
print(f"\nGrafico salvato: {fig_path}")

# ============================================================
# 6. Grafico speciale: residui vs logF colorato per FeH bins
# ============================================================
fig2, ax2 = plt.subplots(figsize=(8, 5))
feh_bins  = pd.cut(df_hot['FeH'], bins=[-2, -0.2, 0.1, 0.5],
                   labels=['bassa [Fe/H]<-0.2', 'media -0.2 to 0.1', 'alta [Fe/H]>0.1'])
colors = {'bassa [Fe/H]<-0.2': 'steelblue',
          'media -0.2 to 0.1': 'orange',
          'alta [Fe/H]>0.1':   'tomato'}

for label, grp in df_hot.groupby(feh_bins, observed=True):
    ax2.scatter(grp['logF'], grp['residuo'], s=18, alpha=0.6,
                color=colors.get(str(label), 'grey'), label=str(label))
    # Fit lineare per gruppo
    xg = grp['logF'].values
    yg = grp['residuo'].values
    mask = np.isfinite(xg) & np.isfinite(yg)
    if mask.sum() > 10:
        lr = LinearRegression().fit(xg[mask].reshape(-1,1), yg[mask])
        xf = np.linspace(xg[mask].min(), xg[mask].max(), 50)
        ax2.plot(xf, lr.predict(xf.reshape(-1,1)),
                 color=colors.get(str(label), 'grey'), lw=2)

ax2.axhline(0, color='k', lw=0.8, ls='--')
ax2.set_xlabel('log F (irraggiamento stellare)', fontsize=11)
ax2.set_ylabel('Residuo (logRp_obs - logRp_pred)', fontsize=11)
ax2.set_title('Residui vs logF per bin di metallicità\n'
              '(linee = fit lineari per gruppo)', fontsize=11)
ax2.legend(fontsize=9)
plt.tight_layout()
fig2_path = os.path.join(OUT_DIR, 'residui_logF_per_FeH.png')
plt.savefig(fig2_path, dpi=150)
plt.close(fig2)
print(f"Grafico salvato: {fig2_path}")

# ============================================================
# 7. Salva tabella risultati
# ============================================================
df_res_out = pd.DataFrame(results).T.reset_index().rename(columns={'index':'variabile'})
df_res_out = df_res_out.sort_values('p_lin')
df_res_out.to_csv(os.path.join(OUT_DIR, 'correlazioni_residui.csv'), index=False)
df_hot[['logMp','logF','FeH','logP','log_rhostar','ecc',
        'logRp','lnRp_pred','residuo']].to_csv(
    os.path.join(OUT_DIR, 'residui_gas_completo.csv'), index=False)

print(f"\nOutput salvati in: {OUT_DIR}/")
print("=== ANALISI COMPLETATA ===")
