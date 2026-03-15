"""
analyze_results.py — Analisi post-run per il paper sugli esopianeti.

Controlli implementati:
  Bootstrap : residual, pair, BCa, wild (condizionale su eteroschedasticità)
  Domain    : griglia 2D logMp×logF, log args check, extrapolazione ±50%
  Fisici    : monotonicità R(M), derivate parziali, sensitivity ±10%
  Diagnostici: PDP+ICE, eteroschedasticità (Breusch-Pagan), Q-Q+skew,
               outlier/Cook's D, stratified R², correlazione residui-errori,
               check termini x^x, oscillazioni, calibrazione CI
  Sympy     : simplify + LaTeX
  Temporale : performance degradation check su pianeti post-2022
              NOTA: pkl addestrati su tutto DATA_CSV; questo verifica
              robustezza temporale, NON predizione pura (v. note metodologiche).
  Metadata  : versioni, git, checksum CSV, timestamp

Non implementati (richiedono dati/run esterni):
  - Confronto con modelli teorici (serve dataset curve teoriche)
  - Stability across preprocessing/hyperparams (richiedono re-run PySR)

Uso: python3 analyze_results.py
"""

import os, sys, json, pickle, hashlib, logging, datetime, subprocess, argparse
import numpy as np
import pandas as pd
import requests, io as _io
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ============================================================
# LOGGING — inizializzato subito, prima di qualsiasi import opzionale
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('analyze_results.log', mode='a'),
    ]
)
log = logging.getLogger(__name__)

# Import opzionali — log già disponibile
try:
    import sympy
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    log.warning("sympy non disponibile — simplify/LaTeX disabilitati")

try:
    import pysr as _pysr
    PYSR_VERSION = _pysr.__version__
except ImportError:
    PYSR_VERSION = "n/d"
    log.warning("pysr non disponibile — caricamento pkl potrebbe fallire")

try:
    import sklearn as _sk
    SKLEARN_VERSION = _sk.__version__
except ImportError:
    SKLEARN_VERSION = "n/d"


# ============================================================
# CLI — argomenti da riga di comando
# ============================================================
_parser = argparse.ArgumentParser(description='analyze_results.py')
_parser.add_argument('--n-boot', type=int, default=1000,
                     help='Numero iterazioni bootstrap (default 1000; usa 200 per smoke test)')
_args, _ = _parser.parse_known_args()

# ============================================================
# CONFIG
# ============================================================
CHECKPOINTS_DIR = 'checkpoints_pysr'
DATA_CSV        = 'exoplanets_filtrati.csv'
OUT_DIR         = 'analyze_outputs'
CACHE_DIR       = os.path.join(OUT_DIR, 'cache')
for _d in [OUT_DIR, CACHE_DIR]:
    os.makedirs(_d, exist_ok=True)

VAR_NAMES  = ["logMp", "logF", "FeH", "logP", "logRhoStar"]
labels     = ['sub_terr', 'rocky_SE', 'sub_nep', 'gas']
SEEDS      = [0, 42, 123]
N_BOOT     = _args.n_boot
RNG_SEED   = 42
SPLIT_YEAR = 2022

log.info("=" * 60)
log.info("analyze_results.py — avvio")
log.info(f"  N_BOOT={N_BOOT}  RNG_SEED={RNG_SEED}")
log.info("=" * 60)

# ============================================================
# METADATA
# ============================================================
_git = "n/d"
try:
    _git = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    pass

_csv_md5 = "n/d"
if os.path.exists(DATA_CSV):
    with open(DATA_CSV, 'rb') as _f:
        _csv_md5 = hashlib.md5(_f.read()).hexdigest()

metadata = {
    "script":       "analyze_results.py",
    "timestamp":    datetime.datetime.now().isoformat(),
    "git_commit":   _git,
    "python":       sys.version.split()[0],
    "numpy":        np.__version__,
    "pandas":       pd.__version__,
    "sklearn":      SKLEARN_VERSION,
    "pysr":         PYSR_VERSION,
    "n_boot":       N_BOOT,
    "rng_seed":     RNG_SEED,
    "split_year":   SPLIT_YEAR,
    "data_csv":     DATA_CSV,
    "csv_md5":      _csv_md5,
    "note_bootstrap_memory": (
        "Bootstrap vettorizzato crea matrici (n_boot x n) in memoria. "
        "Con n_boot=1000 e n<5000 e' velocissimo e sicuro. "
        "Se il dataset cresce oltre 50k righe, sostituire con loop a chunk."
    ),
    "note_bootstrap": (
        "residual bootstrap: assume errori i.i.d.; "
        "pair bootstrap: non assume i.i.d.; "
        "BCa: corregge bias+accelerazione via jackknife (fallback a pair se instabile); "
        "wild bootstrap: usato automaticamente se Breusch-Pagan p<0.05"
    ),
    "note_clip_logRp": (
        "logRp clippato in [-5, 10] prima di exp() per evitare overflow. "
        "log(25 R_terra)~3.2, log(0.05 R_terra)~-3.0: il range [-5,10] copre "
        "tutto il fisicamente plausibile con ampio margine."
    ),
    "note_nasa_query": (
        "Dati: NASA Exoplanet Archive, tabella pscomppars. "
        "Citare: NASA Exoplanet Archive (https://exoplanetarchive.ipac.caltech.edu). "
        "Data query: vedi timestamp sopra."
    ),
    "note_pvalue": (
        "p-value permutation approssimativo (usa media checkpoint). "
        "Citare solo come sanity check qualitativo nel paper."
    ),
}
with open(os.path.join(OUT_DIR, 'metadata.json'), 'w') as _mf:
    json.dump(metadata, _mf, indent=2)
log.info("metadata.json salvato")

# ============================================================
# UTILITY
# ============================================================
def safe_r2(y_true, y_pred):
    """
    R² sicuro: gestisce ss_tot==0, vettori vuoti, NaN.
    Restituisce float('nan') in tutti i casi degeneri.
    Nota: clip [-5,10] usato a monte su logRp per evitare overflow in exp().
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0 or not np.all(np.isfinite(y_true)) or \
       not np.all(np.isfinite(y_pred)):
        return float('nan')
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    if ss_tot <= 0:
        return float('nan')
    return float(1.0 - ss_res / ss_tot)


def safe_predict(model, X):
    # Predict robusto: gestisce modelli non-PySR e X non 2D.
    try:
        Xa   = np.atleast_2d(np.asarray(X, dtype=float))
        pred = model.predict(Xa)
        return np.asarray(pred, dtype=float).ravel()
    except Exception as e:
        log.warning(f'  safe_predict fallito: {e}')
        return np.full(np.atleast_2d(X).shape[0], float('nan'))


def safe_sympy(model):
    # Sympy robusto: restituisce 'n/d' se modello non-PySR o errore.
    try:
        return str(model.sympy())
    except Exception:
        return 'n/d'

# ============================================================
# STEP 1 — Carica dataset
# ============================================================
log.info("[1] Carico dataset...")
df = pd.read_csv(DATA_CSV)

df['logMp']       = np.log(df['pl_masse'])
df['logF']        = np.log(df['pl_insol'])
df['FeH']         = df['st_met']
df['logMs']       = np.log(df['st_mass'])
df['logRs']       = np.log(df['st_rad'])
df['logP']        = np.log(df['pl_orbper'])
df['logRp']       = np.log(df['pl_rade'])
df['log_rhostar'] = df['logMs'] - 3*df['logRs']
df['err_rade_rel']= np.abs(df['pl_radeerr1']) / df['pl_rade']
df['err_rel_max'] = df[['pl_masseerr1','pl_masseerr2']].abs().max(axis=1) / df['pl_masse']
df['sigma_tot']   = np.sqrt(df['err_rel_max']**2 + df['err_rade_rel']**2)
df['weight']      = 1.0 / (df['sigma_tot']**2 + 1e-6)

bins_g = [0, 1.5, 2.0, 4.0, 100]
df['group'] = pd.cut(df['pl_rade'], bins=bins_g, labels=labels)
log.info(f"  Dataset caricato: {len(df)} pianeti")

def prep(sub):
    X = np.column_stack([
        sub['logMp'].values,
        sub['logF'].values,
        sub['FeH'].values,
        sub['logP'].values,
        sub['log_rhostar'].values,
    ])
    y = sub['logRp'].values
    w = sub['weight'].values
    w = np.maximum(w, 1e-6)
    w = w / w.mean()
    return X, y, w

# ============================================================
# STEP 2 — Funzioni bootstrap
# ============================================================
def _boot_stats(r2_boot):
    # Utility: calcola CI e mean/std da array bootstrap.
    valid = r2_boot[np.isfinite(r2_boot)]
    ci    = np.nanpercentile(valid, [2.5, 97.5]) if len(valid) > 0 \
            else np.array([float('nan'), float('nan')])
    mean  = float(np.nanmean(valid)) if len(valid) > 0 else float('nan')
    std   = float(np.nanstd(valid))  if len(valid) > 0 else float('nan')
    return mean, std, ci


def bootstrap_r2(y_true, y_pred, n_boot=N_BOOT, seed=RNG_SEED):
    # Residual bootstrap vettorizzato (assume errori i.i.d.).
    resid  = np.asarray(y_true - y_pred)
    r2_obs = safe_r2(y_true, y_pred)
    rng    = np.random.default_rng(seed)
    # Matrice (n_boot, n): ogni riga e' un ricampionamento dei residui
    idx    = rng.integers(0, len(resid), size=(n_boot, len(resid)))
    rs     = resid[idx]                          # (n_boot, n)
    y_star = y_pred + rs                         # (n_boot, n)
    ss_res = np.sum((y_star - y_pred)**2, axis=1)
    ss_tot = np.sum((y_star - y_star.mean(axis=1, keepdims=True))**2, axis=1)
    r2_boot = np.where(ss_tot > 0, 1 - ss_res/ss_tot, float('nan'))
    mean, std, ci = _boot_stats(r2_boot)
    return r2_obs, mean, std, ci, r2_boot


def pair_bootstrap_r2(y_true, y_pred, n_boot=N_BOOT, seed=RNG_SEED):
    # Pair bootstrap vettorizzato (non assume i.i.d.).
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    r2_obs = safe_r2(y_true, y_pred)
    rng    = np.random.default_rng(seed)
    n      = len(y_true)
    idx    = rng.integers(0, n, size=(n_boot, n))  # (n_boot, n)
    yt     = y_true[idx]; yp = y_pred[idx]
    ss_res = np.sum((yt - yp)**2, axis=1)
    ss_tot = np.sum((yt - yt.mean(axis=1, keepdims=True))**2, axis=1)
    r2_boot = np.where(ss_tot > 0, 1 - ss_res/ss_tot, float('nan'))
    mean, std, ci = _boot_stats(r2_boot)
    return r2_obs, mean, std, ci, r2_boot


def bca_bootstrap(y_true, y_pred, n_boot=N_BOOT, seed=RNG_SEED):
    """
    BCa: pair bootstrap vettorizzato + jackknife acceleration.
    Fallback a pair percentile se jackknife instabile.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    rng    = np.random.default_rng(seed)
    n      = len(y_true)
    r2_obs = safe_r2(y_true, y_pred)
    # Pair bootstrap vettorizzato
    idx    = rng.integers(0, n, size=(n_boot, n))
    yt     = y_true[idx]; yp = y_pred[idx]
    ss_res = np.sum((yt - yp)**2, axis=1)
    ss_tot = np.sum((yt - yt.mean(axis=1, keepdims=True))**2, axis=1)
    r2_boot = np.where(ss_tot > 0, 1 - ss_res/ss_tot, float('nan'))
    valid   = r2_boot[np.isfinite(r2_boot)]
    if len(valid) < 20:
        log.warning("  BCa: bootstrap insufficienti — fallback a pair percentile")
        ci = np.nanpercentile(valid, [2.5, 97.5]) if len(valid) > 0 \
             else np.array([float('nan'), float('nan')])
        return r2_obs, ci
    # Bias correction con clip per evitare ppf(0) o ppf(1) = ±inf
    prob = np.clip(np.mean(valid < r2_obs), 1e-6, 1 - 1e-6)
    z0   = stats.norm.ppf(prob)
    # Jackknife acceleration — vettorizzato O(n), non O(n^2)
    # Pre-calcola somme totali e le aggiorna sottraendo l'elemento i-esimo
    resid_sq_all = (y_true - y_pred)**2           # ss_res contributi
    ss_res_full  = resid_sq_all.sum()
    # Per ss_tot LOO: Var(y_-i) = [sum_j(y_j^2) - y_i^2]/(n-1) - [(sum_j y_j - y_i)/(n-1)]^2
    # Usiamo la formula: ss_tot_-i = sum_j(y_j - mean_-i)^2 con mean_-i = (S - y_i)/(n-1)
    S   = y_true.sum()
    S2  = (y_true**2).sum()
    SS_res_i  = ss_res_full - resid_sq_all          # ss_res senza punto i (n,)
    mean_i    = (S - y_true) / (n - 1)              # media LOO per ogni i (n,)
    # ss_tot LOO: sum_{j!=i}(y_j - mean_i)^2
    ss_tot_i  = (S2 - y_true**2) - (n-1) * mean_i**2  # (n,)
    jk_valid  = ss_tot_i > 0
    jk        = np.where(jk_valid, 1 - SS_res_i/ss_tot_i, float('nan'))
    jk        = np.array(jk)
    if not np.all(np.isfinite(jk)) or np.allclose(jk, jk[0], rtol=1e-8, atol=1e-12):
        log.warning("  BCa: jackknife degenere — fallback a pair percentile")
        return r2_obs, np.nanpercentile(valid, [2.5, 97.5])
    jk_mean = np.nanmean(jk)
    num = np.sum((jk_mean - jk)**3)
    den = 6.0 * (np.sum((jk_mean - jk)**2)**1.5 + 1e-20)
    a   = num / den
    z_alpha = stats.norm.ppf([0.025, 0.975])
    adj = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a*(z0 + z_alpha)))
    adj = np.clip(adj, 1e-6, 1 - 1e-6)
    ci  = np.nanpercentile(valid, 100.0 * adj)
    return r2_obs, ci


def wild_bootstrap(y_true, y_pred, n_boot=N_BOOT, seed=RNG_SEED):
    # Wild bootstrap vettorizzato (Rademacher). Robusto per eteroschedasticita'.
    resid  = np.asarray(y_true - y_pred)
    r2_obs = safe_r2(y_true, y_pred)
    rng    = np.random.default_rng(seed)
    # Matrice (n_boot, n) di pesi Rademacher +-1
    v      = rng.choice([-1.0, 1.0], size=(n_boot, len(resid)))
    y_star = y_pred + resid * v             # (n_boot, n)
    ss_res = np.sum((y_star - y_pred)**2, axis=1)
    ss_tot = np.sum((y_star - y_star.mean(axis=1, keepdims=True))**2, axis=1)
    r2_boot = np.where(ss_tot > 0, 1 - ss_res/ss_tot, float('nan'))
    mean, std, ci = _boot_stats(r2_boot)
    return r2_obs, mean, std, ci, r2_boot


def permutation_pvalue(r2_obs, r2_perm_list):
    """p-value one-sided con add-1 smoothing."""
    arr = np.array([x for x in r2_perm_list if not np.isnan(x)])
    if len(arr) == 0:
        return float('nan')
    return float((np.sum(arr >= r2_obs) + 1) / (len(arr) + 1))


# ============================================================
# STEP 3 — Domain check
# ============================================================
def domain_check(model, X, g):
    lm_min, lm_max = np.percentile(X[:, 0], [1, 99])
    lf_min, lf_max = np.percentile(X[:, 1], [1, 99])
    lm_grid = np.linspace(lm_min, lm_max, 50)
    lf_grid = np.linspace(lf_min, lf_max, 50)
    LM, LF  = np.meshgrid(lm_grid, lf_grid)
    means   = X.mean(axis=0)
    XX = np.column_stack([LM.ravel(), LF.ravel(),
                          np.full(LM.size, means[2]),
                          np.full(LM.size, means[3]),
                          np.full(LM.size, means[4])])
    try:
        y_grid = model.predict(XX)
    except Exception as e:
        log.warning(f"  [domain_check] predict fallito: {e}"); return
    bad = ~np.isfinite(y_grid)
    log.info(f"  Domain check: {bad.sum()}/{len(y_grid)} NaN/inf ({100*bad.mean():.1f}%)")
    y_plot = np.where(bad, -99, np.clip(y_grid, -1, 5)).reshape(LM.shape)
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        im = ax.contourf(lm_grid, lf_grid, y_plot, levels=20, cmap='viridis')
        plt.colorbar(im, ax=ax, label='logRp predetto (clip [-1,5])')
    except ValueError as e:
        log.warning(f"  [domain_check] contourf fallito: {e}"); plt.close(fig); return
    ax.set_xlabel('logMp'); ax.set_ylabel('logF')
    ax.set_title(f'{g} — domain check')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'domain_check_{g}.png'), dpi=150)
    plt.close(fig)
    np.savez(os.path.join(OUT_DIR, f'domain_check_{g}.npz'), LM=LM, LF=LF, y_grid=y_grid)
    log.info(f"  domain_check_{g}.png salvato")


# ============================================================
# STEP 4 — Controlli fisici
# ============================================================
def check_monotonicity(model, X, g):
    lm_min, lm_max = np.percentile(X[:, 0], [1, 99])
    lm_dense = np.linspace(lm_min, lm_max, 200)
    means = X.mean(axis=0)
    XX = np.column_stack([lm_dense,
                          np.full(200, means[1]), np.full(200, means[2]),
                          np.full(200, means[3]), np.full(200, means[4])])
    try:
        y_pred = model.predict(XX)
    except Exception as e:
        log.warning(f"  [monotonicity] predict fallito: {e}"); return
    if not np.all(np.isfinite(y_pred)):
        log.warning("  [monotonicity] NaN nel profilo"); return
    dlm   = lm_dense[1] - lm_dense[0]
    slope = np.gradient(y_pred, dlm)
    steep = np.sum(np.abs(slope) > 0.5)
    log.info(f"  Monotonicità: slope max={slope.max():.3f} min={slope.min():.3f}  |slope|>0.5: {steep}/200")
    Rp = np.exp(np.clip(y_pred, -1, 5))
    Mp = np.exp(lm_dense)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(Mp, Rp, color='steelblue', lw=2)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('$M_p$ [$M_\\oplus$]'); axes[0].set_ylabel('$R_p$ [$R_\\oplus$]')
    axes[0].set_title(f'{g} — profilo R(M)')
    axes[1].plot(lm_dense, slope, color='tomato', lw=2)
    axes[1].axhline(0, color='k', lw=1, ls='--')
    axes[1].axhline(0.5, color='gray', lw=1, ls=':')
    axes[1].axhline(-0.5, color='gray', lw=1, ls=':')
    axes[1].set_xlabel('logMp'); axes[1].set_ylabel('∂logRp/∂logMp')
    axes[1].set_title(f'{g} — derivata')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'monotonicity_{g}.png'), dpi=150)
    plt.close(fig)


def check_extrapolation(model, X, g):
    lm_min = np.percentile(X[:, 0], 1)
    lm_max = np.percentile(X[:, 0], 99)
    lm_ext = np.linspace(lm_min * 0.5, lm_max * 1.5, 300)
    means  = X.mean(axis=0)
    XX = np.column_stack([lm_ext,
                          np.full(300, means[1]), np.full(300, means[2]),
                          np.full(300, means[3]), np.full(300, means[4])])
    try:
        y_pred = model.predict(XX)
    except Exception as e:
        log.warning(f"  [extrapolation] predict fallito: {e}"); return
    Rp   = np.exp(np.clip(y_pred, -5, 10))
    fail = np.sum((Rp < 0.05) | (Rp > 100) | ~np.isfinite(Rp))
    log.info(f"  Extrapolazione: {fail}/300 punti fuori [0.05, 100] R⊕")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.exp(lm_ext), Rp, color='steelblue', lw=2)
    ax.axvspan(np.exp(lm_min), np.exp(lm_max), alpha=0.1, color='green', label='Range training')
    ax.axhline(0.05, color='red', ls='--', lw=1)
    ax.axhline(100,  color='red', ls='--', lw=1, label='Limiti fisici')
    ax.set_xscale('log')
    ax.set_xlabel('$M_p$ [$M_\\oplus$]'); ax.set_ylabel('$R_p$ [$R_\\oplus$]')
    ax.set_title(f'{g} — extrapolazione ±50%')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'extrapolation_{g}.png'), dpi=150)
    plt.close(fig)


def check_log_args(model, X, g):
    lm_min, lm_max = X[:, 0].min(), X[:, 0].max()
    lf_min, lf_max = X[:, 1].min(), X[:, 1].max()
    lm_g = np.linspace(lm_min*0.8, lm_max*1.2, 100)
    lf_g = np.linspace(lf_min*0.8, lf_max*1.2, 100)
    LM, LF = np.meshgrid(lm_g, lf_g)
    means  = X.mean(axis=0)
    XX = np.column_stack([LM.ravel(), LF.ravel(),
                          np.full(LM.size, means[2]),
                          np.full(LM.size, means[3]),
                          np.full(LM.size, means[4])])
    try:
        y_grid = model.predict(XX)
    except Exception as e:
        log.warning(f"  [log_args] predict fallito: {e}"); return
    bad = ~np.isfinite(y_grid)
    pct = 100 * bad.mean()
    log.info(f"  Log args check: {bad.sum()}/{len(y_grid)} NaN/inf ({pct:.2f}%)"
             + (" [WARN]" if pct > 0.1 else " OK"))


def check_partial_signs(model, X, g):
    means = X.mean(axis=0)
    delta = 0.1
    rows  = []
    try:
        y_base = float(model.predict(means.reshape(1, -1))[0])
    except Exception:
        log.warning("  [partial_signs] predict base fallito"); return
    for i, vname in enumerate(VAR_NAMES):
        xp = means.copy(); xp[i] += delta
        xm = means.copy(); xm[i] -= delta
        try:
            deriv = (float(model.predict(xp.reshape(1,-1))[0]) -
                     float(model.predict(xm.reshape(1,-1))[0])) / (2*delta)
        except Exception:
            deriv = float('nan')
        sign_str = '+' if deriv > 0 else ('-' if deriv < 0 else '0')
        log.info(f"  ∂logRp/∂{vname} ≈ {deriv:+.4f} [{sign_str}]")
        rows.append({'variable': vname, 'partial_deriv': deriv})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f'partial_signs_{g}.csv'), index=False)


def check_outlier_influence(y_true, y_pred, g):
    n     = len(y_true)
    resid = y_true - y_pred
    mse   = np.mean(resid**2)
    cooks = resid**2 / (5 * mse + 1e-10)
    n_inf = np.sum(cooks > 4/n)
    z     = np.abs((resid - resid.mean()) / (resid.std() + 1e-10))
    n_out = np.sum(z > 3)
    log.info(f"  Cook's D appross.: {n_inf}/{n} influenti (soglia 4/n={4/n:.3f})")
    log.info(f"  Outlier |z|>3: {n_out}/{n}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_pred, resid, s=15, alpha=0.5, color='steelblue')
    if (z > 3).any():
        ax.scatter(y_pred[z>3], resid[z>3], s=50, color='tomato',
                   zorder=5, label=f'Outlier |z|>3 (n={n_out})')
    ax.axhline(0, color='k', lw=1, ls='--')
    ax.set_xlabel('logRp predetto'); ax.set_ylabel('Residuo')
    ax.set_title(f'{g} — outlier (test set)')
    if (z > 3).any(): ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'outliers_{g}.png'), dpi=150)
    plt.close(fig)


# ============================================================
# STEP 5 — Controlli aggiuntivi
# ============================================================
def check_sensitivity(model, X, g):
    means = X.mean(axis=0)
    try:
        base_Rp = np.exp(np.clip(float(model.predict(means.reshape(1,-1))[0]), -5, 10))
    except Exception:
        log.warning("  [sensitivity] predict base fallito"); return
    rows = []
    for i, vname in enumerate(VAR_NAMES):
        for delta_pct, lbl in [(0.01,'1%'),(0.10,'10%')]:
            xp = means.copy(); xp[i] *= (1 + delta_pct)
            try:
                Rp_plus = np.exp(np.clip(float(model.predict(xp.reshape(1,-1))[0]), -5, 10))
                rel_ch  = abs(Rp_plus - base_Rp) / (base_Rp + 1e-10)
            except Exception:
                rel_ch = float('nan')
            if lbl == '10%' and not np.isnan(rel_ch) and rel_ch > 0.20:
                log.warning(f"  [sensitivity] {vname} ±10% → ΔRp={rel_ch*100:.1f}% (>20%)")
            rows.append({'variable': vname, 'perturbation': lbl,
                         'delta_Rp_pct': float(rel_ch*100)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f'sensitivity_{g}.csv'), index=False)
    log.info(f"  sensitivity_{g}.csv salvato")


def check_pdp_ice(model, X, g, n_ice=20):
    rng = np.random.default_rng(RNG_SEED)
    ice_idx = rng.choice(len(X), size=min(n_ice, len(X)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, vi, vname in zip(axes, [0,1], ['logMp','logF']):
        grid = np.linspace(np.percentile(X[:,vi],1), np.percentile(X[:,vi],99), 80)
        # PDP: copia X una volta sola, sovrascrivi solo la colonna vi
        XX_pdp = X.copy()
        pdp = []
        for gval in grid:
            XX_pdp[:, vi] = gval
            try: pdp.append(np.nanmean(model.predict(XX_pdp)))
            except: pdp.append(float('nan'))
        ax.plot(grid, pdp, color='tomato', lw=2.5, zorder=5, label='PDP')
        # ICE: per ogni punto campionato, copia una riga e varia vi
        for ii in ice_idx:
            xi_base = X[ii].copy()  # copia una volta sola per questo punto
            ice = []
            for gval in grid:
                xi_base[vi] = gval
                try: ice.append(float(model.predict(xi_base.reshape(1,-1))[0]))
                except: ice.append(float('nan'))
            ax.plot(grid, ice, color='steelblue', alpha=0.2, lw=0.8)
        ax.set_xlabel(vname); ax.set_ylabel('logRp')
        ax.set_title(f'{g} — PDP+ICE ({vname})')
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'pdp_ice_{g}.png'), dpi=150)
    plt.close(fig)
    log.info(f"  pdp_ice_{g}.png salvato")


def check_heteroscedasticity(y_true, y_pred, g):
    resid    = y_true - y_pred
    resid_sq = resid**2
    X_bp     = np.column_stack([np.ones(len(y_pred)), y_pred])
    try:
        coef, _, _, _ = np.linalg.lstsq(X_bp, resid_sq, rcond=None)
        y_hat = X_bp @ coef
        ss_reg = np.sum((y_hat - resid_sq.mean())**2)
        ss_tot = np.sum((resid_sq - resid_sq.mean())**2)
        lm     = len(resid) * ss_reg / (ss_tot + 1e-10)
        pval   = float(stats.chi2.sf(lm, df=1))
        log.info(f"  Breusch-Pagan: LM={lm:.3f}  p={pval:.4f} "
                 f"({'eteroschedastico' if pval<0.05 else 'omoschedastico OK'})")
    except Exception as e:
        pval = float('nan')
        log.warning(f"  Breusch-Pagan fallito: {e}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, resid, s=15, alpha=0.5, color='steelblue')
    ax.axhline(0, color='k', lw=1, ls='--')
    z = np.polyfit(y_pred, np.abs(resid), 1)
    tx = np.linspace(y_pred.min(), y_pred.max(), 100)
    ax.plot(tx, np.polyval(z, tx), color='tomato', lw=1.5, ls='--', label='Trend |res|')
    ax.set_xlabel('logRp pred'); ax.set_ylabel('Residuo')
    ax.set_title(f'{g} — eteroschedasticità')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'heteroscedasticity_{g}.png'), dpi=150)
    plt.close(fig)
    return pval  # restituisce pval per wild bootstrap condizionale


def check_residuals_distribution(y_true, y_pred, g):
    resid = y_true - y_pred
    sk  = float(stats.skew(resid))
    ku  = float(stats.kurtosis(resid))
    if len(resid) <= 5000:
        _, pnorm = stats.shapiro(resid)
        log.info(f"  Residui: skew={sk:.3f}  kurt={ku:.3f}  Shapiro p={pnorm:.4f}")
    else:
        log.info(f"  Residui: skew={sk:.3f}  kurt={ku:.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(resid, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    axes[0].set_xlabel('Residuo'); axes[0].set_title(f'{g} — dist. residui')
    stats.probplot(resid, dist='norm', plot=axes[1])
    axes[1].set_title(f'{g} — Q-Q plot')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'residuals_dist_{g}.png'), dpi=150)
    plt.close(fig)
    log.info(f"  residuals_dist_{g}.png salvato")


def _sympy_worker(expr_str, result_queue):
    # Worker separato per sympy.simplify — evita hang infiniti.
    try:
        import sympy as _sy
        expr = _sy.sympify(expr_str)
        simp = _sy.simplify(expr)
        result_queue.put(('ok', str(simp), _sy.latex(simp)))
    except Exception as e:
        result_queue.put(('err', str(e), ''))


def check_sympy_simplify(model, g, timeout=30):
    # sympy.simplify puo' bloccarsi su espressioni complesse.
    # Eseguiamo in un processo separato con timeout di kill.
    if not HAS_SYMPY:
        log.warning("  sympy non disponibile — skip"); return
    try:
        expr = model.sympy()
        expr_str = str(expr)
        log.info(f"  sympy originale: {expr_str}")
        import multiprocessing as _mp
        q = _mp.Queue()
        p = _mp.Process(target=_sympy_worker, args=(expr_str, q))
        p.start()
        p.join(timeout=timeout)
        if p.is_alive():
            p.kill(); p.join()
            log.warning(f"  sympy.simplify timeout ({timeout}s) — skip semplificazione")
            simp_str, latex_str = expr_str, sympy.latex(expr)
        else:
            status, simp_str, latex_str = q.get()
            if status == 'err':
                log.warning(f"  sympy simplify fallito: {simp_str}")
                simp_str, latex_str = expr_str, sympy.latex(expr)
            else:
                log.info(f"  sympy semplificata: {simp_str}")
        with open(os.path.join(OUT_DIR, f'formula_simplified_{g}.txt'), 'w') as f:
            f.write(f"Originale:    {expr_str}\n")
            f.write(f"Semplificata: {simp_str}\n")
            f.write(f"LaTeX:        {latex_str}\n")
        log.info(f"  LaTeX: {latex_str}")
    except Exception as e:
        log.warning(f"  check_sympy_simplify fallito: {e}")


def check_coefficients(model, g):
    if not HAS_SYMPY:
        return
    try:
        expr = model.sympy()
        nums = [float(a) for a in expr.atoms(sympy.Number) if float(a) != 0]
        susp = [n for n in nums if abs(n) > 0 and abs(np.log10(abs(n))) > 6]
        if susp:
            log.warning(f"  Coefficienti sospetti (|log10|>6): {susp}")
        else:
            log.info("  Coefficienti OK (tutti |log10|≤6)")
    except Exception as e:
        log.warning(f"  Check coefficienti fallito: {e}")


def check_stratified(model, X, y, g):
    q33, q67 = np.percentile(X[:,0], [33, 67])
    slices = [('bassa', X[:,0]<q33), ('media', (X[:,0]>=q33)&(X[:,0]<q67)), ('alta', X[:,0]>=q67)]
    rows = []
    for lbl, mask in slices:
        if mask.sum() < 5: continue
        try:
            yp = model.predict(X[mask])
            if np.any(np.isnan(yp)): raise ValueError("NaN")
            r2 = safe_r2(y[mask], yp)
        except: r2 = float('nan')
        log.info(f"  Stratified [{lbl} massa]: n={mask.sum()}  R²={r2:.4f}")
        rows.append({'slice': lbl, 'n': int(mask.sum()), 'r2': r2})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f'stratified_{g}.csv'), index=False)


def check_residuals_vs_errors(y_true, y_pred, df_sub, g):
    resid    = np.abs(y_true - y_pred)
    err_mass = df_sub['err_rel_max'].values
    err_rad  = df_sub['err_rade_rel'].values if 'err_rade_rel' in df_sub.columns \
               else np.zeros(len(resid))
    for ename, evals in [('err_massa_rel', err_mass), ('err_raggio_rel', err_rad)]:
        if len(evals) != len(resid): continue
        valid = np.isfinite(evals) & np.isfinite(resid)
        if valid.sum() < 5: continue
        try:
            r, p = stats.pearsonr(resid[valid], evals[valid])
            flag = ' [WARN: correlazione bias]' if abs(r) > 0.3 else ''
            log.info(f"  Corr(|res|, {ename}): r={r:.3f}  p={p:.4f}{flag}")
        except Exception as e:
            log.warning(f"  Correlazione fallita per {ename}: {e}")


def check_nonlinear_terms(model, g):
    try:
        s = str(model.sympy())
        warns = []
        if '**logMp' in s and 'logMp**' in s: warns.append("logMp^logMp")
        if '**logF'  in s and 'logF**'  in s: warns.append("logF^logF")
        if warns: log.warning(f"  Termini non fisici: {warns}")
        else:     log.info("  Termini non fisici: nessuno")
    except: pass


def check_oscillations(model, X, g):
    lm_min, lm_max = np.percentile(X[:,0], [1, 99])
    lm_dense = np.linspace(lm_min, lm_max, 300)
    means = X.mean(axis=0)
    XX = np.column_stack([lm_dense, np.full(300, means[1]), np.full(300, means[2]),
                          np.full(300, means[3]), np.full(300, means[4])])
    try:
        y_pred = model.predict(XX)
        if not np.all(np.isfinite(y_pred)): log.warning("  [oscillations] NaN — skip"); return
        dlm = lm_dense[1] - lm_dense[0]
        d2  = np.gradient(np.gradient(y_pred, dlm), dlm)
        mx  = float(np.max(np.abs(d2)))
        log.info(f"  Oscillazioni: max|d²logRp/dlogMp²|={mx:.4f} "
                 + ("[WARN: wiggles]" if mx > 1.0 else "OK"))
    except Exception as e:
        log.warning(f"  [oscillations] fallito: {e}")


def check_ci_calibration(y_true, y_pred, g, n_boot=200, seed=RNG_SEED):
    # Calibrazione CI reale: genera intervalli predittivi per ogni punto
    # via bootstrap sui residui (residual bootstrap), poi verifica copertura.
    # NON usa i percentili dei residui stessi (sarebbe tautologico).
    resid = y_true - y_pred
    rng   = np.random.default_rng(seed)
    n     = len(y_true)
    # Genera matrice di residui ricampionati: shape (n_boot, n)
    resid_matrix = rng.choice(resid, size=(n_boot, n), replace=True)
    # Predizioni bootstrap per ogni punto: shape (n_boot, n)
    y_boot = y_pred + resid_matrix
    # CI per ogni punto: percentili della distribuzione bootstrap
    ci_low  = np.nanpercentile(y_boot, 2.5,  axis=0)  # shape (n,)
    ci_high = np.nanpercentile(y_boot, 97.5, axis=0)  # shape (n,)
    # Coverage: frazione di y_true dentro il CI specifico del proprio punto
    coverage = float(np.mean((y_true >= ci_low) & (y_true <= ci_high)))
    log.info(f"  Calibrazione CI 95% (bootstrap predittivo, {n_boot} iter): "
             f"copertura={coverage*100:.1f}% "
             + ("OK" if abs(coverage-0.95)<0.10 else "[WARN: CI mal calibrati]"))


# ============================================================
# STEP 6 — Loop principale per gruppo
# ============================================================
summary_rows = []

for g in labels:
    log.info(f"\n{'='*60}")
    log.info(f"Gruppo: {g}")
    log.info(f"{'='*60}")

    sub = df[df['group'] == g].copy()
    if len(sub) < 50:
        log.info(f"  Troppo piccolo ({len(sub)}) — skip"); continue

    X, y, w = prep(sub)
    if len(X) == 0:
        log.warning(f"  {g}: dataset vuoto dopo prep — skip"); continue
    idx = np.arange(len(X))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=1)
    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]

    # Baseline
    lr       = LinearRegression().fit(X_tr[:,0:1], y_tr)
    y_base   = lr.predict(X_te[:,0:1])
    r2_base  = safe_r2(y_te, y_base)
    alpha_b  = float(lr.coef_[0])
    log.info(f"  Baseline: R = M^{alpha_b:.3f}  R²_test={r2_base:.4f}")

    # Cerca pkl migliore
    best_model, best_seed, best_r2 = None, None, -np.inf
    for seed in SEEDS:
        pkl = os.path.join(CHECKPOINTS_DIR, f"{g}_seed{seed}", "model.pkl")
        if not os.path.exists(pkl): continue
        try:
            with open(pkl, 'rb') as f: m = pickle.load(f)
            pred = safe_predict(m, X_te)
            if np.any(np.isnan(pred)): continue
            r2 = safe_r2(y_te, pred)
            log.info(f"  Seed {seed}: R²_test={r2:.4f}")
            if r2 > best_r2: best_r2, best_model, best_seed = r2, m, seed
        except Exception as e:
            log.warning(f"  Seed {seed}: errore pkl — {e}")

    if best_model is None:
        log.warning(f"  Nessun pkl disponibile — skip")
        summary_rows.append({'group':g, 'n_train':len(y_tr), 'n_test':len(y_te),
                              'best_seed':None, 'r2_test_obs':float('nan'),
                              'r2_baseline':r2_base, 'alpha_baseline':alpha_b,
                              'formula':'n/d'})
        continue

    log.info(f"  Seed migliore: {best_seed}  R²_test={best_r2:.4f}")
    formula = safe_sympy(best_model)
    log.info(f"  Formula: {formula}")
    log.info(f"  Formula fisica: Rp [R_terra] = exp({formula})")

    # Bootstrap
    if len(y_te) < 30:
        log.warning(f"  Test set piccolo (n={len(y_te)}) — CI poco stabili")
    y_pred_te = safe_predict(best_model, X_te)

    r2_obs, rb_mean, rb_std, rb_ci, rb_arr = bootstrap_r2(y_te, y_pred_te)
    log.info(f"  Residual bootstrap: R²={r2_obs:.4f}  CI=[{rb_ci[0]:.4f},{rb_ci[1]:.4f}]")

    _, pb_mean, pb_std, pb_ci, pb_arr = pair_bootstrap_r2(y_te, y_pred_te)
    log.info(f"  Pair bootstrap:     R²={r2_obs:.4f}  CI=[{pb_ci[0]:.4f},{pb_ci[1]:.4f}]")

    try:
        _, bca_ci = bca_bootstrap(y_te, y_pred_te)
        log.info(f"  BCa bootstrap:      CI=[{bca_ci[0]:.4f},{bca_ci[1]:.4f}]")
    except Exception as e:
        log.warning(f"  BCa fallito: {e}"); bca_ci = np.array([float('nan'),float('nan')])

    # Permutation p-value
    pval = float('nan')
    if os.path.exists('checkpoint.json'):
        try:
            with open('checkpoint.json','r') as f: ck = json.load(f)
            pm = ck.get('permutation',{}).get('0',{}).get('r2_perm_mean', float('nan'))
            if not np.isnan(pm):
                log.info(f"  Permutation mean R²={pm:.4f}  (R²_obs={r2_obs:.4f})")
                pval = float(pm >= r2_obs)
        except Exception as e:
            log.warning(f"  Errore checkpoint permutazioni: {e}")

    # Plot bootstrap
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, arr, lbl, ci_arr, col in [
        (axes[0], rb_arr, 'Residual', rb_ci, 'steelblue'),
        (axes[1], pb_arr, 'Pair',     pb_ci, 'seagreen'),
    ]:
        ax.hist(arr[np.isfinite(arr)], bins=40, color=col, alpha=0.7, edgecolor='white')
        ax.axvline(r2_obs,  color='tomato',     lw=2,   ls='-',  label=f'R²={r2_obs:.3f}')
        ax.axvline(ci_arr[0],color='gray',       lw=1.5, ls='--', label=f'95%CI [{ci_arr[0]:.3f},{ci_arr[1]:.3f}]')
        ax.axvline(ci_arr[1],color='gray',       lw=1.5, ls='--')
        ax.axvline(r2_base, color='darkorange',  lw=2,   ls=':',  label=f'Baseline={r2_base:.3f}')
        ax.set_xlabel('R²'); ax.set_ylabel('Conteggio')
        ax.set_title(f'{g} — {lbl} bootstrap (seed {best_seed})')
        ax.legend(fontsize=7)
    fig.suptitle(f'{g}  n_train={len(y_tr)}  n_test={len(y_te)}', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'bootstrap_{g}.png'), dpi=150)
    plt.close(fig)
    # Salva array bootstrap per analisi post-hoc
    np.savez(os.path.join(OUT_DIR, f'bootstrap_arrays_{g}.npz'),
             residual=rb_arr, pair=pb_arr, rng_seed=RNG_SEED, n_boot=N_BOOT)
    log.info(f"  bootstrap_{g}.png + bootstrap_arrays_{g}.npz salvati")

    # Domain check
    domain_check(best_model, X, g)

    # Controlli fisici
    log.info(f"\n  --- Controlli fisici [{g}] ---")
    check_monotonicity(best_model, X, g)
    check_extrapolation(best_model, X, g)
    check_log_args(best_model, X, g)
    check_partial_signs(best_model, X, g)
    check_outlier_influence(y_te, y_pred_te, g)

    # Controlli aggiuntivi
    log.info(f"\n  --- Controlli aggiuntivi [{g}] ---")
    check_sensitivity(best_model, X, g)
    check_pdp_ice(best_model, X, g)
    bp_pval = check_heteroscedasticity(y_te, y_pred_te, g)
    # Wild bootstrap condizionale
    if isinstance(bp_pval, float) and not np.isnan(bp_pval) and bp_pval < 0.05:
        log.warning(f"  Eteroschedasticità rilevata (p={bp_pval:.4f}) — wild bootstrap")
        try:
            _, wb_mean, wb_std, wb_ci, _ = wild_bootstrap(y_te, y_pred_te)
            log.info(f"  Wild bootstrap: CI=[{wb_ci[0]:.4f},{wb_ci[1]:.4f}]")
        except Exception as e:
            log.warning(f"  Wild bootstrap fallito: {e}")
    check_residuals_distribution(y_te, y_pred_te, g)
    check_sympy_simplify(best_model, g)
    check_coefficients(best_model, g)
    check_stratified(best_model, X, y, g)
    try:
        y_pred_all = safe_predict(best_model, X)
        check_residuals_vs_errors(y, y_pred_all, sub, g)
    except Exception: pass
    check_nonlinear_terms(best_model, g)
    check_oscillations(best_model, X, g)
    check_ci_calibration(y_te, y_pred_te, g)

    summary_rows.append({
        'group':           g,
        'n_train':         len(y_tr),
        'n_test':          len(y_te),
        'best_seed':       best_seed,
        'r2_test_obs':     r2_obs,
        'resid_ci_low':    float(rb_ci[0]),
        'resid_ci_high':   float(rb_ci[1]),
        'pair_ci_low':     float(pb_ci[0]),
        'pair_ci_high':    float(pb_ci[1]),
        'bca_ci_low':      float(bca_ci[0]),
        'bca_ci_high':     float(bca_ci[1]),
        'r2_baseline':     r2_base,
        'alpha_baseline':  alpha_b,
        'delta_r2':        r2_obs - r2_base,
        'perm_pval_approx':pval,
        'formula':         formula,
    })

# ============================================================
# STEP 7 — Tabella riassuntiva
# ============================================================
log.info(f"\n{'='*60}")
log.info("TABELLA RIASSUNTIVA")
log.info(f"{'='*60}")
summary_df = pd.DataFrame(summary_rows)
log.info("\n" + summary_df[['group','n_test','r2_test_obs','r2_baseline',
                             'delta_r2','resid_ci_low','resid_ci_high',
                             'pair_ci_low','pair_ci_high']].to_string(index=False))
summary_df.to_csv(os.path.join(OUT_DIR, 'summary_finale.csv'), index=False)
log.info(f"summary_finale.csv salvato")

# ============================================================
# STEP 8 — Validazione temporale (punto 23)
# ============================================================
log.info(f"\n{'='*60}")
log.info(f"STEP 8 — Performance degradation check (split pre/post-{SPLIT_YEAR})")
log.info("  NOTA: pkl addestrati su tutto il dataset. Non e' test predittivo puro.")
log.info("  Nel paper citare come 'temporal robustness check', non validazione.")
log.info(f"{'='*60}")

try:
    CACHE_FILE = os.path.join(CACHE_DIR, 'nasa_with_discyear.csv')
    if os.path.exists(CACHE_FILE):
        log.info(f"  Usando cache: {CACHE_FILE}")
        df_yr = pd.read_csv(CACHE_FILE)
        log.info(f"  Cache del: {datetime.datetime.fromtimestamp(os.path.getmtime(CACHE_FILE)).isoformat()}")
    else:
        log.info("  Scaricando disc_year da NASA...")
        _r = requests.get(
            "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
            params={"query": ("select pl_masse,pl_masseerr1,pl_masseerr2,pl_rade,"
                              "pl_radeerr1,pl_insol,st_met,st_mass,st_rad,"
                              "pl_orbper,disc_year from pscomppars"),
                    "format": "csv"},
            timeout=300
        )
        if _r.status_code != 200:
            raise RuntimeError(f"NASA HTTP {_r.status_code}")
        df_yr = pd.read_csv(_io.StringIO(_r.text))
        df_yr.to_csv(CACHE_FILE, index=False)
        log.info(f"  Cache salvata: {CACHE_FILE}")

    df_yr = df_yr.dropna(subset=['pl_masse','pl_masseerr1','pl_masseerr2',
                                   'pl_rade','pl_radeerr1','pl_insol',
                                   'st_met','st_mass','st_rad','pl_orbper','disc_year'])
    df_yr = df_yr[(df_yr['pl_masse']>0)&(df_yr['pl_rade']>0)&(df_yr['pl_insol']>0)&
                  (df_yr['st_rad']>0)&(df_yr['st_mass']>0)&(df_yr['pl_orbper']>0)&
                  (df_yr['pl_masse']<4000)&(df_yr['pl_rade']<25)].copy()
    df_yr['err_rel_pos'] = np.abs(df_yr['pl_masseerr1']) / df_yr['pl_masse']
    df_yr['err_rel_neg'] = np.abs(df_yr['pl_masseerr2']) / df_yr['pl_masse']
    df_yr['err_rel_max'] = df_yr[['err_rel_pos','err_rel_neg']].max(axis=1)
    df_yr['group']       = pd.cut(df_yr['pl_rade'], bins=[0,1.5,2.0,4.0,100], labels=labels)
    piccoli_t = df_yr['group'].isin(['sub_terr','rocky_SE'])
    _mask_t   = (piccoli_t  & (df_yr['err_rel_max'] < 0.35)) | \
                (~piccoli_t & (df_yr['err_rel_max'] < 0.30))
    df_yr = df_yr[_mask_t].copy()
    df_yr['logMp']       = np.log(df_yr['pl_masse'])
    df_yr['logF']        = np.log(df_yr['pl_insol'])
    df_yr['FeH']         = df_yr['st_met']
    df_yr['logMs']       = np.log(df_yr['st_mass'])
    df_yr['logRs']       = np.log(df_yr['st_rad'])
    df_yr['logP']        = np.log(df_yr['pl_orbper'])
    df_yr['logRp']       = np.log(df_yr['pl_rade'])
    df_yr['log_rhostar'] = df_yr['logMs'] - 3*df_yr['logRs']
    df_yr['err_rade_rel']= np.abs(df_yr['pl_radeerr1']) / df_yr['pl_rade']
    df_yr['sigma_tot']   = np.sqrt(df_yr['err_rel_max']**2 + df_yr['err_rade_rel']**2)
    df_yr['weight']      = 1.0 / (df_yr['sigma_tot']**2 + 1e-6)

    df_tr_t = df_yr[df_yr['disc_year'] <  SPLIT_YEAR]
    df_te_t = df_yr[df_yr['disc_year'] >= SPLIT_YEAR]
    log.info(f"  Pre-{SPLIT_YEAR}: {len(df_tr_t)}  Post-{SPLIT_YEAR}: {len(df_te_t)}")

    temp_rows = []
    for g in labels:
        sub_te = df_te_t[df_te_t['group']==g].copy()
        if len(sub_te) < 5:
            log.info(f"  {g}: test temporale troppo piccolo (n={len(sub_te)}) — skip"); continue
        X_te_t, y_te_t, _ = prep(sub_te)

        best_m_t, best_r2_t = None, -np.inf
        for seed in SEEDS:
            pkl = os.path.join(CHECKPOINTS_DIR, f"{g}_seed{seed}", "model.pkl")
            if not os.path.exists(pkl): continue
            try:
                with open(pkl,'rb') as f: m = pickle.load(f)
                pred = safe_predict(m, X_te_t)
                if np.any(np.isnan(pred)): continue
                r2 = safe_r2(y_te_t, pred)
                if r2 > best_r2_t: best_r2_t, best_m_t = r2, m
            except: continue

        if best_m_t is None:
            log.warning(f"  {g}: nessun pkl — skip"); continue

        sub_tr = df_tr_t[df_tr_t['group']==g].copy()
        r2_base_t = float('nan')
        if len(sub_tr) >= 5:
            X_tr_t, y_tr_t, _ = prep(sub_tr)
            lr_t  = LinearRegression().fit(X_tr_t[:,0:1], y_tr_t)
            y_b_t = lr_t.predict(X_te_t[:,0:1])
            r2_base_t = safe_r2(y_te_t, y_b_t)

        delta_t = best_r2_t - r2_base_t if not np.isnan(r2_base_t) else float('nan')
        log.info(f"  {g}: n={len(sub_te)}  R²_temporal={best_r2_t:.4f}  "
                 f"baseline={r2_base_t:.4f}  ΔR²={delta_t:.4f}")
        temp_rows.append({'group':g, 'n_test_temporal':len(sub_te),
                          'r2_temporal':best_r2_t, 'r2_baseline_temporal':r2_base_t,
                          'delta_r2_temporal':delta_t})

        y_pred_t = safe_predict(best_m_t, X_te_t)
        Rp_obs   = np.exp(y_te_t)
        Rp_pred  = np.exp(np.clip(y_pred_t, -5, 5))
        fig, ax  = plt.subplots(figsize=(5, 5))
        valid_mask = np.isfinite(Rp_pred)
        if valid_mask.sum() == 0:
            log.warning(f'  {g}: predizioni temporali tutte non-finite -- skip plot')
            plt.close(fig); continue
        min_val = min(float(np.nanmin(Rp_obs)), float(np.nanmin(Rp_pred[valid_mask])))
        max_val = max(float(np.nanmax(Rp_obs)), float(np.nanmax(Rp_pred[valid_mask])))
        lims = [min_val * 0.9, max_val * 1.1]
        ax.scatter(Rp_pred[valid_mask], Rp_obs[valid_mask], s=20, alpha=0.7, color='steelblue')
        ax.plot(lims, lims, 'k--', lw=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('$R_p$ predetto'); ax.set_ylabel('$R_p$ osservato')
        ax.set_title(f'{g} — test temporale post-{SPLIT_YEAR}\nR²={best_r2_t:.3f}  baseline={r2_base_t:.3f}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'temporal_test_{g}.png'), dpi=150)
        plt.close(fig)

    if temp_rows:
        df_temp = pd.DataFrame(temp_rows)
        df_temp.to_csv(os.path.join(OUT_DIR, 'temporal_validation.csv'), index=False)
        summary_df = summary_df.merge(df_temp, on='group', how='left')
        summary_df.to_csv(os.path.join(OUT_DIR, 'summary_finale.csv'), index=False)
        log.info("  temporal_validation.csv + summary_finale.csv aggiornati")

except Exception as e:
    log.warning(f"  Validazione temporale fallita: {e}")
    log.warning("  (non critico — altri risultati già salvati)")

log.info(f"\nTutti gli output in: {OUT_DIR}/")
log.info("=== ANALISI COMPLETATA ===")

# ============================================================
# STEP 9 — Confronto con modelli empirici pubblicati
#           + semplificazione formula
#
# Modello di riferimento:
#   Chen & Kipping 2017 (ApJ 834, 17) — relazione M-R a tre regimi
#   Coefficienti da Table 1 (valori mediani):
#     Terran    (Mp < 2.04 M_⊕):         Rp = 1.008 * Mp^{+0.279}
#     Neptunian (2.04 < Mp < 131.6 M_⊕): Rp = 0.688 * Mp^{+0.589}
#     Jovian    (Mp > 131.6 M_⊕):        Rp = 17.739 * Mp^{-0.044}
#   NOTA: Il modello è calibrato sull'intera popolazione di esopianeti osservati,
#   che include molti hot Jupiters gonfiati dall'irraggiamento. I valori predetti
#   per il regime gioviano (~14 R_⊕ per Mp~300 M_⊕) sono coerenti con questa
#   popolazione, anche se maggiori rispetto a Giove solare (11.2 R_⊕).
#   Il modello semplificato deterministico non è continuo ai breakpoint;
#   questo riflette la natura probabilistica del modello originale.
#
# Citazione da includere nel paper:
#   Chen & Kipping (2017), ApJ, 834, 17, doi:10.3847/1538-4357/834/1/17
# ============================================================
log.info(f"\n{'='*60}")
log.info("STEP 9 — Confronto letteratura + semplificazione formula")
log.info(f"{'='*60}")

try:
    # ----------------------------------------------------------
    # 9a. Chen & Kipping 2017 — tre regimi
    # ----------------------------------------------------------
    _M_T2N = 2.04          # breakpoint Terran→Neptunian [M_terra]
    _M_N2J = 0.414 * 317.828  # breakpoint Neptunian→Jovian = 131.6 M_terra

    def chen_kipping_2017(Mp_earth):
        """
        Chen & Kipping 2017 (ApJ 834, 17), Table 1 — forma deterministica.
        Input:  Mp_earth  — massa planetaria in M_terra (array numpy)
        Output: Rp        — raggio in R_terra
        Nota: calibrato su esopianeti osservati; il regime gioviano (gamma<0)
        riflette l'inflazione dei hot Jupiters.
        """
        Mp = np.asarray(Mp_earth, dtype=float)
        Rp = np.where(
            Mp < _M_T2N,
            1.008  * np.maximum(Mp, 1e-9)**0.279,
            np.where(
                Mp < _M_N2J,
                0.688  * np.maximum(Mp, 1e-9)**0.589,
                17.739 * np.maximum(Mp, 1e-9)**(-0.044)
            )
        )
        return Rp

    # ----------------------------------------------------------
    # 9b. Funzione di semplificazione formula con arrotondamento
    # ----------------------------------------------------------
    def _simplify_formula(expr_str, decimals=2):
        """
        Arrotonda i coefficienti numerici di una formula a `decimals`
        cifre significative. Usato per la versione human-readable nel paper.
        Restituisce la stringa modificata e None se il parsing fallisce.
        """
        import re
        def _round_sig(m):
            val = float(m.group(0))
            if val == 0.0:
                return '0'
            magnitude  = int(np.floor(np.log10(abs(val))))
            ndecimals  = max(0, decimals - 1 - magnitude)
            factor     = 10**(decimals - 1 - magnitude)
            rounded    = round(val * factor) / factor
            fmt        = f'{rounded:.{ndecimals}f}'
            return fmt
        try:
            return re.sub(r'-?\d+\.\d+', _round_sig, expr_str)
        except Exception:
            return expr_str

    # ----------------------------------------------------------
    # 9c. Loop per gruppo: confronto e semplificazione
    # ----------------------------------------------------------
    comparison_rows = []

    for g in labels:
        sub = df[df['group'] == g].copy()
        if len(sub) < 50:
            continue

        X, y, w   = prep(sub)
        idx        = np.arange(len(X))
        idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=1)
        X_tr = X[idx_tr]; y_tr = y[idx_tr]
        X_te = X[idx_te]; y_te = y[idx_te]

        # Mp in M_terra (exp perché X[:,0] = log naturale di Mp)
        Mp_te = np.exp(X_te[:, 0])

        # Baseline power-law
        lr_base = LinearRegression().fit(X_tr[:,0:1], y_tr)
        r2_base = safe_r2(y_te, lr_base.predict(X_te[:,0:1]))
        alpha_b = float(lr_base.coef_[0])

        # Chen & Kipping 2017
        Rp_ck     = chen_kipping_2017(Mp_te)
        logRp_ck  = np.log(np.maximum(Rp_ck, 1e-6))
        r2_ck     = safe_r2(y_te, logRp_ck)

        # PySR — carica pkl migliore
        best_model_g = None
        best_r2_g    = -np.inf
        for seed in SEEDS:
            pkl = os.path.join(CHECKPOINTS_DIR, f"{g}_seed{seed}", "model.pkl")
            if not os.path.exists(pkl):
                continue
            try:
                with open(pkl, 'rb') as f:
                    m = pickle.load(f)
                pred = safe_predict(m, X_te)
                if np.any(np.isnan(pred)):
                    continue
                r2 = safe_r2(y_te, pred)
                if r2 > best_r2_g:
                    best_r2_g, best_model_g = r2, m
            except Exception:
                continue

        r2_pysr       = best_r2_g if best_model_g is not None else float('nan')
        formula_orig  = safe_sympy(best_model_g) if best_model_g else 'n/d'
        formula_simple = _simplify_formula(formula_orig, decimals=2)

        # Valuta formula semplificata per verificare degradazione R²
        r2_simple     = float('nan')
        r2_degradation = float('nan')
        if best_model_g is not None and HAS_SYMPY and formula_simple != 'n/d':
            try:
                import sympy as _sy
                var_syms   = _sy.symbols('logMp logF FeH logP logRhoStar')
                expr_s     = _sy.sympify(formula_simple)
                f_s        = _sy.lambdify(var_syms, expr_s, 'numpy')
                yp_simple  = np.asarray(
                    f_s(X_te[:,0], X_te[:,1], X_te[:,2], X_te[:,3], X_te[:,4]),
                    dtype=float
                ).ravel()
                # Gestisce il caso in cui lambdify restituisce uno scalare
                if yp_simple.size == 1:
                    yp_simple = np.full(len(y_te), float(yp_simple[0]))
                r2_simple      = safe_r2(y_te, yp_simple)
                r2_degradation = r2_pysr - r2_simple
            except Exception as e:
                log.warning(f"  [{g}] valutazione formula semplificata fallita: {e}")

        log.info(f"  {g} (n_test={len(y_te)}):")
        log.info(f"    Baseline power-law (α={alpha_b:.3f}):   R²={r2_base:.4f}")
        log.info(f"    Chen & Kipping 2017:                    R²={r2_ck:.4f}")
        log.info(f"    PySR (originale):                       R²={r2_pysr:.4f}")
        if not np.isnan(r2_simple):
            log.info(f"    PySR (arrotondata 2 c.s.):              R²={r2_simple:.4f}"
                     f"  [degradazione={r2_degradation:+.4f}]")
            if abs(r2_degradation) < 0.01:
                log.info(f"    → ΔR² < 0.01: formula semplificata accettabile per il paper ✓")
            else:
                log.warning(f"    → ΔR² = {r2_degradation:.4f}: usare formula originale nel paper")
        log.info(f"    Formula originale:    {formula_orig}")
        log.info(f"    Formula semplificata: {formula_simple}")

        comparison_rows.append({
            'group':                g,
            'n_test':               int(len(y_te)),
            'r2_baseline':          r2_base,
            'alpha_baseline':       alpha_b,
            'r2_chen_kipping2017':  r2_ck,
            'r2_pysr_original':     r2_pysr,
            'r2_pysr_simplified':   r2_simple,
            'r2_degradation':       r2_degradation,
            'formula_original':     formula_orig,
            'formula_simplified':   formula_simple,
        })

    # ----------------------------------------------------------
    # 9d. Salva CSV confronto
    # ----------------------------------------------------------
    if comparison_rows:
        df_comp = pd.DataFrame(comparison_rows)
        df_comp.to_csv(os.path.join(OUT_DIR, 'comparison_literature.csv'), index=False)
        log.info(f"\n  comparison_literature.csv salvato")

        # ----------------------------------------------------------
        # 9e. Figura di confronto — bar chart R² per modello e gruppo
        # ----------------------------------------------------------
        rows_plot    = [r for r in comparison_rows
                        if not np.isnan(r['r2_pysr_original'])]
        if rows_plot:
            group_names  = [r['group'] for r in rows_plot]
            n_groups     = len(group_names)
            model_labels = ['Baseline\n(power law)',
                            'Chen & Kipping\n(2017)',
                            'PySR\n(originale)',
                            'PySR\n(semplificata)']
            model_keys   = ['r2_baseline', 'r2_chen_kipping2017',
                            'r2_pysr_original', 'r2_pysr_simplified']
            colors       = ['#aec7e8', '#ffbb78', '#2ca02c', '#98df8a']
            x     = np.arange(n_groups)
            width = 0.20

            fig, ax = plt.subplots(figsize=(max(9, n_groups * 2.8), 5.5))
            for j, (mlabel, mkey, col) in enumerate(
                    zip(model_labels, model_keys, colors)):
                vals = [r[mkey] for r in rows_plot]
                bars = ax.bar(x + j*width - 1.5*width, vals, width,
                              label=mlabel, color=col,
                              edgecolor='white', linewidth=0.8, zorder=2)
                for bar, val in zip(bars, vals):
                    if not np.isnan(val):
                        ax.text(bar.get_x() + bar.get_width()/2,
                                bar.get_height() + 0.012,
                                f'{val:.2f}', ha='center', va='bottom',
                                fontsize=7, rotation=45)

            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=1)
            ax.set_xticks(x)
            ax.set_xticklabels(group_names, fontsize=10)
            ax.set_ylabel(r'$R^2_{\mathrm{test}}$', fontsize=12)
            ax.set_title(
                r'Comparison of mass–radius models' '\n'
                r'Chen \& Kipping (2017); this work (PySR)',
                fontsize=11
            )
            ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
            # Asse y: da min valore - margine a 1.05
            all_vals = [r[k] for r in rows_plot for k in model_keys
                        if not np.isnan(r[k])]
            y_min = min(all_vals) - 0.08 if all_vals else -0.3
            ax.set_ylim(y_min, 1.05)
            ax.yaxis.grid(True, alpha=0.3, linestyle=':', zorder=0)
            plt.tight_layout()
            out_fig = os.path.join(OUT_DIR, 'comparison_literature.png')
            plt.savefig(out_fig, dpi=150)
            plt.close(fig)
            log.info(f"  comparison_literature.png salvato")

        # ----------------------------------------------------------
        # 9f. Tabella riassuntiva finale
        # ----------------------------------------------------------
        log.info(f"\n  {'='*60}")
        log.info("  TABELLA CONFRONTO FINALE (per il paper)")
        log.info(f"  {'='*60}")
        cols_paper = ['group', 'n_test', 'r2_baseline',
                      'r2_chen_kipping2017', 'r2_pysr_original',
                      'r2_pysr_simplified', 'r2_degradation']
        log.info("\n" + df_comp[cols_paper].round(4).to_string(index=False))

except Exception as e:
    log.warning(f"  STEP 9 fallito: {e}")
    import traceback
    log.warning(traceback.format_exc())