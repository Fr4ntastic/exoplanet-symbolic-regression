import os
import sys
import time
import json
import pickle
import shutil
import signal
import inspect
import logging
import subprocess
import numpy as np
import pandas as pd
import requests
import io
import matplotlib
matplotlib.use('Agg')  # headless, nessun display necessario
import matplotlib.pyplot as plt
from pysr import PySRRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ============================================================
# LOGGING — scrive su console e su file, tail -f run_sr.log
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('run_sr.log', mode='a'),
    ]
)
log = logging.getLogger(__name__)

# ============================================================
# VERSIONI — riproducibilità
# ============================================================
import pysr, sklearn
_julia_ver = "n/d"
try:
    _julia_ver = subprocess.check_output(
        ["julia", "--version"], stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    pass

log.info("=== VERSIONI ===")
log.info(f"  Python  : {sys.version.split()[0]}")
log.info(f"  numpy   : {np.__version__}")
log.info(f"  pandas  : {pd.__version__}")
log.info(f"  pysr    : {pysr.__version__}")
log.info(f"  sklearn : {sklearn.__version__}")
log.info(f"  Julia   : {_julia_ver}")

# Salva pip freeze per riproducibilità completa
try:
    _freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL
    ).decode()
    with open("requirements_run.txt", "w") as _f:
        _f.write(f"# Python {sys.version}\n# Julia {_julia_ver}\n\n{_freeze}")
    log.info("  requirements_run.txt salvato")
except Exception as e:
    log.warning(f"  pip freeze fallito: {e}")

# Sanity check spazio disco — avvisa se meno di 2GB liberi
try:
    _disk = shutil.disk_usage('.')
    _free_gb = _disk.free / 1e9
    if _free_gb < 2.0:
        log.warning(f"SPAZIO DISCO BASSO: {_free_gb:.1f} GB liberi — rischio crash durante il run")
    else:
        log.info(f"Spazio disco libero: {_free_gb:.1f} GB — OK")
except Exception as e:
    log.warning(f"Controllo spazio disco fallito: {e}")

# ============================================================
# CONFIGURAZIONE
# ============================================================
PILOT = False          # True per test veloce, False per run overnight
KEEP_TOP_N_DIRS = 1    # Tieni solo la cartella del seed migliore per gruppo

# Config per gruppo — sub_terr e rocky_SE: parsimony alta, maxsize basso
#                     sub_nep e gas: parsimony bassa, maxsize alto
_base = dict(
    binary_operators=["+", "-", "*", "/", "pow"],
    unary_operators=["log"],
    constraints={"pow": (-1, 1)},
    turbo=True,
    weight_optimize=0.001,
    verbosity=1,
)
GROUP_CONFIG = {
    'sub_terr': dict(**_base, niterations=100000, population_size=50, populations=24,
                     maxsize=13, parsimony=5e-3),
    'rocky_SE': dict(**_base, niterations=100000, population_size=50, populations=24,
                     maxsize=13, parsimony=5e-3),
    'sub_nep':  dict(**_base, niterations=100000, population_size=50, populations=24,
                     maxsize=17, parsimony=1e-3),
    'gas':      dict(**_base, niterations=100000, population_size=50, populations=24,
                     maxsize=17, parsimony=1e-3),
}
if PILOT:
    for g in GROUP_CONFIG:
        GROUP_CONFIG[g].update(niterations=1000, population_size=20, populations=15)

n_cpus = min(8, os.cpu_count() or 1)
log.info(f"PILOT={PILOT}  CPU usati={n_cpus}")
seeds = [0, 42, 123]

# ============================================================
# VERIFICA PARAMETRI VALIDI — logga esplicitamente se turbo/weight_optimize scartati
# ============================================================
_valid_params = set(inspect.signature(PySRRegressor.__init__).parameters.keys())
_IMPORTANT = {'turbo', 'weight_optimize', 'populations', 'constraints'}

def clean_config(cfg):
    cleaned = {k: v for k, v in cfg.items() if k in _valid_params}
    removed = set(cfg.keys()) - set(cleaned.keys())
    if removed:
        important_removed = removed & _IMPORTANT
        if important_removed:
            log.warning(f"ATTENZIONE — parametri importanti scartati (aggiorna PySR?): {important_removed}")
        other_removed = removed - _IMPORTANT
        if other_removed:
            log.warning(f"Parametri ignorati (non validi): {other_removed}")
    return cleaned

# Verifica all'avvio
log.info("Verifica parametri config all'avvio:")
for g, cfg in GROUP_CONFIG.items():
    cleaned = clean_config(cfg.copy())
    log.info(f"  {g}: maxsize={cleaned.get('maxsize')}  parsimony={cleaned.get('parsimony')}")

# ============================================================
# CHECKPOINT
# ============================================================
CHECKPOINT_FILE = 'checkpoint.json'
CHECKPOINTS_DIR = 'checkpoints_pysr'
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}

def save_checkpoint(ckpt):
    """Salvataggio atomico — scrive su .tmp poi rinomina."""
    tmp = CHECKPOINT_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(ckpt, f, indent=2)
    os.replace(tmp, CHECKPOINT_FILE)

def is_done(ckpt, group, seed):
    return ckpt.get(str(group), {}).get(str(seed), False)

def mark_done(ckpt, group, seed, result):
    if str(group) not in ckpt:
        ckpt[str(group)] = {}
    ckpt[str(group)][str(seed)] = result
    save_checkpoint(ckpt)

checkpoint = load_checkpoint()
if checkpoint:
    log.info("Checkpoint trovato — riprendo da dove mi ero fermato:")
    for g, seeds_done in checkpoint.items():
        log.info(f"  {g}: seed completati {list(seeds_done.keys())}")

# ============================================================
# GESTIONE SIGTERM / CTRL+C
# ============================================================
def handle_exit(signum, frame):
    log.warning("Segnale ricevuto — checkpoint salvato, uscita pulita.")
    save_checkpoint(checkpoint)
    raise SystemExit(0)

signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)

# ============================================================
# SALVATAGGIO ATOMICO
# ============================================================
def atomic_pickle(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)

def atomic_csv(df, path):
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

# ============================================================
# PULIZIA TEMPDIR — tieni solo i top-N seed per R²_test
# ============================================================
def cleanup_seed_dirs(group, risultati, keep_n=KEEP_TOP_N_DIRS):
    """Cancella le cartelle dei seed peggiori, tieni solo i top-N per R²_test."""
    validi = [r for r in risultati if not np.isnan(r['r2_te'])]
    if not validi:
        return
    validi_sorted = sorted(validi, key=lambda x: x['r2_te'], reverse=True)
    keep_seeds  = {str(r['seed']) for r in validi_sorted[:keep_n]}
    all_seeds   = {str(r['seed']) for r in risultati}
    delete_seeds = all_seeds - keep_seeds
    for s in delete_seeds:
        d = os.path.join(CHECKPOINTS_DIR, f"{group}_seed{s}")
        if os.path.exists(d):
            shutil.rmtree(d)
            log.info(f"  [cleanup] Rimossa cartella seed {s} ({group}) — non è top-{keep_n}")
    if keep_seeds:
        log.info(f"  [cleanup] Mantenute cartelle seed {keep_seeds} ({group})")

# ============================================================
# SAFE EVAL
# ============================================================
def safe_eval_formula(model, X_sample):
    try:
        y = model.predict(X_sample)
        return bool(np.isfinite(y).all())
    except Exception:
        return False

# K-Fold rimosso: causerebbe data leakage (X include il test set) e non
# ri-ottimizza le costanti su ogni fold — il Train/Test split 80/20 è la metrica corretta.

# ============================================================
# PLOT osservato vs predetto + residui — per paper
# ============================================================
def make_plots(model, X_tr, y_tr, X_te, y_te, g, seed):
    """
    Produce due grafici per ogni gruppo/seed:
      1. Rp_obs vs Rp_pred (spazio fisico, in R_terra)
      2. Residui vs Rp_pred
    Le predizioni sono in log(Rp), quindi esponiamo con exp().
    """
    try:
        y_pred_tr = model.predict(X_tr)
        y_pred_te = model.predict(X_te)
        if np.any(np.isnan(y_pred_tr)) or np.any(np.isnan(y_pred_te)):
            log.warning(f"  Plot saltato: NaN nelle predizioni")
            return

        # Clip in log-space prima di exp() — evita overflow tipo exp(100) = 2.6e43
        # log(25 R_terra) ≈ 3.2, quindi [-5, 5] copre tutto il range fisico
        y_pred_tr = np.clip(y_pred_tr, -5, 5)
        y_pred_te = np.clip(y_pred_te, -5, 5)

        # Torna in spazio fisico R_terra
        Rp_obs_tr  = np.exp(y_tr)
        Rp_pred_tr = np.exp(y_pred_tr)
        Rp_obs_te  = np.exp(y_te)
        Rp_pred_te = np.exp(y_pred_te)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.suptitle(f'{g}  —  seed {seed}', fontsize=13)

        # --- Plot 1: osservato vs predetto ---
        ax = axes[0]
        ax.scatter(Rp_pred_tr, Rp_obs_tr, s=18, alpha=0.55,
                   color='steelblue', label='Train', zorder=2)
        ax.scatter(Rp_pred_te, Rp_obs_te, s=28, alpha=0.85,
                   color='tomato', label='Test', zorder=3)
        lims = [min(Rp_obs_tr.min(), Rp_pred_tr.min()) * 0.9,
                max(Rp_obs_tr.max(), Rp_pred_tr.max()) * 1.1]
        ax.plot(lims, lims, 'k--', lw=1, label='1:1')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('$R_p$ predetto  [$R_\\oplus$]')
        ax.set_ylabel('$R_p$ osservato  [$R_\\oplus$]')
        ax.set_title('Osservato vs Predetto')
        ax.legend(fontsize=8)

        # --- Plot 2: residui ---
        ax = axes[1]
        res_tr = Rp_obs_tr - Rp_pred_tr
        res_te = Rp_obs_te - Rp_pred_te
        ax.scatter(Rp_pred_tr, res_tr, s=18, alpha=0.55,
                   color='steelblue', label='Train', zorder=2)
        ax.scatter(Rp_pred_te, res_te, s=28, alpha=0.85,
                   color='tomato', label='Test', zorder=3)
        ax.axhline(0, color='k', lw=1, ls='--')
        ax.set_xlabel('$R_p$ predetto  [$R_\\oplus$]')
        ax.set_ylabel('Residuo  ($R_{obs} - R_{pred}$)  [$R_\\oplus$]')
        ax.set_title('Residui')
        ax.legend(fontsize=8)

        plt.tight_layout()
        os.makedirs('plots', exist_ok=True)
        fname = os.path.join('plots', f'plot_{g}_seed{seed}.png')
        plt.savefig(fname, dpi=150)
        plt.close(fig)
        log.info(f"  Plot salvato: {fname}")

        # CSV con Rp_obs, Rp_pred, residuo — per analisi e figure paper
        df_res = pd.DataFrame({
            'split':    ['train']*len(Rp_obs_tr) + ['test']*len(Rp_obs_te),
            'Rp_obs':   np.concatenate([Rp_obs_tr,  Rp_obs_te]),
            'Rp_pred':  np.concatenate([Rp_pred_tr, Rp_pred_te]),
            'residuo':  np.concatenate([res_tr,      res_te]),
        })
        csv_fname = os.path.join('plots', f'residui_{g}_seed{seed}.csv')
        atomic_csv(df_res, csv_fname)
        log.info(f"  Residui CSV salvato: {csv_fname}")
    except Exception as e:
        log.warning(f"  Plot fallito: {e}")

# ============================================================
# STEP 1 - SCARICA DATI NASA
# ============================================================
log.info("Scaricando dati NASA...")
_r = requests.get(
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
    params={
        "query": (
            "select pl_masse,pl_masseerr1,pl_masseerr2,pl_rade,pl_radeerr1,"
            "pl_insol,st_met,st_mass,st_rad,pl_orbper "
            "from pscomppars"
        ),
        "format": "csv"
    },
    timeout=300
)
if _r.status_code != 200:
    log.error(f"Errore NASA: {_r.status_code}  {_r.text[:500]}")
    raise SystemExit(1)
df = pd.read_csv(io.StringIO(_r.text))
df = df.dropna(subset=['pl_masse','pl_masseerr1','pl_masseerr2',
                        'pl_rade','pl_radeerr1','pl_insol',
                        'st_met','st_mass','st_rad','pl_orbper'])
df = df[
    (df['pl_masse'] > 0) & (df['pl_rade'] > 0) &
    (df['pl_insol'] > 0) & (df['st_rad'] > 0) &
    (df['pl_orbper'] > 0) & (df['pl_masse'] < 4000) &
    (df['pl_rade'] < 25)
]
log.info(f"Pianeti totali prima dei filtri: {len(df)}")

# ============================================================
# STEP 2 - FILTRO ERRORI ASIMMETRICO
# ============================================================
df['err_rel_pos'] = np.abs(df['pl_masseerr1']) / df['pl_masse']
df['err_rel_neg'] = np.abs(df['pl_masseerr2']) / df['pl_masse']
df['err_rel_max'] = df[['err_rel_pos', 'err_rel_neg']].max(axis=1)

bins   = [0, 1.5, 2.0, 4.0, 100]
labels = ['sub_terr', 'rocky_SE', 'sub_nep', 'gas']
df['group'] = pd.cut(df['pl_rade'], bins=bins, labels=labels)

piccoli = df['group'].isin(['sub_terr', 'rocky_SE'])
df_filt = pd.concat([
    df[piccoli  & (df['err_rel_max'] < 0.35)],
    df[~piccoli & (df['err_rel_max'] < 0.30)]
]).copy()

log.info(f"Pianeti dopo filtro errori: {len(df_filt)}")
atomic_csv(df_filt, 'exoplanets_filtrati.csv')
log.info("Pianeti per gruppo DOPO tutti i filtri:")
for g, n in df_filt.groupby('group', observed=True).size().items():
    log.info(f"  {g}: {n}")

# ============================================================
# STEP 3 - LOG-TRASFORMAZIONE + FEATURE ENGINEERING
# Nota naming: colonna pandas = 'log_rhostar', nome in equazioni PySR = 'logRhoStar'
# ============================================================
df_filt['logMp']       = np.log(df_filt['pl_masse'])
df_filt['logF']        = np.log(df_filt['pl_insol'])
df_filt['FeH']         = df_filt['st_met']
df_filt['logMs']       = np.log(df_filt['st_mass'])
df_filt['logRs']       = np.log(df_filt['st_rad'])
df_filt['logP']        = np.log(df_filt['pl_orbper'])
df_filt['logRp']       = np.log(df_filt['pl_rade'])
df_filt['log_rhostar'] = df_filt['logMs'] - 3*df_filt['logRs']  # colonna interna

df_filt['err_rade_rel'] = np.abs(df_filt['pl_radeerr1']) / df_filt['pl_rade']
df_filt['sigma_tot']    = np.sqrt(df_filt['err_rel_max']**2 + df_filt['err_rade_rel']**2)
df_filt['weight']       = 1.0 / (df_filt['sigma_tot']**2 + 1e-6)
# Normalizzazione locale spostata in prep() — ogni gruppo normalizza i propri pesi

# VAR_NAMES: ordine deve corrispondere all'ordine di colonne in prep()
# logMp, logF, FeH, logP, logRhoStar  (logRhoStar = log_rhostar in pandas)
VAR_NAMES  = ["logMp", "logF", "FeH", "logP", "logRhoStar"]
cols_input = ['logMp', 'logF', 'FeH', 'logP', 'log_rhostar']

log.info("Matrice di correlazione variabili input:")
corr = df_filt[cols_input].corr().round(2)
log.info(f"\n{corr.to_string()}")
trovate = False
for i in range(len(cols_input)):
    for j in range(i+1, len(cols_input)):
        r = corr.iloc[i, j]
        if abs(r) > 0.85:
            log.warning(f"Correlazione alta: {cols_input[i]} vs {cols_input[j]}: r={r}")
            trovate = True
if not trovate:
    log.info("Nessuna coppia con |r| > 0.85 — ottimo!")

# ============================================================
# STEP 4 - ESPONENTE BASELINE
# ============================================================
log.info("Esponente baseline massa-raggio:")
for g in labels:
    sub = df_filt[df_filt['group'] == g]
    if len(sub) < 20:
        continue
    X = sub['logMp'].values.reshape(-1, 1)
    y = sub['logRp'].values
    lr = LinearRegression().fit(X, y)
    log.info(f"  {g} (N={len(sub)}): R = M^{lr.coef_[0]:.3f}  R²={lr.score(X,y):.3f}")

# ============================================================
# STEP 5 - PREPROCESSING
# Ordine colonne = ordine VAR_NAMES: logMp, logF, FeH, logP, logRhoStar
# ============================================================
def prep(sub):
    X = np.column_stack([
        sub['logMp'].values,        # x0 → logMp
        sub['logF'].values,         # x1 → logF
        sub['FeH'].values,          # x2 → FeH
        sub['logP'].values,         # x3 → logP
        sub['log_rhostar'].values,  # x4 → logRhoStar
    ])
    y = sub['logRp'].values
    w = sub['weight'].values
    w = np.maximum(w, 1e-6)      # floor: evita pesi zero o negativi
    w = w / w.mean()             # normalizzazione LOCALE al gruppo — media = 1
    return X, y, w

# ============================================================
# STEP 6 - RUN PySR CON 3 SEMI + CHECKPOINT + PICKLE + PLOT
# ============================================================
for g in labels:
    sub = df_filt[df_filt['group'] == g].copy()
    if len(sub) < 50:
        log.info(f"Gruppo {g} troppo piccolo ({len(sub)}) - skip")
        continue

    log.info(f"{'='*60}")
    log.info(f"Gruppo {g} ({len(sub)} pianeti)")
    log.info(f"{'='*60}")

    X, y, w = prep(sub)
    idx = np.arange(len(X))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=1)
    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    w_tr = w[idx_tr]
    log.info(f"Train: {len(X_tr)}  Test: {len(X_te)}")

    risultati = []
    for seed in seeds:
        if is_done(checkpoint, g, seed):
            r = checkpoint[str(g)][str(seed)]
            if isinstance(r, dict) and 'formula' in r:
                log.info(f"Seed {seed} [{g}] già completato — skip")
                log.info(f"  R²_train={r['r2_tr']:.4f}  R²_test={r['r2_te']:.4f}  {r['formula']}")
                risultati.append(r)
                continue

        log.info(f"--- Seed {seed} [{g}] AVVIO ---")
        t0 = time.time()

        seed_tempdir = os.path.join(CHECKPOINTS_DIR, f"{g}_seed{seed}", "pysr_tmp")
        os.makedirs(seed_tempdir, exist_ok=True)
        ckpt_dir = os.path.join(CHECKPOINTS_DIR, f"{g}_seed{seed}")
        os.makedirs(ckpt_dir, exist_ok=True)

        run_config = clean_config(GROUP_CONFIG[g].copy())
        run_config['random_state'] = seed
        run_config['procs'] = n_cpus
        run_config['tempdir'] = seed_tempdir
        run_config['temp_equation_file'] = True

        model = None
        best_eq = "n/d"
        r2_tr = float('nan')
        r2_te = float('nan')
        mae_te = float('nan')

        try:
            model = PySRRegressor(**run_config)
            model.fit(X_tr, y_tr, weights=w_tr, variable_names=VAR_NAMES)

            elapsed = time.time() - t0
            log.info(f"Seed {seed} [{g}] completato in {elapsed/60:.1f} min")

            # Snapshot dati per riproducibilità
            np.savez(
                os.path.join(ckpt_dir, 'data_snapshot.npz'),
                X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
                w_tr=w_tr, idx_tr=idx_tr, idx_te=idx_te
            )
            atomic_pickle(run_config, os.path.join(ckpt_dir, 'run_config.pkl'))

            # Salvataggio atomico top10 e pickle
            atomic_csv(model.equations_.head(10), os.path.join(ckpt_dir, 'top10.csv'))
            atomic_csv(model.equations_.head(10), f'eq_{g}_seed{seed}_top10.csv')
            atomic_pickle(model, os.path.join(ckpt_dir, 'model.pkl'))

            # Copia hall_of_fame.csv — compatibile con PySR >= 1.5.9
            try:
                # Nuova API: output_directory_ + run_id_
                hof_src = os.path.join(
                    str(model.output_directory_), str(model.run_id_), 'hall_of_fame.csv'
                )
                if os.path.exists(hof_src):
                    shutil.copy(hof_src, os.path.join(ckpt_dir, 'hall_of_fame.csv'))
                    log.info(f"  hall_of_fame.csv copiato")
                else:
                    # Fallback: vecchia API
                    hof_src2 = getattr(model, 'equation_file_', None)
                    if hof_src2 and os.path.exists(str(hof_src2)):
                        shutil.copy(str(hof_src2), os.path.join(ckpt_dir, 'hall_of_fame.csv'))
                        log.info(f"  hall_of_fame.csv copiato (fallback vecchia API)")
                    else:
                        log.warning(f"  hall_of_fame.csv non trovato in {hof_src}")
            except Exception as e:
                log.warning(f"  Copia hall_of_fame fallita: {e}")

            # Controllo dominio
            if not safe_eval_formula(model, X_tr):
                log.warning("  Formula instabile su X_tr (NaN/inf) — trattare con cautela")

            # Formula migliore
            try:
                best_eq = str(model.sympy())
            except Exception as e:
                best_eq = f"sympy_error: {e}"

            # R² con protezione NaN
            try:
                y_pred_tr = model.predict(X_tr)
                y_pred_te = model.predict(X_te)
                if np.any(np.isnan(y_pred_tr)) or np.any(np.isnan(y_pred_te)):
                    raise ValueError("NaN nelle predizioni")
                r2_tr  = float(1 - np.sum((y_tr-y_pred_tr)**2)/np.sum((y_tr-y_tr.mean())**2))
                r2_te  = float(1 - np.sum((y_te-y_pred_te)**2)/np.sum((y_te-y_te.mean())**2))
                mae_te = float(np.mean(np.abs(y_te - y_pred_te)))
            except Exception as e:
                log.warning(f"  Errore in predict: {e}")

            # Plot osservato vs predetto + residui
            make_plots(model, X_tr, y_tr, X_te, y_te, g, seed)

        except Exception as e:
            elapsed = time.time() - t0
            log.error(f"Seed {seed} [{g}] FALLITO dopo {elapsed/60:.1f} min: {e}")
            mark_done(checkpoint, g, seed, {
                'seed': seed, 'r2_tr': float('nan'), 'r2_te': float('nan'),
                'mae_te': float('nan'), 'formula': f'ERROR: {e}'
            })
            continue

        log.info(f"  R²_train={r2_tr:.4f}  R²_test={r2_te:.4f}  MAE_test={mae_te:.4f}")
        log.info(f"  Formula (log space): {best_eq}")
        log.info(f"  Formula (fisico):    Rp [R_terra] = exp({best_eq})")

        with open(f'best_{g}_seed{seed}.txt', 'w') as f:
            f.write(f"log space:   logRp = {best_eq}\n")
            f.write(f"fisico:      Rp [R_terra] = exp({best_eq})\n")

        result = {
            'seed': seed, 'r2_tr': r2_tr, 'r2_te': r2_te,
            'mae_te': mae_te, 'formula': best_eq,
            # metadati per autoesplicabilità
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_train': int(len(X_tr)),
            'n_test':  int(len(X_te)),
            'elapsed_min': round((time.time() - t0) / 60, 1),
            'config_clean': {k: str(v) for k, v in run_config.items()
                             if k not in ('procs', 'tempdir', 'temp_equation_file')},
        }
        risultati.append(result)
        mark_done(checkpoint, g, seed, result)

    log.info(f"--- Stability {g} ---")
    for r in risultati:
        log.info(f"  Seed {r['seed']}: R²_train={r['r2_tr']:.4f}  R²_test={r['r2_te']:.4f}  {r['formula']}")

    # results_summary.csv — include baseline power-law per confronto
    if risultati:
        # Baseline: regressione lineare logMp → logRp (power law R ∝ M^α)
        X_base = X_tr[:, 0:1]  # solo logMp
        lr_base = LinearRegression().fit(X_base, y_tr)
        y_pred_base_te = lr_base.predict(X_te[:, 0:1])
        r2_base = float(1 - np.sum((y_te - y_pred_base_te)**2) /
                        np.sum((y_te - y_te.mean())**2))
        alpha_base = float(lr_base.coef_[0])
        log.info(f"  Baseline power-law: R = M^{alpha_base:.3f}  R²_test={r2_base:.4f}")

        freq = {}
        for r in risultati:
            f = r['formula']
            if f not in freq:
                freq[f] = {'formula': f, 'count': 0, 'r2_te_list': []}
            freq[f]['count'] += 1
            if not np.isnan(r['r2_te']):
                freq[f]['r2_te_list'].append(r['r2_te'])
        summary_rows = []
        # Riga baseline
        summary_rows.append({
            'group': g, 'formula': f'BASELINE: logRp = {alpha_base:.3f}*logMp + C',
            'freq': 0, 'mean_r2_te': r2_base, 'std_r2_te': float('nan'),
            'delta_r2_vs_baseline': 0.0,
        })
        for f, v in sorted(freq.items(), key=lambda x: -x[1]['count']):
            r2_list = v['r2_te_list']
            mean_r2 = float(np.mean(r2_list)) if r2_list else float('nan')
            summary_rows.append({
                'group': g, 'formula': f, 'freq': v['count'],
                'mean_r2_te': mean_r2,
                'std_r2_te':  float(np.std(r2_list)) if len(r2_list) > 1 else float('nan'),
                'delta_r2_vs_baseline': mean_r2 - r2_base if not np.isnan(mean_r2) else float('nan'),
            })
        atomic_csv(pd.DataFrame(summary_rows), f'results_summary_{g}.csv')
        log.info(f"  results_summary_{g}.csv salvato (con baseline)")

    # Pulizia: tieni solo la cartella del seed migliore
    cleanup_seed_dirs(g, risultati, keep_n=KEEP_TOP_N_DIRS)

# ============================================================
# STEP 7 - PERMUTATION TEST
# ============================================================
if not is_done(checkpoint, 'permutation', 0):
    log.info("=== PERMUTATION TEST ===")
    g_test = 'sub_nep'
    sub_test = df_filt[df_filt['group'] == g_test].copy()

    if len(sub_test) >= 50:
        X_t, y_t, w_t = prep(sub_test)
        idx = np.arange(len(X_t))
        idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=1)
        X_tr_p, X_te_p = X_t[idx_tr], X_t[idx_te]
        y_tr_p, y_te_p = y_t[idx_tr], y_t[idx_te]
        w_tr_p = w_t[idx_tr]

        perm_config = clean_config(GROUP_CONFIG[g_test].copy())
        perm_config.update({
            'niterations': 1000 if PILOT else 5000,
            'verbosity': 0,
            'population_size': 30,
            'populations': 10,
            'procs': n_cpus,
            'temp_equation_file': True,
        })

        r2_perm_list = []
        log.info("5 permutazioni del target...")
        for i in range(5):
            perm_tempdir = os.path.join(CHECKPOINTS_DIR, f"permutation_{i}", "pysr_tmp")
            os.makedirs(perm_tempdir, exist_ok=True)
            ytr_shuff = np.random.permutation(y_tr_p)
            perm_config['random_state'] = i
            perm_config['tempdir'] = perm_tempdir
            try:
                model_p = PySRRegressor(**perm_config)
                model_p.fit(X_tr_p, ytr_shuff, weights=w_tr_p,
                            variable_names=VAR_NAMES)
                y_pred_p = model_p.predict(X_te_p)
                if np.any(np.isnan(y_pred_p)):
                    raise ValueError("NaN")
                r2_p = float(1 - np.sum((y_te_p-y_pred_p)**2)/np.sum((y_te_p-y_te_p.mean())**2))
            except Exception as e:
                log.warning(f"Permutazione {i+1}: ERRORE ({e})")
                r2_p = float('nan')
            r2_perm_list.append(r2_p)
            log.info(f"  Permutazione {i+1}: R²_test={r2_p:.4f}")

        valid = [x for x in r2_perm_list if not np.isnan(x)]
        mean_perm = float(np.mean(valid)) if valid else float('nan')
        log.info(f"R²_test medio permutato: {mean_perm:.4f} (deve essere vicino a 0)")
        mark_done(checkpoint, 'permutation', 0, {'r2_perm_mean': mean_perm})
else:
    log.info("=== PERMUTATION TEST [già completato] ===")
    log.info(f"  R²_test medio permutato: {checkpoint['permutation']['0']['r2_perm_mean']:.4f}")

log.info("=== COMPLETATO ===")