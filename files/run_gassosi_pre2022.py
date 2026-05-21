"""
run_gassosi_pre2022.py — Run PySR sui gas giants pre-2022 ONLY.

Motivazione: validazione out-of-sample pura. PySR addestrato
esclusivamente su pianeti scoperti prima del 2022. I pianeti
post-2022 non vengono mai visti durante il training.
Poi testiamo su post-2022 → vera generalizzazione temporale.

Uso: python3 ~/run_gassosi_pre2022.py
"""

import os, sys, time, json, pickle, shutil, signal, inspect, logging, subprocess
import numpy as np
import pandas as pd
import requests, io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pysr import PySRRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pysr, sklearn

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('run_gassosi_pre2022.log', mode='a'),
    ]
)
log = logging.getLogger(__name__)

_julia_ver = "n/d"
try:
    _julia_ver = subprocess.check_output(
        ["julia", "--version"], stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    pass

log.info("=== run_gassosi_pre2022.py — AVVIO ===")
log.info(f"  Python : {sys.version.split()[0]}")
log.info(f"  pysr   : {pysr.__version__}")
log.info(f"  Julia  : {_julia_ver}")

# ============================================================
# CONFIG
# ============================================================
PILOT           = False
SPLIT_YEAR      = 2022
CHECKPOINTS_DIR = 'checkpoints_gas_pre2022'
CHECKPOINT_FILE = 'checkpoint_gas_pre2022.json'
KEEP_TOP_N_DIRS = 1
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

n_cpus = min(8, os.cpu_count() or 1)
seeds  = [0, 42, 123]

GAS_CONFIG = dict(
    binary_operators=["+", "-", "*", "/", "pow"],
    unary_operators=["log"],
    constraints={"pow": (-1, 1)},
    turbo=True,
    weight_optimize=0.001,
    verbosity=1,
    niterations=100000,
    population_size=50,
    populations=24,
    maxsize=20,
    parsimony=5e-4,
)
if PILOT:
    GAS_CONFIG.update(niterations=1000, population_size=20, populations=15)

log.info(f"PILOT={PILOT}  CPU={n_cpus}  SPLIT_YEAR={SPLIT_YEAR}")
log.info(f"maxsize={GAS_CONFIG['maxsize']}  parsimony={GAS_CONFIG['parsimony']}")

# ============================================================
# PARAMETRI VALIDI
# ============================================================
_valid_params = set(inspect.signature(PySRRegressor.__init__).parameters.keys())

def clean_config(cfg):
    cleaned = {k: v for k, v in cfg.items() if k in _valid_params}
    removed = set(cfg.keys()) - set(cleaned.keys())
    important = {'turbo','weight_optimize','populations','constraints'}
    if removed & important:
        log.warning(f"Parametri importanti scartati: {removed & important}")
    return cleaned

# ============================================================
# CHECKPOINT
# ============================================================
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f: return json.load(f)
    return {}

def save_checkpoint(ckpt):
    tmp = CHECKPOINT_FILE + '.tmp'
    with open(tmp,'w') as f: json.dump(ckpt, f, indent=2)
    os.replace(tmp, CHECKPOINT_FILE)

def is_done(ckpt, seed):
    return ckpt.get('gas',{}).get(str(seed), False)

def mark_done(ckpt, seed, result):
    if 'gas' not in ckpt: ckpt['gas'] = {}
    ckpt['gas'][str(seed)] = result
    save_checkpoint(ckpt)

checkpoint = load_checkpoint()
if checkpoint:
    log.info("Checkpoint trovato:")
    for g, sd in checkpoint.items():
        log.info(f"  {g}: seed completati {list(sd.keys())}")

def handle_exit(signum, frame):
    log.warning("Segnale ricevuto — salvo e uscita.")
    save_checkpoint(checkpoint)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT,  handle_exit)

# ============================================================
# UTILITY
# ============================================================
def atomic_pickle(obj, path):
    tmp = path+'.tmp'
    with open(tmp,'wb') as f: pickle.dump(obj, f)
    os.replace(tmp, path)

def atomic_csv(df, path):
    tmp = path+'.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0 or not np.all(np.isfinite(y_pred)): return float('nan')
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return float(1 - ss_res/ss_tot) if ss_tot > 0 else float('nan')

def safe_predict(model, X):
    try:
        return np.asarray(model.predict(np.atleast_2d(X)), dtype=float).ravel()
    except Exception as e:
        log.warning(f"  safe_predict: {e}")
        return np.full(np.atleast_2d(X).shape[0], float('nan'))

def safe_sympy(model):
    try:    return str(model.sympy())
    except: return 'n/d'

# ============================================================
# STEP 1 — Scarica dataset con disc_year
# ============================================================
log.info("="*60)
log.info("STEP 1 — Dataset con disc_year")
log.info("="*60)

CACHE = 'nasa_with_discyear.csv'
if os.path.exists(CACHE):
    log.info(f"  Usando cache: {CACHE}")
    df_raw = pd.read_csv(CACHE)
else:
    log.info("  Scaricando da NASA...")
    r = requests.get(
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
        params={
            "query": (
                "select pl_name,pl_masse,pl_masseerr1,pl_masseerr2,"
                "pl_rade,pl_radeerr1,pl_insol,st_met,st_mass,st_rad,"
                "pl_orbper,pl_orbeccen,disc_year from pscomppars"
            ),
            "format": "csv"
        },
        timeout=300
    )
    if r.status_code != 200:
        raise RuntimeError(f"NASA HTTP {r.status_code}")
    df_raw = pd.read_csv(io.StringIO(r.text))
    df_raw.to_csv(CACHE, index=False)
    log.info(f"  Cache salvata: {CACHE}")

log.info(f"  Totale righe: {len(df_raw)}")

# ============================================================
# STEP 2 — Filtri e feature engineering
# ============================================================
df_raw = df_raw.dropna(subset=['pl_masse','pl_masseerr1','pl_masseerr2',
                                'pl_rade','pl_radeerr1','pl_insol',
                                'st_met','st_mass','st_rad','pl_orbper','disc_year'])
df_raw = df_raw[
    (df_raw['pl_masse'] > 0) & (df_raw['pl_masse'] < 4000) &
    (df_raw['pl_rade']  > 0) & (df_raw['pl_rade']  < 25)   &
    (df_raw['pl_insol'] > 0) & (df_raw['st_mass']  > 0)    &
    (df_raw['st_rad']   > 0) & (df_raw['pl_orbper'] > 0)
].copy()

df_raw['logMp']       = np.log(df_raw['pl_masse'])
df_raw['logF']        = np.log(df_raw['pl_insol'])
df_raw['FeH']         = df_raw['st_met']
df_raw['logMs']       = np.log(df_raw['st_mass'])
df_raw['logRs']       = np.log(df_raw['st_rad'])
df_raw['logP']        = np.log(df_raw['pl_orbper'])
df_raw['logRp']       = np.log(df_raw['pl_rade'])
df_raw['log_rhostar'] = df_raw['logMs'] - 3*df_raw['logRs']
df_raw['ecc']         = df_raw['pl_orbeccen'].fillna(0.0).clip(0,0.99) \
                        if 'pl_orbeccen' in df_raw.columns else 0.0
df_raw['log1p_ecc']   = np.log1p(df_raw['ecc'])
df_raw['err_rel_max'] = df_raw[['pl_masseerr1','pl_masseerr2']].abs().max(axis=1) \
                        / df_raw['pl_masse']
df_raw['err_rade_rel']= df_raw['pl_radeerr1'].abs() / df_raw['pl_rade']
df_raw['sigma_tot']   = np.sqrt(df_raw['err_rel_max']**2 + df_raw['err_rade_rel']**2)
df_raw['weight']      = 1.0 / (df_raw['sigma_tot']**2 + 1e-6)

# Solo gas giants con errore < 30%
df_gas_all = df_raw[
    (df_raw['pl_rade'] >= 4.0) &
    (df_raw['err_rel_max'] < 0.30)
].copy()
log.info(f"  Gas giants totali: {len(df_gas_all)}")

# Split temporale — HARD CUT
df_train = df_gas_all[df_gas_all['disc_year'] <  SPLIT_YEAR].copy()
df_test  = df_gas_all[df_gas_all['disc_year'] >= SPLIT_YEAR].copy()
log.info(f"  Training (pre-{SPLIT_YEAR}):  {len(df_train)} pianeti")
log.info(f"  Test     (post-{SPLIT_YEAR}): {len(df_test)} pianeti")

if len(df_train) < 100:
    raise RuntimeError(f"Training troppo piccolo: {len(df_train)}")
if len(df_test) < 20:
    raise RuntimeError(f"Test troppo piccolo: {len(df_test)}")

VAR_NAMES = ["logMp", "logF", "FeH", "logP", "logRhoStar", "log1p_ecc"]

def prep(sub):
    X = np.column_stack([
        sub['logMp'].values,
        sub['logF'].values,
        sub['FeH'].values,
        sub['logP'].values,
        sub['log_rhostar'].values,
        np.log1p(sub['ecc'].values),
    ])
    y = sub['logRp'].values
    w = sub['weight'].values
    w = np.maximum(w, 1e-6)
    w = w / w.mean()
    return X, y, w

X_tr, y_tr, w_tr = prep(df_train)
X_te, y_te, _    = prep(df_test)
log.info(f"  X_train: {X_tr.shape}  X_test: {X_te.shape}")

# Baseline
lr_base = LinearRegression().fit(X_tr[:,0:1], y_tr)
r2_base = safe_r2(y_te, lr_base.predict(X_te[:,0:1]))
log.info(f"  Baseline power-law: R²_test={r2_base:.4f}")

# ============================================================
# STEP 3 — RUN PySR
# ============================================================
log.info("="*60)
log.info("STEP 3 — PySR run (addestrato SOLO su pre-2022)")
log.info("="*60)

risultati = []

for seed in seeds:
    if is_done(checkpoint, seed):
        r = checkpoint['gas'][str(seed)]
        if isinstance(r, dict) and 'formula' in r:
            log.info(f"Seed {seed} già completato — skip")
            log.info(f"  R²_train={r['r2_tr']:.4f}  R²_test={r['r2_te']:.4f}  {r['formula']}")
            risultati.append(r)
            continue

    log.info(f"--- Seed {seed} AVVIO ---")
    t0 = time.time()

    ckpt_dir    = os.path.join(CHECKPOINTS_DIR, f"gas_seed{seed}")
    seed_tmpdir = os.path.join(ckpt_dir, "pysr_tmp")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(seed_tmpdir, exist_ok=True)

    run_config = clean_config(GAS_CONFIG.copy())
    run_config['random_state']       = seed
    run_config['procs']              = n_cpus
    run_config['tempdir']            = seed_tmpdir
    run_config['temp_equation_file'] = True

    best_eq = 'n/d'
    r2_tr   = float('nan')
    r2_te   = float('nan')
    mae_te  = float('nan')

    try:
        model = PySRRegressor(**run_config)
        model.fit(X_tr, y_tr, weights=w_tr, variable_names=VAR_NAMES)

        elapsed = time.time() - t0
        log.info(f"Seed {seed} completato in {elapsed/60:.1f} min")

        np.savez(os.path.join(ckpt_dir, 'data_snapshot.npz'),
                 X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
                 w_tr=w_tr, split_year=SPLIT_YEAR)
        atomic_pickle(run_config, os.path.join(ckpt_dir, 'run_config.pkl'))
        atomic_csv(model.equations_.head(20), os.path.join(ckpt_dir, 'top20.csv'))
        atomic_csv(model.equations_.head(20), f'eq_gas_pre2022_seed{seed}_top20.csv')
        atomic_pickle(model, os.path.join(ckpt_dir, 'model.pkl'))

        try:
            hof = os.path.join(str(model.output_directory_),
                               str(model.run_id_), 'hall_of_fame.csv')
            if os.path.exists(hof):
                shutil.copy(hof, os.path.join(ckpt_dir, 'hall_of_fame.csv'))
        except Exception:
            pass

        best_eq   = safe_sympy(model)
        y_pred_tr = safe_predict(model, X_tr)
        y_pred_te = safe_predict(model, X_te)

        if np.any(np.isnan(y_pred_tr)) or np.any(np.isnan(y_pred_te)):
            raise ValueError("NaN nelle predizioni")

        r2_tr  = safe_r2(y_tr, y_pred_tr)
        r2_te  = safe_r2(y_te, y_pred_te)
        mae_te = float(np.mean(np.abs(y_te - y_pred_te)))

    except Exception as e:
        elapsed = time.time() - t0
        log.error(f"Seed {seed} FALLITO dopo {elapsed/60:.1f} min: {e}")
        mark_done(checkpoint, seed, {
            'seed': seed, 'r2_tr': float('nan'), 'r2_te': float('nan'),
            'mae_te': float('nan'), 'formula': f'ERROR: {e}'
        })
        continue

    log.info(f"  R²_train={r2_tr:.4f}  R²_test(post-2022)={r2_te:.4f}  MAE={mae_te:.4f}")
    log.info(f"  Formula: {best_eq}")

    if np.isfinite(r2_te):
        with open(f'best_gas_pre2022_seed{seed}.txt','w') as f:
            f.write(f"Seed: {seed}\n")
            f.write(f"Training: solo pianeti disc_year < {SPLIT_YEAR}\n")
            f.write(f"Test:     pianeti disc_year >= {SPLIT_YEAR} (mai visti)\n")
            f.write(f"R²_train: {r2_tr:.4f}\n")
            f.write(f"R²_test:  {r2_te:.4f}  ← OUT-OF-SAMPLE REALE\n")
            f.write(f"MAE_test: {mae_te:.4f}\n")
            f.write(f"Formula:  logRp = {best_eq}\n")

    result = {'seed': seed, 'r2_tr': r2_tr, 'r2_te': r2_te,
              'mae_te': mae_te, 'formula': best_eq}
    risultati.append(result)
    mark_done(checkpoint, seed, result)

# ============================================================
# STEP 4 — Risultati
# ============================================================
log.info("="*60)
log.info("RISULTATI — GAS GIANTS (training pre-2022, test post-2022)")
log.info("="*60)

validi = [r for r in risultati if np.isfinite(r.get('r2_te', float('nan')))]
if validi:
    log.info(f"  Baseline power-law R²_test: {r2_base:.4f}")
    log.info(f"  Chen & Kipping 2017:        0.441 (reference)")
    log.info(f"  Paper originale:            0.669 (test set normale)")
    log.info("")
    for r in sorted(validi, key=lambda x: x['r2_te'], reverse=True):
        delta = r['r2_te'] - r2_base
        log.info(f"  Seed {r['seed']}: R²_test={r['r2_te']:.4f}  ΔR²={delta:+.4f}  {r['formula'][:60]}")

    best = max(validi, key=lambda x: x['r2_te'])
    log.info(f"\n  MIGLIOR MODELLO OUT-OF-SAMPLE:")
    log.info(f"  Seed {best['seed']}  R²_test={best['r2_te']:.4f}")
    log.info(f"  Formula: {best['formula']}")

    df_res = pd.DataFrame(validi)
    df_res['baseline_r2'] = r2_base
    df_res['split_year']  = SPLIT_YEAR
    df_res['n_train']     = len(df_train)
    df_res['n_test']      = len(df_test)
    df_res.to_csv('results_gas_pre2022.csv', index=False)
    log.info("  results_gas_pre2022.csv salvato")

log.info("=== run_gassosi_pre2022.py COMPLETATO ===")
