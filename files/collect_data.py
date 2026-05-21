"""
collect_data.py — Dataset unificato per symbolic regression su esopianeti.

Strategia (da paper MNRAS):
  - NASA pscomppars come BACKBONE: nessun pianeta viene aggiunto
    da altri cataloghi. Solo i parametri vengono migliorati.
  - TEPCat come supplemento per il RAGGIO: se ha errore minore
    di NASA per quel pianeta, sostituiamo solo pl_rade e pl_radeerr.
  - PlanetS come supplemento per la METALLICITA': se disponibile
    e NASA ha NaN, usiamo il valore PlanetS.
  - Il flusso stellare (pl_insol) viene preso SOLO da NASA.
    Non viene mai ricostruito: rischio di incoerenza sistematica.
  - Tracciamento completo della provenienza per ogni parametro.
  - Flag di qualità alta (HQ): errori massa<20%, raggio<10%.
  - Gruppo fuso rocky_combined = sub_terr + rocky_SE per training
    PySR (poi validato separatamente).

Output: exoplanets_merged.csv
"""

import requests, io, os, hashlib, datetime, logging
import numpy as np
import pandas as pd

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(),
              logging.FileHandler('collect_data.log', mode='w')]
)
log = logging.getLogger(__name__)

TIMEOUT  = 300
OUT_FILE = 'exoplanets_merged.csv'

# ============================================================
# STEP 1 — NASA backbone
# ============================================================
log.info("=" * 60)
log.info("STEP 1 — NASA Exoplanet Archive (backbone)")
log.info("=" * 60)

r = requests.get(
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
    params={
        "query": (
            "select pl_name,pl_masse,pl_masseerr1,pl_masseerr2,"
            "pl_rade,pl_radeerr1,pl_insol,st_met,st_mass,st_rad,"
            "pl_orbper,pl_orbeccen "
            "from pscomppars"
        ),
        "format": "csv"
    },
    timeout=TIMEOUT
)
if r.status_code != 200:
    raise RuntimeError(f"NASA HTTP {r.status_code}")

df = pd.read_csv(io.StringIO(r.text))
log.info(f"  NASA raw: {len(df)} righe")

# Dropna su colonne obbligatorie
# st_met NON è in required_nasa: vogliamo i NaN passare fino a STEP 3
# dove PlanetS può riempirli. Viene filtrato solo alla fine.
required_nasa = ['pl_name','pl_masse','pl_masseerr1','pl_masseerr2',
                 'pl_rade','pl_radeerr1','pl_insol',
                 'st_mass','st_rad','pl_orbper']
df = df.dropna(subset=required_nasa)

# Filtri fisici
df = df[
    (df['pl_masse'] > 0) & (df['pl_masse'] < 4000) &
    (df['pl_rade']  > 0) & (df['pl_rade']  < 25)   &
    (df['pl_insol'] > 0) & (df['st_mass']  > 0)    &
    (df['st_rad']   > 0) & (df['pl_orbper'] > 0)
].copy()
log.info(f"  Dopo filtri fisici: {len(df)}")

# Errori relativi
df['pl_masseerr_rel'] = (df[['pl_masseerr1','pl_masseerr2']]
                         .abs().max(axis=1) / df['pl_masse'])
df['pl_radeerr_rel']  = df['pl_radeerr1'].abs() / df['pl_rade']

# Eccentricità: imputa 0 se mancante
# Salva flag PRIMA di fillna — dopo non si distingue zero vero da imputato
df['ecc_imputed'] = df['pl_orbeccen'].isna()
df['pl_orbeccen'] = df['pl_orbeccen'].fillna(0.0).clip(0.0, 0.99)
log.info(f"  Eccentricità imputata a 0: "
         f"{df['ecc_imputed'].sum()}/{len(df)}")

# Tracciamento provenienza colonne
df['source_masse'] = 'NASA'
df['source_rade']  = 'NASA'
df['source_met']   = 'NASA'

# Filtro M sin i
log.info("  Scaricando tabella ps per filtro M sin i...")
r2 = requests.get(
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
    params={"query":"select pl_name,tran_flag,discoverymethod from ps",
            "format":"csv"},
    timeout=TIMEOUT
)
df_ps = pd.read_csv(io.StringIO(r2.text))
ps_grp = df_ps.groupby('pl_name').agg(
    tran_flag_any=('tran_flag','max'),
    disc_methods=('discoverymethod',
                  lambda x: '|'.join(x.dropna().unique()))
).reset_index()
df = df.merge(ps_grp, on='pl_name', how='left')

def has_true_mass(row):
    if row['tran_flag_any'] == 1: return True
    if pd.isna(row['tran_flag_any']): return True
    if 'Radial Velocity' not in str(row.get('disc_methods','')): return True
    return False

df['true_mass'] = df.apply(has_true_mass, axis=1)
n_before = len(df)
df = df[df['true_mass']].copy()
log.info(f"  Filtro M sin i: rimossi {n_before-len(df)} → {len(df)} pianeti")

# gruppo e filtro errori spostati dopo TEPCat (FIX 2)

# ============================================================
# STEP 2 — TEPCat: migliora SOLO il raggio
# ============================================================
log.info("=" * 60)
log.info("STEP 2 — TEPCat (solo miglioramento raggio)")
log.info("=" * 60)

n_improved_rade = 0
try:
    r_tep = requests.get(
        "https://www.astro.keele.ac.uk/jkt/tepcat/allplanets-csv.csv",
        timeout=TIMEOUT
    )
    if r_tep.status_code != 200:
        raise RuntimeError(f"HTTP {r_tep.status_code}")

    df_tep = pd.read_csv(io.StringIO(r_tep.text), comment='#')
    log.info(f"  TEPCat raw: {len(df_tep)} righe")
    log.info(f"  Colonne disponibili: {list(df_tep.columns)}")

    # Normalizza nome pianeta per join robusto
    def norm_name(s):
        return str(s).replace(' ','').replace('-','').lower().strip()

    df['pl_name_norm']  = df['pl_name'].apply(norm_name)
    df_tep['pl_name_norm'] = df_tep[df_tep.columns[0]].apply(norm_name)

    # Identifica colonne raggio in TEPCat (nomi variano per versione)
    # Colonne TEPCat reali: R_b (raggio pianeta), errup.7/errdn.5 (errori asimmetrici)
    rade_col = None
    rade_err_up_col = None
    rade_err_dn_col = None
    for candidate in ['R_b', 'Rpl', 'Rp', 'R_pl', 'radius', 'Rplanet']:
        if candidate in df_tep.columns:
            rade_col = candidate
            break
    # Errori asimmetrici TEPCat — prendi il massimo come errore conservativo
    for candidate in ['errup.7', 'Rpl_err_up', 'Rp_err']:
        if candidate in df_tep.columns:
            rade_err_up_col = candidate
            break
    for candidate in ['errdn.5', 'Rpl_err_dn', 'Rp_err']:
        if candidate in df_tep.columns:
            rade_err_dn_col = candidate
            break
    rade_err_col = rade_err_up_col  # usato sotto

    if rade_col is None:
        raise RuntimeError("Colonna raggio non trovata in TEPCat")

    RJ_TO_RE = 11.209
    df_tep['pl_rade_tep'] = pd.to_numeric(
        df_tep[rade_col], errors='coerce') * RJ_TO_RE
    # Errore conservativo: max tra errore superiore e inferiore
    err_up = pd.to_numeric(df_tep[rade_err_up_col], errors='coerce') \
             if rade_err_up_col else pd.Series(np.nan, index=df_tep.index)
    err_dn = pd.to_numeric(df_tep[rade_err_dn_col], errors='coerce') \
             if rade_err_dn_col else pd.Series(np.nan, index=df_tep.index)
    df_tep['pl_radeerr_tep'] = pd.concat(
        [err_up, err_dn], axis=1).abs().max(axis=1) * RJ_TO_RE

    df_tep['pl_radeerr_rel_tep'] = (
        df_tep['pl_radeerr_tep'].abs() /
        df_tep['pl_rade_tep'].replace(0, np.nan)
    )

    # Deduplicazione TEPCat: per sistemi con più righe,
    # tieni quella con errore raggio minore
    df_tep_dedup = (
        df_tep[['pl_name_norm','pl_rade_tep','pl_radeerr_rel_tep']]
        .dropna(subset=['pl_rade_tep','pl_radeerr_rel_tep'])
        .sort_values('pl_radeerr_rel_tep')
        .drop_duplicates(subset='pl_name_norm', keep='first')
    )
    n_dup = len(df_tep) - len(df_tep_dedup)
    if n_dup > 0:
        log.info(f"  TEPCat: rimossi {n_dup} duplicati per nome")
    tep_idx = df_tep_dedup.set_index('pl_name_norm')[
        ['pl_rade_tep','pl_radeerr_rel_tep']
    ]

    # Sostituisci raggio solo se TEPCat è più preciso
    df = df.set_index('pl_name_norm')
    common = df.index.intersection(tep_idx.index)
    for name in common:
        tep_err = tep_idx.loc[name, 'pl_radeerr_rel_tep']
        nasa_err = df.loc[name, 'pl_radeerr_rel']
        if pd.isna(tep_err): continue
        if tep_err < nasa_err * 0.9:   # TEPCat almeno 10% più preciso
            df.loc[name, 'pl_rade']       = tep_idx.loc[name, 'pl_rade_tep']
            df.loc[name, 'pl_radeerr_rel'] = tep_err
            df.loc[name, 'source_rade']   = 'TEPCat'
            n_improved_rade += 1
    df = df.reset_index()

    log.info(f"  Raggi migliorati con TEPCat: {n_improved_rade}/{len(common)} comuni")

except Exception as e:
    log.warning(f"  TEPCat non disponibile o errore: {e}")
    log.warning("  Procedo con soli raggi NASA")
    if 'pl_name_norm' not in df.columns:
        df['pl_name_norm'] = df['pl_name'].apply(
            lambda s: str(s).replace(' ','').replace('-','').lower().strip())

# ============================================================
# STEP 2b — Gruppo planetario e filtro errori
# Fatto DOPO TEPCat perché il raggio potrebbe essere cambiato.
# ============================================================
log.info("Ricalcolo gruppo planetario dopo aggiornamento raggi TEPCat...")
bins   = [0, 1.5, 2.0, 4.0, 100]
labels = ['sub_terr', 'rocky_SE', 'sub_nep', 'gas']
df['group'] = pd.cut(df['pl_rade'], bins=bins, labels=labels)

piccoli = df['group'].isin(['sub_terr','rocky_SE'])
df = pd.concat([
    df[piccoli  & (df['pl_masseerr_rel'] < 0.35)],
    df[~piccoli & (df['pl_masseerr_rel'] < 0.30)]
]).copy()
log.info(f"  Dopo filtro errori: {len(df)}")
log.info(f"  Per gruppo:")
for g, n in df.groupby('group', observed=True).size().items():
    log.info(f"    {g}: {n}")

# ============================================================
# STEP 3 — PlanetS: migliora SOLO metallicità mancante
# ============================================================
log.info("=" * 60)
log.info("STEP 3 — PlanetS (solo metallicità mancante)")
log.info("=" * 60)

n_improved_met = 0
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    r_pl = requests.get(
        "https://dace.unige.ch/exoplanets/csv/",
        timeout=TIMEOUT,
        verify=False  # SSL cert non verificabile da WSL — dati non sensibili
    )
    if r_pl.status_code != 200:
        raise RuntimeError(f"HTTP {r_pl.status_code}")

    df_pl = pd.read_csv(io.StringIO(r_pl.text), comment='#')
    log.info(f"  PlanetS raw: {len(df_pl)} righe")

    def norm_name(s):
        return str(s).replace(' ','').replace('-','').lower().strip()

    if 'pl_name_norm' not in df.columns:
        df['pl_name_norm'] = df['pl_name'].apply(norm_name)

    # Trova colonna nome e metallicità in PlanetS
    name_col = df_pl.columns[0]
    df_pl['pl_name_norm'] = df_pl[name_col].apply(norm_name)

    met_col = None
    for candidate in ['Fe_H','FeH','feh','metallicity','[Fe/H]','st_met']:
        if candidate in df_pl.columns:
            met_col = candidate
            break

    if met_col is None:
        raise RuntimeError("Colonna metallicità non trovata in PlanetS")

    df_pl['st_met_pl'] = pd.to_numeric(df_pl[met_col], errors='coerce')
    pl_idx = df_pl.set_index('pl_name_norm')['st_met_pl'].dropna()

    # Aggiorna metallicità SOLO dove NASA ha NaN
    df = df.set_index('pl_name_norm')
    common_pl = df.index.intersection(pl_idx.index)
    for name in common_pl:
        if pd.isna(df.loc[name, 'st_met']):
            df.loc[name, 'st_met']      = pl_idx.loc[name]
            df.loc[name, 'source_met']  = 'PlanetS'
            n_improved_met += 1
    df = df.reset_index()

    log.info(f"  Metallicità integrate da PlanetS: {n_improved_met}")

except Exception as e:
    log.warning(f"  PlanetS non disponibile o errore: {e}")
    log.warning("  Procedo con sola metallicità NASA")

# ============================================================
# STEP 4 — Flag qualità + gruppo fuso per training
# ============================================================
log.info("=" * 60)
log.info("STEP 4 — Flag qualità e gruppo fuso")
log.info("=" * 60)

# Flag qualità alta: errori stringenti
df['flag_hq'] = (
    (df['pl_masseerr_rel'] < 0.20) &
    (df['pl_radeerr_rel']  < 0.10)
)
log.info(f"  Pianeti alta qualità (HQ): "
         f"{df['flag_hq'].sum()}/{len(df)}")
log.info(f"  HQ per gruppo:")
for g, sub in df.groupby('group', observed=True):
    log.info(f"    {g}: {sub['flag_hq'].sum()}/{len(sub)}")

# Gruppo fuso per training PySR sui pianeti rocciosi.
# Motivazione: la fisica della compressione di sfere di
# ferro/silicati è continua attraverso il confine 1.5 R_terra.
# Training congiunto → più dati per definire la pendenza.
# Validazione rimane separata per i due sottogruppi.
df['group_train'] = df['group'].astype(str)
df.loc[df['group'].isin(['sub_terr','rocky_SE']), 'group_train'] = 'rocky_combined'
log.info("\n  Distribuzione group_train:")
for g, n in df.groupby('group_train').size().items():
    log.info(f"    {g}: {n}")

# ============================================================
# STEP 5 — Salvataggio e report
# ============================================================
log.info("=" * 60)
log.info("STEP 5 — Salvataggio")
log.info("=" * 60)

# Ordina colonne per chiarezza
cols_out = [
    'pl_name','pl_masse','pl_masseerr_rel','pl_rade','pl_radeerr_rel',
    'pl_insol','st_met','st_mass','st_rad','pl_orbper','pl_orbeccen',
    'group','group_train','flag_hq',
    'source_masse','source_rade','source_met','ecc_imputed',
    'tran_flag_any','true_mass'
]
cols_out = [c for c in cols_out if c in df.columns]
df_out = df[cols_out].copy()
log.info(f"\n  Dataset prima filtro st_met: {len(df_out)} pianeti")
log.info(f"  Distribuzione per gruppo:")
for g, n in df_out.groupby('group', observed=True).size().items():
    log.info(f"    {g}: {n}")
log.info(f"\n  Provenienza raggi:")
for src, n in df_out.groupby('source_rade').size().items():
    log.info(f"    {src}: {n}")
log.info(f"  Provenienza metallicità:")
for src, n in df_out.groupby('source_met').size().items():
    log.info(f"    {src}: {n}")

n_met_nan = df_out['st_met'].isna().sum()
if n_met_nan > 0:
    log.warning(f"  {n_met_nan} pianeti ancora senza metallicità "
                f"dopo NASA+PlanetS — verranno scartati")
df_out = df_out.dropna(subset=['st_met'])
log.info(f"  Dataset dopo filtro st_met: {len(df_out)} pianeti")
df_out.to_csv(OUT_FILE, index=False)
with open(OUT_FILE,'rb') as f:
    md5 = hashlib.md5(f.read()).hexdigest()
log.info(f"\n  Salvato: {OUT_FILE}")
log.info(f"  MD5: {md5}")
log.info(f"  Timestamp: {datetime.datetime.now().isoformat()}")
log.info("\n  Per usarlo in run_finale.py:")
log.info("    DATA_CSV = 'exoplanets_merged.csv'")
log.info("    (usa group_train per allenare, group per validare)")
