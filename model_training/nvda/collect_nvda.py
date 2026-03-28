from ib_insync import IB, Stock, util
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

# ===================== Config =====================
HOST       = "127.0.0.1"
PORT       = 4002          # 4002 = paper Gateway / 4001 = live Gateway / 7497 = TWS paper
CLIENT_ID  = 1
SYMBOL     = "NVDA"
OUTPUT_CSV = f"{SYMBOL}_10min_2years.csv"
YEARS_BACK = 2
USE_RTH    = True          # True = heures de marché uniquement (recommandé pour TRADES)

# ===================== Connexion =====================
ib = IB()
ib.connect(HOST, PORT, clientId=CLIENT_ID)
log.info(f"Connecté à IB Gateway ({HOST}:{PORT})")

contract = Stock(SYMBOL, "SMART", "USD")
ib.qualifyContracts(contract)   # Résout le contrat (exchange, conid, etc.)

# ===================== Calcul des fenêtres =====================
# IB autorise "1 W" (1 semaine) avec des barres de 10 min → ~104 requêtes pour 2 ans
# au lieu de 730 avec "1 D"
end_date   = datetime.now()
start_date = end_date - timedelta(days=365 * YEARS_BACK)

# Générer les dates de fin de chaque fenêtre hebdomadaire (du plus récent au plus ancien)
windows = []
current = end_date
while current > start_date:
    windows.append(current)
    current -= timedelta(weeks=1)

log.info(f"Nombre de requêtes prévues : {len(windows)}")

# ===================== Récupération =====================
all_data = []
checkpoint_file = Path(f"{SYMBOL}_checkpoint.csv")

# Reprise depuis checkpoint si existant
if checkpoint_file.exists():
    existing = pd.read_csv(checkpoint_file, parse_dates=["date"])
    all_data.append(existing)
    last_fetched = existing["date"].min()
    windows = [w for w in windows if w > last_fetched]
    log.info(f"Reprise depuis checkpoint — {len(existing)} barres déjà récupérées")

for i, end_dt in enumerate(windows):
    log.info(f"[{i+1}/{len(windows)}] Récupération semaine se terminant le {end_dt.strftime('%Y-%m-%d')}")

    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_dt,
            durationStr="1 W",
            barSizeSetting="10 mins",
            whatToShow="TRADES",
            useRTH=USE_RTH,
            formatDate=1,
        )

        if bars:
            df = util.df(bars)
            all_data.append(df)
            log.info(f"  → {len(df)} barres récupérées")
        else:
            log.warning(f"  → Aucune donnée (weekend ou jour férié ?)")

    except Exception as e:
        log.error(f"  → Erreur : {e} — sauvegarde du checkpoint et arrêt")
        # Sauvegarde intermédiaire avant de planter
        if all_data:
            pd.concat(all_data).drop_duplicates(subset=["date"]).sort_values("date").to_csv(
                checkpoint_file, index=False
            )
        ib.disconnect()
        raise

    ib.sleep(0.5)   # Rate limit IB : ~50 req / 10 min, 0.5s largement suffisant

# ===================== Fusion & nettoyage =====================
df_all = (
    pd.concat(all_data)
    .drop_duplicates(subset=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)

# Supprimer les barres avec volume nul (barres fantômes hors marché)
before = len(df_all)
df_all = df_all[df_all["volume"] > 0]
log.info(f"Barres supprimées (volume=0) : {before - len(df_all)}")

# Vérification de la couverture temporelle
log.info(f"Période couverte : {df_all['date'].min()} → {df_all['date'].max()}")
log.info(f"Total barres : {len(df_all)}")

# ===================== Sauvegarde =====================
df_all.to_csv(OUTPUT_CSV, index=False)
log.info(f"Données sauvegardées dans {OUTPUT_CSV}")

# Nettoyage checkpoint
if checkpoint_file.exists():
    checkpoint_file.unlink()

ib.disconnect()
log.info("Déconnecté.")