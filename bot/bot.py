import logging
import os
import time

import numpy as np
import requests
from ib_insync import IB, Stock, util

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

# ===================== Config (via variables d'environnement) =====================
IB_HOST = os.getenv("IB_HOST", "ib-gateway")  # nom du service docker
IB_PORT = int(os.getenv("IB_PORT", "4002"))  # 4002 = paper
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "10"))
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://model-api:5000")
SYMBOL = os.getenv("SYMBOL", "NVDA")
WINDOW_SIZE = 32
FEATURE_COLS = ["open", "high", "low", "close", "volume", "average", "barCount"]
POLL_INTERVAL = 600  # secondes entre chaque prédiction (10 min = 1 barre)


# ===================== Connexion IB =====================
def connect_ib() -> IB:
    ib = IB()
    for attempt in range(5):
        try:
            ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
            log.info(f"Connecté à IB Gateway ({IB_HOST}:{IB_PORT})")
            return ib
        except Exception as e:
            log.warning(f"Tentative {attempt + 1}/5 échouée : {e} — retry dans 10s")
            time.sleep(10)
    raise ConnectionError("Impossible de se connecter à IB Gateway")


# ===================== Récupération des données =====================
def fetch_last_bars(ib: IB, contract) -> np.ndarray | None:
    """Récupère les WINDOW_SIZE dernières barres de 10 min."""
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="6 H",
            barSizeSetting="10 mins",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            log.warning("Aucune barre reçue depuis IB")
            return None

        df = util.df(bars)[FEATURE_COLS].tail(WINDOW_SIZE)

        if len(df) < WINDOW_SIZE:
            log.warning(f"Pas assez de barres : {len(df)}/{WINDOW_SIZE}")
            return None

        return df.values.tolist()

    except Exception as e:
        log.error(f"Erreur récupération barres : {e}")
        return None


# ===================== Appel API modèle =====================
def get_prediction(bars: list) -> dict | None:
    try:
        response = requests.post(
            f"{MODEL_API_URL}/predict",
            json={"bars": bars},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        log.error("Impossible de joindre l'API modèle")
        return None
    except Exception as e:
        log.error(f"Erreur appel API : {e}")
        return None


# ===================== Logique de trading (paper) =====================
def handle_signal(ib: IB, contract, signal: str, current_price: float):
    """
    Version paper trading — log uniquement les signaux.
    Remplace par des ordres réels quand tu es prêt.
    """
    if signal == "BUY":
        log.info(f"📈 SIGNAL ACHAT  — prix actuel ${current_price:.2f}")
        # Exemple ordre réel (décommenter quand prêt) :
        # order = MarketOrder("BUY", 1)
        # trade = ib.placeOrder(contract, order)
        # log.info(f"Ordre envoyé : {trade}")

    elif signal == "SELL":
        log.info(f"📉 SIGNAL VENTE  — prix actuel ${current_price:.2f}")
        # order = MarketOrder("SELL", 1)
        # trade = ib.placeOrder(contract, order)

    else:
        log.info(f"⏸  HOLD          — prix actuel ${current_price:.2f}")


# ===================== Boucle principale =====================
def main():
    # Attendre que l'API modèle soit prête
    log.info("Attente de l'API modèle...")
    for _ in range(30):
        try:
            r = requests.get(f"{MODEL_API_URL}/health", timeout=5)
            if r.json().get("status") == "ok":
                log.info("API modèle prête ✓")
                break
        except Exception:
            pass
        time.sleep(5)
    else:
        raise RuntimeError("API modèle non disponible après 150s")

    ib = connect_ib()
    contract = Stock(SYMBOL, "SMART", "USD")
    ib.qualifyContracts(contract)

    log.info(
        f"Démarrage de la boucle de trading — {SYMBOL} toutes les {POLL_INTERVAL}s"
    )

    while True:
        log.info("— Nouvelle itération —")

        # 1. Récupérer les données
        bars = fetch_last_bars(ib, contract)
        if bars is None:
            log.warning("Données indisponibles, on attend...")
            ib.sleep(60)
            continue

        # 2. Demander une prédiction
        result = get_prediction(bars)
        if result is None:
            ib.sleep(60)
            continue

        log.info(
            f"Prix actuel : ${result['current_price']:.2f} | "
            f"Prédit : ${result['predicted_price']:.2f} | "
            f"Variation : {result['change_pct']:+.2f}% | "
            f"Signal : {result['signal']}"
        )

        # 3. Agir sur le signal
        handle_signal(ib, contract, result["signal"], result["current_price"])

        # 4. Attendre la prochaine barre
        ib.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
