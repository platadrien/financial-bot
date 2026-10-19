import logging
import os
import signal as os_signal
import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests
from ib_insync import IB, MarketOrder, Stock, util

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")


def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# ===================== Config (via variables d'environnement) =====================
IB_HOST = os.getenv("IB_HOST", "ib-gateway")
IB_PORT = int(os.getenv("IB_PORT", "4004"))  # 4004 = paper (port interne gnzsnz)
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "10"))
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://model-api:5000")
SYMBOL = os.getenv("SYMBOL", "NVDA")

N_BARS = 33  # 32 barres de fenêtre + 1 barre précédente (pour calculer les rendements)
FEATURE_COLS = ["open", "high", "low", "close", "volume", "average", "barCount"]
BAR_SECONDS = 600  # barres de 10 min
BAR_DELAY = 5  # secondes d'attente après la clôture d'une barre
MAX_BAR_AGE = 900  # barre close plus vieille que 15 min => marché fermé / données figées

# --- Sécurité des ordres ---
ORDERS_ENABLED = env_bool("ORDERS_ENABLED", False)  # False = dry-run (logs uniquement)
MAX_POSITION = int(os.getenv("MAX_POSITION", "1"))  # nb max d'actions détenues (long-only)
ORDER_QTY = int(os.getenv("ORDER_QTY", "1"))  # taille d'un ordre d'achat
LIVE_CONFIRM = os.getenv("LIVE_CONFIRM", "")  # doit valoir "yes" pour trader un compte réel

STOP = False


def _handle_stop(signum, frame):
    global STOP
    STOP = True
    log.info("Signal d'arrêt reçu, fin de la boucle...")


os_signal.signal(os_signal.SIGTERM, _handle_stop)
os_signal.signal(os_signal.SIGINT, _handle_stop)


# ===================== Temps =====================
def bar_end_utc(bar_date) -> datetime:
    """Heure de fin (UTC) d'une barre IB dont `bar_date` est l'heure de début."""
    if bar_date.tzinfo is None:
        bar_date = bar_date.replace(tzinfo=NY)
    return bar_date.astimezone(timezone.utc) + timedelta(seconds=BAR_SECONDS)


def seconds_until_next_bar(now_ts: float | None = None) -> float:
    now_ts = time.time() if now_ts is None else now_ts
    next_boundary = (int(now_ts) // BAR_SECONDS + 1) * BAR_SECONDS
    return next_boundary + BAR_DELAY - now_ts


def can_trade_now() -> bool:
    """Ordres autorisés uniquement en séance régulière (lun-ven 9:30-15:55 New York)."""
    now = datetime.now(NY)
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= minutes < 15 * 60 + 55


def wait_until_next_bar(ib: IB):
    """Dort jusqu'à la prochaine clôture de barre (par paliers pour réagir à l'arrêt)."""
    remaining = seconds_until_next_bar()
    log.info(f"Prochaine barre dans {remaining:.0f}s")
    while remaining > 0 and not STOP:
        ib.sleep(min(remaining, 30))
        remaining = seconds_until_next_bar() if remaining > 30 else 0


# ===================== Connexion IB =====================
def ensure_connected(ib: IB) -> bool:
    if ib.isConnected():
        return True
    for attempt in range(5):
        if STOP:
            return False
        try:
            ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=20)
            log.info(f"Connecté à IB Gateway ({IB_HOST}:{IB_PORT})")
            return True
        except Exception as e:
            log.warning(f"Connexion {attempt + 1}/5 échouée : {e} — retry dans 10s")
            time.sleep(10)
    return False


def check_account_safety(ib: IB):
    """Refuse de passer des ordres sur un compte réel sans confirmation explicite."""
    accounts = ib.managedAccounts()
    is_paper = bool(accounts) and all(a.startswith("DU") for a in accounts)
    log.info(f"Compte(s) : {accounts} — {'PAPER' if is_paper else 'RÉEL'}")
    log.info(f"Ordres : {'ACTIVÉS' if ORDERS_ENABLED else 'désactivés (dry-run)'}")
    if ORDERS_ENABLED and not is_paper and LIVE_CONFIRM.lower() != "yes":
        log.error("Compte réel détecté : définis LIVE_CONFIRM=yes pour autoriser les ordres.")
        sys.exit(1)


# ===================== Données =====================
def fetch_closed_bars(ib: IB, contract):
    """
    Retourne (bars, last_bar_end) avec exactement N_BARS barres *closes*,
    ou None si les données sont indisponibles, incomplètes ou périmées.
    """
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="2 D",  # assez pour avoir 32 barres même en début de séance
        barSizeSetting="10 mins",
        whatToShow="TRADES",
        useRTH=True,  # même réglage que l'entraînement
        formatDate=1,
    )
    if not bars:
        log.warning("Aucune barre reçue depuis IB")
        return None

    now = datetime.now(timezone.utc)
    closed = [b for b in bars if bar_end_utc(b.date) <= now]  # exclut la barre en cours
    if len(closed) < N_BARS:
        log.warning(f"Pas assez de barres closes : {len(closed)}/{N_BARS}")
        return None

    last_end = bar_end_utc(closed[-1].date)
    age = (now - last_end).total_seconds()
    if age > MAX_BAR_AGE:
        log.info(f"Dernière barre close vieille de {age / 60:.0f} min — marché fermé, on passe")
        return None

    df = util.df(closed)[FEATURE_COLS].tail(N_BARS)
    if not df.notna().all().all() or (df["volume"] <= 0).any():
        log.warning("Barres invalides (NaN ou volume nul), on passe")
        return None

    return df.to_numpy(dtype=float).tolist(), last_end


# ===================== API modèle =====================
def wait_for_model_api() -> bool:
    log.info("Attente de l'API modèle...")
    for _ in range(30):
        if STOP:
            return False
        try:
            r = requests.get(f"{MODEL_API_URL}/health", timeout=5)
            if r.json().get("status") == "ok":
                log.info("API modèle prête ✓")
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def get_prediction(bars: list) -> dict | None:
    try:
        response = requests.post(f"{MODEL_API_URL}/predict", json={"bars": bars}, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        log.error("Impossible de joindre l'API modèle")
    except Exception as e:
        log.error(f"Erreur appel API : {e}")
    return None


# ===================== Trading =====================
def get_position(ib: IB) -> int:
    """Position réelle côté IB (source de vérité, survit aux redémarrages du bot)."""
    return int(sum(p.position for p in ib.positions() if p.contract.symbol == SYMBOL))


def has_open_order(ib: IB) -> bool:
    return any(t.contract.symbol == SYMBOL and not t.isDone() for t in ib.openTrades())


def handle_signal(ib: IB, contract, signal: str, price: float):
    """
    Long-only : BUY augmente la position jusqu'à MAX_POSITION,
    SELL clôture la position, jamais de short.
    """
    if signal not in ("BUY", "SELL"):
        log.info(f"⏸  HOLD — prix actuel ${price:.2f}")
        return

    position = get_position(ib)

    if has_open_order(ib):
        log.info(f"Ordre déjà en attente sur {SYMBOL}, on n'en ajoute pas")
        return

    if signal == "BUY":
        qty = min(ORDER_QTY, MAX_POSITION - position)
        if qty <= 0:
            log.info(f"📈 BUY ignoré — position {position} déjà au max ({MAX_POSITION})")
            return
        action = "BUY"
    else:
        if position <= 0:
            log.info("📉 SELL ignoré — pas de position à clôturer (long-only)")
            return
        qty, action = position, "SELL"

    log.info(f"{'📈' if action == 'BUY' else '📉'} {action} {qty} {SYMBOL} @ ~${price:.2f} (position actuelle : {position})")

    if not ORDERS_ENABLED:
        log.info("[DRY-RUN] ordre non envoyé (ORDERS_ENABLED=false)")
        return
    if not can_trade_now():
        log.info("Hors séance régulière — ordre non envoyé")
        return

    trade = ib.placeOrder(contract, MarketOrder(action, qty))
    ib.sleep(2)
    log.info(f"Ordre envoyé : statut={trade.orderStatus.status} filled={trade.orderStatus.filled}")


# ===================== Boucle principale =====================
def main():
    if not wait_for_model_api():
        raise RuntimeError("API modèle non disponible après 150s")

    ib = IB()
    if not ensure_connected(ib):
        raise ConnectionError("Impossible de se connecter à IB Gateway")

    contract = Stock(SYMBOL, "SMART", "USD")
    ib.qualifyContracts(contract)
    check_account_safety(ib)

    last_processed = None  # fin de la dernière barre traitée (évite les doublons)
    log.info(f"Démarrage — {SYMBOL}, une décision par barre de 10 min")

    while not STOP:
        wait_until_next_bar(ib)
        if STOP:
            break

        try:
            if not ensure_connected(ib):
                log.error("IB Gateway injoignable, nouvelle tentative à la prochaine barre")
                continue

            result = fetch_closed_bars(ib, contract)
            if result is None:
                continue
            bars, last_end = result

            if last_end == last_processed:
                log.info("Barre déjà traitée, on passe")
                continue

            prediction = get_prediction(bars)
            if prediction is None:
                continue
            last_processed = last_end

            log.info(
                f"Prix : ${prediction['current_price']:.2f} | "
                f"Prédit : ${prediction['predicted_price']:.3f} | "
                f"Variation : {prediction['change_pct']:+.3f}% | "
                f"Signal : {prediction['signal']}"
            )
            handle_signal(ib, contract, prediction["signal"], prediction["current_price"])

        except Exception:
            log.exception("Erreur dans l'itération, on continue à la prochaine barre")

    ib.disconnect()
    log.info("Bot arrêté proprement")


if __name__ == "__main__":
    main()
