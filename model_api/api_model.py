import json
import logging
import os
import threading

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from tools.features import FEATURE_NAMES, N_FEATURES, N_RAW, apply_scaler, compute_features
from tools.positionnal_encoding import PositionalEncoding

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

# ===================== Chargement modèle & métadonnées =====================
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/nvda_transformer.keras")
META_PATH = os.getenv("META_PATH", "/app/models/meta.json")

log.info(f"Chargement du modèle : {MODEL_PATH}")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"PositionalEncoding": PositionalEncoding},
)

log.info(f"Chargement des métadonnées : {META_PATH}")
with open(META_PATH) as f:
    meta = json.load(f)

# Garde-fou : le modèle doit avoir été entraîné avec les mêmes features que le code actuel
if meta["n_raw_bars"] != N_RAW or meta["feature_names"] != FEATURE_NAMES:
    raise RuntimeError(
        "meta.json incompatible avec tools/features.py — ré-entraîne le modèle "
        "ou aligne les versions du code."
    )

SCALER_MEAN = np.array(meta["scaler_mean"], dtype=np.float64)
SCALER_SCALE = np.array(meta["scaler_scale"], dtype=np.float64)
TARGET_STD = float(meta["target_std"])
# Seuil de signal en points de base (surchargeable sans ré-entraîner)
THRESHOLD_BPS = float(os.getenv("THRESHOLD_BPS", meta["threshold_bps"]))

_predict_lock = threading.Lock()  # Keras n'aime pas les appels concurrents

log.info(f"Modèle prêt ✓ (seuil de signal : ±{THRESHOLD_BPS:g} bps)")


# ===================== Routes =====================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_version": meta.get("version")})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Attend un JSON :
    {
        "bars": [[open, high, low, close, volume, average, barCount], ...]
        // 33 lignes exactement : 32 de fenêtre + 1 barre précédente, barres CLOSES,
        // de la plus ancienne à la plus récente
    }
    Retourne :
    {
        "current_price": 134.50,
        "predicted_price": 134.5134,
        "change_pct": 0.01,             // rendement prédit de la barre suivante, en %
        "predicted_return_bps": 1.0,    // idem en points de base
        "signal": "HOLD"
    }
    """
    try:
        data = request.get_json(silent=True)
        if not data or "bars" not in data:
            return jsonify({"error": "Champ 'bars' manquant"}), 400

        bars = np.array(data["bars"], dtype=np.float64)

        # Features stationnaires (même code que l'entraînement) puis standardisation
        feats = compute_features(bars[np.newaxis])  # (1, 32, 7) ; lève ValueError si invalide
        X = apply_scaler(feats, SCALER_MEAN, SCALER_SCALE).astype(np.float32)
        assert X.shape[-1] == N_FEATURES

        with _predict_lock:
            pred_scaled = float(model(X, training=False).numpy()[0, 0])

        predicted_return = pred_scaled * TARGET_STD  # fraction (0.0001 = 1 bp)
        current_price = float(bars[-1, 3])
        predicted_price = current_price * (1 + predicted_return)
        return_bps = predicted_return * 1e4

        if return_bps > THRESHOLD_BPS:
            signal = "BUY"
        elif return_bps < -THRESHOLD_BPS:
            signal = "SELL"
        else:
            signal = "HOLD"

        return jsonify(
            {
                "current_price": round(current_price, 2),
                "predicted_price": round(predicted_price, 4),
                "change_pct": round(predicted_return * 100, 4),
                "predicted_return_bps": round(return_bps, 3),
                "signal": signal,
            }
        )

    except ValueError as e:  # barres invalides : erreur du client, pas du serveur
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        log.error(f"Erreur /predict : {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
