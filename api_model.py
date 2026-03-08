from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

# ===================== Chargement modèle & scaler =====================
MODEL_PATH  = os.getenv("MODEL_PATH",  "/app/models/nvda_transformer.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "/app/models/scaler.pkl")

log.info(f"Chargement du modèle : {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

log.info(f"Chargement du scaler : {SCALER_PATH}")
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

WINDOW_SIZE  = 32
N_FEATURES   = 7
FEATURE_COLS = ["open", "high", "low", "close", "volume", "average", "barCount"]

log.info("Modèle prêt ✓")

# ===================== Routes =====================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Attend un JSON :
    {
        "bars": [[open, high, low, close, volume, average, barCount], ...]
        // 32 lignes exactement
    }
    Retourne :
    {
        "current_price": 134.50,
        "predicted_price": 135.10,
        "change_pct": 0.45,
        "signal": "BUY"
    }
    """
    try:
        data = request.get_json()

        if "bars" not in data:
            return jsonify({"error": "Champ 'bars' manquant"}), 400

        bars = np.array(data["bars"], dtype=np.float32)

        if bars.shape != (WINDOW_SIZE, N_FEATURES):
            return jsonify({
                "error": f"Shape incorrecte : reçu {bars.shape}, attendu ({WINDOW_SIZE}, {N_FEATURES})"
            }), 400

        # Normalisation
        bars_scaled = scaler.transform(bars)
        X = bars_scaled.reshape(1, WINDOW_SIZE, N_FEATURES)

        # Prédiction
        pred_norm = model.predict(X, verbose=0)[0][0]

        # Dénormalisation
        dummy = np.zeros((1, N_FEATURES))
        dummy[0, 3] = pred_norm  # index 3 = close
        predicted_price = float(scaler.inverse_transform(dummy)[0, 3])

        # Prix actuel = dernière barre
        current_price = float(bars[-1, 3])
        change_pct    = (predicted_price - current_price) / current_price * 100

        # Signal
        if change_pct > 0.1:
            signal = "BUY"
        elif change_pct < -0.1:
            signal = "SELL"
        else:
            signal = "HOLD"

        return jsonify({
            "current_price":   round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "change_pct":      round(change_pct, 4),
            "signal":          signal,
        })

    except Exception as e:
        log.error(f"Erreur /predict : {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
