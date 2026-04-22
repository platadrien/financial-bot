import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks, layers
from tools.positionnal_encoding import PositionalEncoding

load_dotenv()
ROOT_PATH = os.getenv("ROOT_PATH")
# ===================== 1. Chargement des données =====================
data = pd.read_csv(
    os.path.join(ROOT_PATH, "data", "csv", "nvda", "NVDA_10min_2years.csv")
)
data["date"] = pd.to_datetime(data["date"], utc=True)
data = data.sort_values("date").reset_index(drop=True)

# Colonnes utilisées comme features
required_cols = ["open", "high", "low", "close", "volume", "average", "barCount"]
features = data[required_cols].values

# ===================== 2. Split AVANT normalisation (évite le data leakage) =====================
split_idx = int(0.8 * len(features))

scaler = StandardScaler()
scaler.fit(features[:split_idx])  # fit uniquement sur le train
features_scaled = scaler.transform(features)  # transform sur tout


# ===================== 3. Création des séquences =====================
def create_sequences(data, window=32):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(data[i + window, 3])  # index 3 = "close" (normalisé)
    return np.array(X), np.array(y)


window_size = 32
X, y = create_sequences(features_scaled, window=window_size)

# Recalcul du split après création des séquences
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# ===================== 5. Modèle Transformer (corrigé) =====================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.05):
    # Multi-head attention + residual
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(
        inputs, inputs
    )
    attn = layers.Dropout(dropout)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs + attn)

    # Feed-forward + residual (bug corrigé : ff_out séparé de ff)
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff_out = layers.Dense(inputs.shape[-1])(ff)  # projection retour à la dim d'entrée
    ff_out = layers.Dropout(dropout)(ff_out)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff_out)

    return x


def build_transformer_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Positional encoding (d_model = nombre de features)
    x = PositionalEncoding(max_len=512, d_model=input_shape[-1])(inputs)

    # 2 blocs transformer
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.05)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.05)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.05)(x)
    outputs = layers.Dense(1)(x)

    return tf.keras.Model(inputs, outputs)


model = build_transformer_model(X_train.shape[1:])

# Huber loss plus robuste aux outliers financiers
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.Huber(delta=1.0),
)
model.summary()

# ===================== 6. Entraînement =====================
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, verbose=1
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)


# ===================== 7. Dénormalisation =====================
def denormalize_close(values_normalized):
    """Reconstruit un vecteur complet pour pouvoir appeler inverse_transform."""
    dummy = np.zeros((len(values_normalized), len(required_cols)))
    dummy[:, 3] = values_normalized  # index 3 = close
    return scaler.inverse_transform(dummy)[:, 3]


y_pred_norm = model.predict(X_test).flatten()

y_pred_real = denormalize_close(y_pred_norm)
y_test_real = denormalize_close(y_test)

# ===================== 8. Métriques financières =====================
# MAE en dollars
mae = np.mean(np.abs(y_pred_real - y_test_real))
print(f"\nMAE : ${mae:.2f}")

# Hit rate : % de fois où la direction est correcte
actual_direction = np.diff(y_test_real) > 0
predicted_direction = np.diff(y_pred_real) > 0
hit_rate = np.mean(actual_direction == predicted_direction)
print(f"Hit Rate (direction) : {hit_rate:.1%}")

# Sharpe ratio simplifié sur les rendements simulés
returns = np.where(predicted_direction, np.diff(y_test_real), -np.diff(y_test_real))
sharpe = (
    returns.mean() / (returns.std() + 1e-9) * np.sqrt(252 * 39)
)  # 39 barres/jour en 10min
print(f"Sharpe Ratio (approx) : {sharpe:.2f}")

# ===================== 9. Visualisation =====================
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Courbe prix
axes[0].plot(y_test_real, label="Prix réel (close)", alpha=0.8)
axes[0].plot(y_pred_real, label="Prédictions", alpha=0.8)
axes[0].set_title("Backtest Transformer — Prix de clôture NVDA (dénormalisé)")
axes[0].set_ylabel("Prix ($)")
axes[0].legend()

# Courbe de loss
axes[1].plot(history.history["loss"], label="Train loss")
axes[1].plot(history.history["val_loss"], label="Val loss")
axes[1].set_title("Courbe d'apprentissage")
axes[1].set_ylabel("Huber Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig("backtest_results.png", dpi=150)
plt.show()

# ===================== 10. Prédiction suivante (dénormalisée) =====================
last_window = features_scaled[-window_size:]
last_window = last_window.reshape(1, window_size, len(required_cols))

predicted_norm = model.predict(last_window).flatten()
predicted_price = denormalize_close(predicted_norm)[0]

print(f"\nProchain prix prévu (close) : ${predicted_price:.2f}")


# Sauvegarder le modèle
model_path = os.path.join(ROOT_PATH, "models", "nvda")
model.export(os.path.join(model_path, "nvda_transformer.keras"))

# Sauvegarder le scaler (indispensable — mêmes paramètres qu'à l'entraînement)
with open(os.path.join(model_path, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print("Modèle et scaler sauvegardés ✓")
