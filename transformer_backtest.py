# fichier: transformer_backtest.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers

# ===================== 1. Chargement de tes données =====================
# Ici on suppose que tu charges un CSV ou un DataFrame déjà prêt
# data = pd.read_csv("tes_donnees.csv", parse_dates=True, index_col=0)

# Exemple : si tu as déjà "data" défini avant, commente la ligne ci-dessus

# Vérification colonnes nécessaires
required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'rsi', 'macd', 'bb_bbm']
for col in required_cols:
    if col not in data.columns:
        raise ValueError(f"Colonne manquante : {col}")

# ===================== 2. Normalisation =====================
features = data[required_cols]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Fonction pour créer des séquences
def create_sequences(data, window=32):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window][3])  # Index 3 = Close
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_features)

# ===================== 3. Modèle Transformer =====================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)

    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(inputs.shape[-1])(ff)
    x = layers.Dropout(dropout)(ff)
    return layers.LayerNormalization(epsilon=1e-6)(x + ff)

def build_transformer_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = transformer_encoder(inputs, head_size=64, num_heads=2, ff_dim=128)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1)(x)  # Prédiction du prix
    return tf.keras.Model(inputs, outputs)

model = build_transformer_model(X.shape[1:])
model.compile(optimizer="adam", loss="mse")
model.summary()

# ===================== 4. Entraînement =====================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("🚀 Entraînement du modèle...")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=1)

# ===================== 5. Prédictions =====================
print("📈 Prédictions...")
y_pred = model.predict(X_test)

# ===================== 6. Backtest =====================
threshold_pct = 0.005  # 0.5% de seuil

actions, returns = [], []

for i in range(len(y_pred)):
    current_price = y_test[i]
    predicted_price = y_pred[i][0]

    if predicted_price > current_price * (1 + threshold_pct):
        actions.append("BUY")
        returns.append((y_test[i] - current_price) / current_price)
    elif predicted_price < current_price * (1 - threshold_pct):
        actions.append("SELL")
        returns.append((current_price - y_test[i]) / current_price)
    else:
        actions.append("HOLD")
        returns.append(0)

results = pd.DataFrame({"Action": actions, "Return": returns})
results["Equity"] = results["Return"].cumsum()

# ===================== 7. Résultats =====================
total_return = results["Return"].sum()
mean_return = results["Return"].mean()
win_rate = (results["Return"] > 0).mean()
num_trades = (results["Action"] != "HOLD").sum()

print("\n===== Résultats Backtest =====")
print(f"Total Return: {total_return:.2%}")
print(f"Average Return per Trade: {mean_return:.2%}")
print(f"Win Rate: {win_rate:.2%}")
print(f"Number of Trades: {num_trades}")

# ===================== 8. Graphique =====================
plt.figure(figsize=(12, 5))
plt.plot(results["Equity"], label="Évolution capital")
plt.xlabel("Trade #")
plt.ylabel("Capital (cumulé)")
plt.title("Backtest - Évolution du capital")
plt.grid(True)
plt.legend()
plt.show()
