import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, callbacks

# ===================== 1. Chargement des données =====================
# Remplace par ton fichier CSV
data = pd.read_csv("NVDA_10min_2years.csv")

# Conversion de la date
data["date"] = pd.to_datetime(data["date"])

# Colonnes utilisées comme features
required_cols = ["open", "high", "low", "close", "volume", "average", "barCount"]


# ===================== 2. Création des séquences =====================
def create_sequences(data, window=32):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(data[i + window, 3])  # colonne index 3 = "close"
    return np.array(X), np.array(y)


# Features
features = data[required_cols].values

# Normalisation
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Séquences
window_size = 32
X, y = create_sequences(features_scaled, window=window_size)

# ===================== 3. Split train/test =====================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# ===================== 4. Modèle Transformer =====================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.05):
    # Attention multi-têtes
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(
        inputs, inputs
    )
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)

    # Réseau feed-forward
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(inputs.shape[-1])(ff)
    x = layers.Dropout(dropout)(ff)
    return layers.LayerNormalization(epsilon=1e-6)(x + ff)


def build_transformer_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Empilement de 2 couches Transformer
    x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.05)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.05)

    # Pooling + Dense
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.05)(x)
    outputs = layers.Dense(1)(x)  # prédiction du "close"
    return tf.keras.Model(inputs, outputs)


model = build_transformer_model(X_train.shape[1:])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="mse")
model.summary()

# ===================== 5. Entraînement =====================
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1,
)

# ===================== 6. Backtest simple =====================
y_pred = model.predict(X_test).flatten()

plt.figure(figsize=(12, 5))
plt.plot(y_test, label="Vrai prix (close)")
plt.plot(y_pred, label="Prédictions")
plt.legend()
plt.title("Backtest Transformer - Prédiction du prix de clôture")
plt.show()

# ===================== 7. Exemple d’utilisation =====================
# Dernière fenêtre pour prédire le prochain prix
last_window = features_scaled[-window_size:]
last_window = last_window.reshape(1, window_size, len(required_cols))

predicted_price = model.predict(last_window)
print(f"Prochain prix prévu (close) : {predicted_price[0][0]:.2f}")
